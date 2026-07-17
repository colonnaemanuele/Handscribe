import torch
import torch.nn as nn
from handscribe.modules.sync_batchnorm.batchnorm import convert_model

from slt_network_multi import SLTModel

class SLTModelParallel(SLTModel):
    def __init__(self, *args, **kwargs):
        super(SLTModelParallel, self).__init__(*args, **kwargs)
        self.is_model_parallel = True

        # Se è disponibile una sola GPU, degradiamo a single-device (dev1 == dev0)
        # così la stessa configurazione gira anche senza la seconda GPU.
        n_gpus = torch.cuda.device_count()
        self.dev0 = torch.device('cuda:0')
        self.dev1 = torch.device('cuda:1' if n_gpus >= 2 else 'cuda:0')
        if n_gpus < 2:
            print(f"[ModelParallel] Solo {n_gpus} GPU disponibili: fallback single-device su {self.dev0}")

        print(f"[ModelParallel] Moving Video Encoder to {self.dev0}")
        print(f"[ModelParallel] Moving MBART & Decoders to {self.dev1}")

        # --- PARTITIONING DEL MODELLO ---
        
        # GRUPPO 1: Video Encoder -> GPU 0
        # Conv2d (SlowFast) e Conv1d (Fusion) rimangono sulla prima GPU
        self.conv2d.to(self.dev0)
        self.conv1d.to(self.dev0)
        
        # GRUPPO 2: Text Decoder & Heads -> GPU 1
        # mBART è molto pesante, deve stare da solo
        print('[Memory Opt] Enabling gradient checkpointing for mBART model')
        if hasattr(self.mbart_model, 'gradient_checkpointing_enable'):
            self.mbart_model.gradient_checkpointing_enable()
        self.mbart_model.to(self.dev1)
        
        # I decoder e il modello temporale (LSTM) usano l'output visivo per generare testo
        self.decoders.to(self.dev1)
        self.temporal_model.to(self.dev1)
        
        # Se esiste la FC layer in conv1d, controlliamo dove serve. 
        # Di solito serve per loss ausiliarie, la mettiamo su GPU 0 o 1?
        # Dal codice originale sembra usata dentro conv1d, quindi GPU 0.
        if hasattr(self.conv1d, 'fc') and self.conv1d.fc is not None:
             self.conv1d.fc.to(self.dev0)

    def forward(self, x, len_x, gt_sentences=None):
        # 1. INPUT -> GPU 0
        # Assicuriamoci che l'input video sia sulla GPU dell'encoder
        x = x.to(self.dev0)
        len_x = len_x.to(self.dev0)

        # 2. VIDEO ENCODING (GPU 0)
        if len(x.shape) == 5:
            # Permute e conv2d su GPU 0
            framewise = self.conv2d(x.permute(0, 2, 1, 3, 4))
        else:
            framewise = x
        
        # conv1d su GPU 0
        conv1d_outputs = self.conv1d(framewise, len_x)
        lgt = conv1d_outputs['feat_len'] # Lunghezze (Tensor)

        # 3. BRIDGE: SPOSTAMENTO DATI (GPU 0 -> GPU 1)
        # Dobbiamo spostare le feature visive e le lunghezze sulla GPU 1
        lgt = lgt.to(self.dev1)
        
        # Gestione visual_feat (che è una lista o tensore)
        visual_feat_list = conv1d_outputs['visual_feat']
        
        # Normalizzazione lista come nel codice originale
        if not isinstance(visual_feat_list, list):
            visual_feat_list = [visual_feat_list] * 3
        elif len(visual_feat_list) < 3:
            visual_feat_list = visual_feat_list + [visual_feat_list[0]] * (3 - len(visual_feat_list))
            
        # Spostiamo ogni feature della lista su GPU 1
        visual_feat_list = [feat.to(self.dev1) for feat in visual_feat_list]

        # 4. TEXT DECODING (GPU 1)
        decoder_outputs = []
        tm_outputs = []
        
        loop_range = range(3) if self.training else range(1)
        
        for i in loop_range:
            visual_feat = visual_feat_list[i]
            
            # Temporal Model (LSTM) su GPU 1
            tm_output = self.temporal_model[i](visual_feat, lgt)
            tm_outputs.append(tm_output)
            
            # Decoder (mBART) su GPU 1
            # Nota: gt_sentences sono stringhe, non serve .to(device)
            decoder_output = self.decoders[i](tm_output['predictions'], gt_sentences, inference=not self.training, feat_len=lgt)
            decoder_outputs.append(decoder_output)

        # 5. INFERENCE LOGIC (GPU 1)
        pred = None
        if not self.training:
            logits_for_decoding = decoder_outputs[0]
            # Decoders sono su GPU 1, quindi tutto ok
            if isinstance(logits_for_decoding, torch.Tensor) and logits_for_decoding.dim() == 2:
                pred = self.decoders[0].decode_logits(logits_for_decoding)
            elif hasattr(logits_for_decoding, 'logits'):
                pred = self.decoders[0].decode_logits(logits_for_decoding.logits)

        # Ritorna dizionario
        # Nota: conv_logits sono rimasti su GPU 0. sequence_logits sono su GPU 1.
        return {
            "feat_len": lgt, # Su GPU 1
            "conv_logits": conv1d_outputs["conv_logits"], # Su GPU 0 (Attenzione alla Loss!)
            "sequence_logits": decoder_outputs, # Su GPU 1
            "recognized_sents": pred,
            "tm_outputs": tm_outputs
        }

    def criterion_calculation(self, ret_dict, label=None, label_lgt=None, gt_sentences=None):
        # Le loss testuali usano come target `mbart_output.text_labels` (token mBART già su GPU 1),
        # quindi non serve spostare il `label` a vocabolario custom.
        if not self.loss_weights or not isinstance(self.loss_weights, dict):
            raise ValueError("loss_weights must be a non-empty dictionary.")

        total_loss = torch.zeros((), device=self.dev1)
        loss_components = {}

        sequence_logits = ret_dict["sequence_logits"]  # su GPU 1

        def ce_loss(mbart_output):
            loss = self.loss['CrossEntropy'](mbart_output, mbart_output.text_labels)
            if loss.dim() > 0:
                loss = loss.mean()
            return loss.to(self.dev1)

        def rouge_loss(mbart_output):
            return self.loss['CrossRougeEntropy'](mbart_output, mbart_output.text_labels).to(self.dev1)

        for k, weight in self.loss_weights.items():
            if k in ['Slow', 'Fast']:
                i = 1 if k == 'Slow' else 2
                if i < len(sequence_logits) and 'CrossEntropy' in self.loss_weights:
                    loss_val = ce_loss(sequence_logits[i]) * weight
                    loss_components[f'{k}_CrossEntropy'] = loss_val
                    total_loss = total_loss + loss_val

            elif k == 'CrossEntropy':
                loss_val = ce_loss(sequence_logits[0]) * weight
                loss_components['CrossEntropy_Main'] = loss_val
                total_loss = total_loss + loss_val

            elif k == 'CrossRougeEntropy':
                loss_val = rouge_loss(sequence_logits[0]) * weight
                loss_components['CrossRougeEntropy'] = loss_val
                total_loss = total_loss + loss_val

        return total_loss, loss_components