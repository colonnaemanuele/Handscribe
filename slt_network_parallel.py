import torch
import torch.nn as nn
from handscribe.modules.sync_batchnorm.batchnorm import convert_model

from slt_network_multi import SLTModel

class SLTModelParallel(SLTModel):
    def __init__(self, *args, **kwargs):
        super(SLTModelParallel, self).__init__(*args, **kwargs)
        self.dev0 = torch.device('cuda:0')
        self.dev1 = torch.device('cuda:1')
        
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
            decoder_output = self.decoders[i](tm_output['predictions'], gt_sentences, inference=not self.training)
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
        # Override per gestire label su device diversi
        
        if label is None: raise ValueError("label cannot be None.")
        
        # I logits principali (mBART) sono su GPU 1
        # Quindi spostiamo le label su GPU 1 per calcolare la loss testuale
        label = label.to(self.dev1)
        
        # Le loss interne usano self.loss['CrossEntropy'] ecc.
        # Assicuriamoci che i pesi delle loss siano corretti o stateless
        
        # Eseguiamo il calcolo usando il metodo originale, ma con label spostata
        # ATTENZIONE: conv_logits sono su GPU 0. Se hai loss ausiliarie su conv_logits, 
        # questo metodo base fallirà perché label è su GPU 1 e conv_logits su GPU 0.
        
        # Soluzione Custom per Multi-Device Loss:
        total_loss = torch.tensor(0.0).to(self.dev1)
        loss_components = {}
        
        # Estrai output
        sequence_logits = ret_dict["sequence_logits"] # GPU 1
        
        # Helper interno bindato al device corretto
        def compute_loss_on_device(logits, targets, loss_fn_key):
            # Porta targets dove sono i logits
            target_dev = logits.device
            t = targets.to(target_dev)
            
            # Calcola loss
            if loss_fn_key == 'CrossEntropy':
                loss = self.loss['CrossEntropy'](logits, t)
                if loss.dim() > 0: loss = loss.mean()
                return loss
            elif loss_fn_key == 'CrossRougeEntropy':
                return self.loss['CrossRougeEntropy'](logits, t)
            return torch.tensor(0.0).to(target_dev)

        # Ciclo sui pesi (copiato e adattato dal tuo slt_network_multi.py)
        for k, weight in self.loss_weights.items():
            if k in ['Slow', 'Fast']:
                i = 1 if k == 'Slow' else 2
                if 'CrossEntropy' in self.loss_weights:
                    # sequence_logits[i] è su GPU 1
                    loss = compute_loss_on_device(sequence_logits[i], label, 'CrossEntropy')
                    loss_val = loss.to(self.dev1) * weight
                    loss_components[f'{k}_CrossEntropy'] = loss_val
                    total_loss += loss_val

            elif k == 'CrossEntropy':
                # Main output (GPU 1)
                loss = compute_loss_on_device(sequence_logits[0], label, 'CrossEntropy')
                loss_val = loss.to(self.dev1) * self.loss_weights['CrossEntropy']
                loss_components['CrossEntropy_Main'] = loss_val
                total_loss += loss_val

            elif k == 'CrossRougeEntropy':
                # Main output (GPU 1)
                loss = compute_loss_on_device(sequence_logits[0], label, 'CrossRougeEntropy')
                loss_val = loss.to(self.dev1) * self.loss_weights['CrossRougeEntropy']
                loss_components['CrossRougeEntropy'] = loss_val
                total_loss += loss_val

        # Se avessi loss su conv_logits (CTC?), dovresti gestirle qui spostando label su GPU 0
        if "conv_logits" in ret_dict and ret_dict["conv_logits"] is not None:
             # Esempio ipotetico se usassi CTC loss sui conv_logits (che sono su GPU 0)
             pass

        return total_loss, loss_components