# handscribe/slt_network_multi.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from handscribe.modules.criterions import CrossEntropy, CrossRougeEntropy
from handscribe.utils.decoder import SingleDecoderHead
from handscribe.modules import BiLSTMLayer, TemporalSlowFastFuse
from handscribe.utils.decoder_utils import load_mbart, MBART_MODEL_ID
from handscribe.modules.backbones import ViTBackbone, SwinBackbone
import handscribe.slowfast_modules.slowfast as slowfast

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x


class NormLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(NormLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
    def forward(self, x):
        return torch.matmul(x, F.normalize(self.weight, dim=0))


class SLTModel(nn.Module):
    def __init__(
        self,
        num_classes,
        c2d_type,
        conv_type,
        load_pkl,
        slowfast_config,
        slowfast_args=None,
        use_bn=False,
        hidden_size=1024,
        loss_weights=None,
        weight_norm=True,
        language_model=None,
        decoder_args=None,
        backbone='slowfast',
        backbone_model_name=None,
        img_size=224,
        decoder_mode='ensemble',
    ):
        super(SLTModel, self).__init__()
        self.loss = dict()
        self.is_model_parallel = False
        self.num_classes = num_classes
        self.loss_weights = loss_weights if loss_weights else {}
        self.backbone_type = backbone  # <-- salva il tipo

        if decoder_mode not in ('ensemble', 'single'):
            raise ValueError(f"decoder_mode '{decoder_mode}' non supportata. Usa: 'ensemble' o 'single'.")
        self.decoder_mode = decoder_mode

        # === backbone select ===
        if backbone == 'slowfast':
            self.conv2d = getattr(slowfast, c2d_type)(
                slowfast_config=slowfast_config,
                slowfast_args=slowfast_args,
                load_pkl=load_pkl,
                multi=True
            )
        elif backbone == 'vit':
            model_name = backbone_model_name or 'vit_base_patch16_224'
            self.conv2d = ViTBackbone(model_name=model_name)
        elif backbone == 'swin':
            model_name = backbone_model_name or 'swin_base_patch4_window7_224'
            self.conv2d = SwinBackbone(model_name=model_name)
        else:
            raise ValueError(f"Backbone '{backbone}' non supportata. Usa: 'slowfast', 'vit', o 'swin'.")
        # === end backbone ===

        self.conv1d = TemporalSlowFastFuse(
            fast_input_size=256,
            slow_input_size=2048,
            hidden_size=hidden_size,
            conv_type=conv_type,
            use_bn=use_bn,
            num_classes=num_classes
        )

        self.mbart_model, self.mbart_tokenizer, self.lang_codes = load_mbart(MBART_MODEL_ID, language_model)

        if self.decoder_mode == 'ensemble':
            # Ensemble: un decoder + temporal model indipendenti per ciascun path (main/slow/fast)
            self.decoders = nn.ModuleList([
                SingleDecoderHead(
                    mbart_model=self.mbart_model,
                    mbart_tokenizer=self.mbart_tokenizer,
                    lang_code=self.lang_codes[1],
                    **(decoder_args or {})
                ) for _ in range(3)
            ])
            self.temporal_model = nn.ModuleList([
                BiLSTMLayer(rnn_type='LSTM', input_size=hidden_size, hidden_size=hidden_size, num_layers=2, bidirectional=True)
                for _ in range(3)
            ])
        else:
            # Ablation: singolo decoder + singolo temporal model sulle feature combinate
            self.decoder = SingleDecoderHead(
                mbart_model=self.mbart_model,
                mbart_tokenizer=self.mbart_tokenizer,
                lang_code=self.lang_codes[1],
                **(decoder_args or {})
            )
            self.temporal_model = BiLSTMLayer(rnn_type='LSTM', input_size=hidden_size, hidden_size=hidden_size, num_layers=2, bidirectional=True)

        self.criterion_init()

        # conv1d.fc espone sempre 3 teste (main/slow/fast), usate anche in modalita' 'single'
        # per le conv_logits ausiliarie restituite da TemporalSlowFastFuse.
        if weight_norm:
            self.conv1d.fc = nn.ModuleList([NormLinear(hidden_size, self.num_classes) for _ in range(3)])
        else:
            self.conv1d.fc = nn.ModuleList([nn.Linear(hidden_size, self.num_classes) for _ in range(3)])

        self.register_backward_hook(self.backward_hook)

    def backward_hook(self, module, grad_input, grad_output):
        for g in grad_input:
            if g is not None:
                g[g != g] = 0
        for g in grad_output:
            if g is not None and hasattr(g, 'grad_fn'):
                if torch.isnan(g).any() or torch.isinf(g).any():
                    print("Warning: NaN or Inf detected in gradient outputs")

    def masked_bn(self, inputs, len_x):
        # Non usato con ViT/Swin → puoi lasciarlo, ma non verrà chiamato
        def pad(tensor, length):
            return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_()])
        x = torch.cat([inputs[len_x[0] * idx:len_x[0] * idx + lgt] for idx, lgt in enumerate(len_x)])
        x = self.conv2d(x)
        x = torch.cat([pad(x[sum(len_x[:idx]):sum(len_x[:idx + 1])], len_x[0])
                       for idx, lgt in enumerate(len_x)])
        return x

    def forward(self, x, len_x, gt_sentences=None):
        if self.backbone_type == 'slowfast':
            if len(x.shape) == 5:
                framewise = self.conv2d(x.permute(0, 2, 1, 3, 4))
            else:
                framewise = x
        else:
            framewise = self.conv2d(x)
        conv1d_outputs = self.conv1d(framewise, len_x)
        lgt = conv1d_outputs['feat_len']

        if self.decoder_mode == 'ensemble':
            visual_feat_list = conv1d_outputs['visual_feat']
            if not isinstance(visual_feat_list, list):
                visual_feat_list = [visual_feat_list] * 3
            elif len(visual_feat_list) < 3:
                visual_feat_list = visual_feat_list + [visual_feat_list[0]] * (3 - len(visual_feat_list))

            loop_range = range(3) if self.training else range(1)
            decoder_outputs = []
            tm_outputs = []

            for i in loop_range:
                visual_feat = visual_feat_list[i]
                tm_output = self.temporal_model[i](visual_feat, lgt)
                tm_outputs.append(tm_output)
                decoder_output = self.decoders[i](tm_output['predictions'], gt_sentences, inference=not self.training, feat_len=lgt)
                decoder_outputs.append(decoder_output)

            pred = None
            if not self.training:
                logits_for_decoding = decoder_outputs[0]
                if isinstance(logits_for_decoding, torch.Tensor) and logits_for_decoding.dim() == 2:
                    pred = self.decoders[0].decode_logits(logits_for_decoding)
                elif hasattr(logits_for_decoding, 'logits'):
                    pred = self.decoders[0].decode_logits(logits_for_decoding.logits)

            sequence_logits = decoder_outputs
            tm_output_result = tm_outputs
        else:
            # Get combined visual features
            visual_feat = conv1d_outputs['visual_feat']
            if isinstance(visual_feat, list):
                visual_feat = visual_feat[0]  # Take the combined features

            # Process through temporal model
            tm_output = self.temporal_model(visual_feat, lgt)

            # Process through decoder
            decoder_output = self.decoder(tm_output['predictions'], gt_sentences, inference=not self.training, feat_len=lgt)

            # Generate predictions during inference
            pred = None
            if not self.training:
                if isinstance(decoder_output, torch.Tensor) and decoder_output.dim() == 2:
                    pred = self.decoder.decode_logits(decoder_output)
                elif hasattr(decoder_output, 'logits'):
                    pred = self.decoder.decode_logits(decoder_output.logits)

            sequence_logits = decoder_output
            tm_output_result = tm_output

        return {
            "feat_len": lgt,
            "conv_logits": conv1d_outputs["conv_logits"],
            "sequence_logits": sequence_logits,
            "recognized_sents": pred,
            "tm_outputs": tm_output_result
        }

    @staticmethod
    def _output_device(mbart_output):
        """Device su cui vivono i logit/loss del decoder (unica sorgente per total_loss)."""
        if hasattr(mbart_output, 'logits') and mbart_output.logits is not None:
            return mbart_output.logits.device
        if hasattr(mbart_output, 'loss') and mbart_output.loss is not None:
            return mbart_output.loss.device
        return mbart_output.device

    def _ce_loss(self, mbart_output):
        # Il target sono gli STESSI token mBART usati dal decoder (mbart_output.text_labels),
        # non più il vocabolario custom del dataloader.
        loss = self.loss['CrossEntropy'](mbart_output, mbart_output.text_labels)
        if loss.dim() > 0:
            loss = loss.mean()
        return loss

    def _rouge_loss(self, mbart_output):
        return self.loss['CrossRougeEntropy'](mbart_output, mbart_output.text_labels)

    def criterion_calculation(self, ret_dict, label=None, label_lgt=None, gt_sentences=None):
        """
        Calcola la loss totale e la scomposizione per componente.

        Il target è sempre la tokenizzazione mBART di gt_sentences (attaccata come
        `text_labels` al Seq2SeqLMOutput dal decoder), così logits e target sono coerenti.
        L'argomento `label` (tokenizzazione a vocabolario custom) è mantenuto per
        compatibilità con la modalità gloss/feature ma non viene usato per le loss testuali.

        Returns:
            Tuple (total_loss, loss_components_dict)
        """
        if not self.loss_weights or not isinstance(self.loss_weights, dict):
            raise ValueError("loss_weights must be a non-empty dictionary.")

        loss_components = {}

        if self.decoder_mode == 'ensemble':
            sequence_logits = ret_dict["sequence_logits"]
            mbart_output = sequence_logits[0]
            device = self._output_device(mbart_output)
            total_loss = torch.zeros((), device=device)

            for k, weight in self.loss_weights.items():
                if k in ['Slow', 'Fast']:
                    i = 1 if k == 'Slow' else 2
                    if i < len(sequence_logits) and 'CrossEntropy' in self.loss_weights:
                        loss_value = self._ce_loss(sequence_logits[i]) * weight
                        loss_components[f'{k}_CrossEntropy'] = loss_value
                        total_loss = total_loss + loss_value

                elif k == 'CrossEntropy':
                    loss_value = self._ce_loss(mbart_output) * weight
                    loss_components['CrossEntropy_Main'] = loss_value
                    total_loss = total_loss + loss_value

                elif k == 'CrossRougeEntropy':
                    loss_value = self._rouge_loss(mbart_output) * weight
                    loss_components['CrossRougeEntropy'] = loss_value
                    total_loss = total_loss + loss_value
        else:
            mbart_output = ret_dict["sequence_logits"]
            device = self._output_device(mbart_output)
            total_loss = torch.zeros((), device=device)

            if 'CrossEntropy' in self.loss_weights:
                loss_value = self._ce_loss(mbart_output) * self.loss_weights['CrossEntropy']
                loss_components['CrossEntropy'] = loss_value
                total_loss = total_loss + loss_value

            if 'CrossRougeEntropy' in self.loss_weights:
                loss_value = self._rouge_loss(mbart_output) * self.loss_weights['CrossRougeEntropy']
                loss_components['CrossRougeEntropy'] = loss_value
                total_loss = total_loss + loss_value

        return total_loss, loss_components

    def criterion_init(self):
        # I labels usano -100 come indice di padding (coerente con il masking fatto nel decoder
        # e con la loss interna di mBART).
        self.loss['CrossEntropy'] = CrossEntropy(ignore_index=-100, reduction='none')
        self.loss['CrossRougeEntropy'] = CrossRougeEntropy(rouge_weight=1.0, label_smoothing=0.1, ignore_index=-100)
        return self.loss
