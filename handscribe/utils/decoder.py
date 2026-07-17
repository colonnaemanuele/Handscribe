import torch
import torch.nn as nn
from handscribe.utils.decoder_utils import load_mbart


class GFProjectionLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, mbart_dim=1024, nhead=None, num_layers=None):
        super().__init__()
        self.feature_multiplier = 2
        self.mlp = nn.Sequential(
            # Use a LayerNorm to normalize the input given that we map visual features to text features
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, self.feature_multiplier * hidden_dim),
            nn.ReLU(),
            nn.Linear(self.feature_multiplier * hidden_dim, mbart_dim),
        )

    def forward(self, x):
        # Feature expected to have shape (batch_size, num_features, input_dim)
        # Without .transpose(0,1) we have (num_features, batch_size, input_dim)
        return self.mlp(x).transpose(0, 1)
    
class GFTemporalSemanticProjectionLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, mbart_dim=1024, nhead=8, num_layers=1):
        super().__init__()
        self.temporal_adapter = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=hidden_dim),
            num_layers=num_layers
        )
        self.mlp = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 2 * hidden_dim),
            nn.ReLU(),
            nn.Linear(2 * hidden_dim, mbart_dim)
        )
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        # x: (T, B, D)
        x = x.transpose(0, 1)  # → (B, T, D)
        x = self.temporal_adapter(x.transpose(0, 1)).transpose(0, 1)
        return self.scale * self.mlp(x).transpose(0, 1)

class GFResidualProjectionLayer(nn.Module):
    """
    Proiezione temporale + adattamento semantico per passare da feature visuali (SlowFast)
    a feature linguistiche compatibili con mBART.
    """

    def __init__(self, input_dim, hidden_dim=512, mbart_dim=1024, nhead=8, num_layers=1, dropout=0.1):
        super().__init__()
        # Adattatore temporale
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )
        self.temporal_adapter = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Proiezione nello spazio linguistico
        self.mlp = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, mbart_dim),
        )
        self.residual = (input_dim == mbart_dim)
        self.scale = nn.Parameter(torch.ones(1) * 0.7)  # per fare in modo che all’inizio non sia troppo grande

    def forward(self, x):
        """
        x: (batch_size, num_frames, input_dim)
        ritorna: (num_frames, batch_size, mbart_dim)
        """
        x = self.temporal_adapter(x)  # (B, T, D)
        proj = self.mlp(x)

        if self.residual:
            proj = proj + x
        proj = self.scale * proj
        return proj.transpose(0, 1)


class SingleDecoderHead(nn.Module):
    def __init__(
        self,
        mbart_model=None,
        mbart_tokenizer=None,
        mbart_model_id="large-cc",
        mbart_lang="German",
        lang_code=None,
        projector_type=None,  # GFProjectionLayer
        projector_args={},
    ):
        super().__init__()

        self.mbart_model, self.mbart_tokenizer = None, None

        if mbart_model and mbart_tokenizer:
            self.mbart_model = mbart_model
            self.mbart_tokenizer = mbart_tokenizer
            self.lang_code = lang_code
        else:
            self.mbart_model, self.mbart_tokenizer, self.lang_code = load_mbart(mbart_model_id, mbart_lang)

        # Freeze all parameters of the MBART model
        for param in self.mbart_model.parameters():
            param.requires_grad = False

        self.input_dim = 1024
        self.hidden_dim = 512
        self.mbart_dim = self.mbart_model.model.shared.embedding_dim  # 1024
        
        if projector_args is None:
            projector_args = {}
        projector_class = globals()[projector_type]
        self.projector = projector_class(**projector_args)
        # self.projector = GFTemporalSemanticProjectionLayer(self.input_dim, self.hidden_dim, self.mbart_dim)  # type: ignore

    def check_data(self, features):
        if features is None:
            raise ValueError("Input features must be provided.")

        if self.mbart_tokenizer is None:
            raise ValueError("MBART tokenizer is not initialized.")

    def _build_attention_mask(self, projected_features, feat_len):
        """Costruisce la mask (B, T) per l'encoder di mBART a partire dalle lunghezze reali.

        I frame di padding introdotti dalla collate_fn (ripetizione dell'ultimo frame)
        non devono essere attesi dal cross-attention del decoder.
        """
        if feat_len is None:
            return None
        num_frames = projected_features.size(1)
        lengths = feat_len.to(projected_features.device).long().clamp(min=1, max=num_frames)
        idx = torch.arange(num_frames, device=projected_features.device).unsqueeze(0)
        return (idx < lengths.unsqueeze(1)).long()

    def forward(self, input_features=None, target_texts=None, inference=False, feat_len=None):
        self.check_data(input_features)

        # input_features from BiLSTMLayer is (seq_len, batch_size, hidden_size)
        projected_features = self.projector(input_features)  # This will output (batch_size, seq_len, mbart_dim)

        attention_mask = self._build_attention_mask(projected_features, feat_len)

        if inference:
            # In inference, we want to generate text autoregressively
            generated_ids = self.mbart_model.generate(  # type: ignore
                max_length=512,
                inputs_embeds=projected_features,
                attention_mask=attention_mask,
                forced_bos_token_id=self.mbart_tokenizer.lang_code_to_id[self.lang_code],
                num_beams=5,
                do_sample=True,  # Consider setting to False for more deterministic results
                temperature=1.0,
                top_k=50,
                top_p=0.95,
                repetition_penalty=2.0,
                early_stopping=True,
            )
            return generated_ids  # Return token IDs directly for decoding in SLRModel's forward
        else:
            if target_texts is None:
                raise ValueError("Target text(s) must be provided during training.")

            encoded_text = self.mbart_tokenizer(
                text_target=target_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).input_ids.to(projected_features.device)  # type: ignore

            # I token di padding devono valere -100 per essere ignorati dalla loss
            # (sia dalla loss interna di mBART sia dalle nostre CrossEntropy/CrossRougeEntropy).
            labels = encoded_text.clone()
            pad_id = self.mbart_tokenizer.pad_token_id
            if pad_id is not None:
                labels[labels == pad_id] = -100

            # MBART restituisce un Seq2SeqLMOutput con `logits` e `loss` quando si passano i labels.
            mbart_output = self.mbart_model(
                inputs_embeds=projected_features,
                attention_mask=attention_mask,
                labels=labels,
            )
            # Attacchiamo i target usati così che criterion_calculation possa calcolare
            # loss aggiuntive (es. CrossRougeEntropy) sugli STESSI token mBART, garantendo
            # coerenza tra logits e target (unica sorgente di verità del target).
            mbart_output.text_labels = labels
            return mbart_output
        
    def batch_decode_tokens(self, token_ids):
        """
        Decodes a batch of token IDs back into human-readable text.
        This is a helper for inference results.
        """
        if token_ids is None:
            raise ValueError("Token IDs must be provided for decoding.")

        # Ensure token_ids are on the correct device if needed
        # token_ids = token_ids.to(self.mbart_model.device) # Already on device from `generate`

        response = self.mbart_tokenizer.batch_decode(token_ids, skip_special_tokens=True)
        return response

    def decode_logits(self, logits):
        if logits is None:
            raise ValueError("Logits must be provided.")

        if isinstance(logits, torch.Tensor) and logits.dim() == 2:
            # At Inference time we already have token IDs
            pred_ids = logits
        elif hasattr(logits, "logits"):
            # At training time, we have logits from model output, so we need to extract the logits
            logits = logits.logits
            pred_ids = torch.argmax(logits, dim=-1)
        else:
            raise ValueError("Unrecognized logits format.")

        pred_ids = pred_ids.to(self.mbart_model.device)
        response = self.mbart_tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        return response
