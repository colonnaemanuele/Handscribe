# NOTE: this is not the final recognizer, we might use a tuned LLM atm

import torch
from torch import nn

class TransformerDecoderWithFeatures(nn.Module):
    def __init__(self, config, feature_dim):
        """
        Initializes the Transformer decoder with additional feature fusion.
        
        Args:
            config (PretrainedConfig): 
                - d_model: hidden size of embeddings and transformer layers
                - vocab_size: size of the target vocabulary
                - max_position_embeddings: maximum sequence length supported
                - encoder_attention_heads: number of attention heads in cross-attention
                - decoder_layers: number of decoder layers
            feature_dim (int): 
                - dimensionality of the extra (side) features you want to incorporate
        """
        super().__init__()
        self.embed_dim = config.d_model
        
        # Positional & token embeddings for decoder inputs
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb   = nn.Embedding(config.max_position_embeddings, config.d_model)
        
        # Project extra features
        self.feat_proj = nn.Linear(feature_dim, config.d_model)
        
        # Standard Transformer decoder layer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model, nhead=config.encoder_attention_heads
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.decoder_layers)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, decoder_input_ids, encoder_hidden_states, extra_features, fusion_mode='add'):
        """
        Runs the decoder, attending to encoder outputs and fusing extra features.

        Args:
            decoder_input_ids (Tensor):
                - shape [batch_size, target_seq_len]
                - token IDs of the target sequence, shifted right for teacher forcing
            encoder_hidden_states (Tensor):
                - shape [batch_size, source_seq_len, d_model]
                - hidden states from the mBART encoder (context)
            extra_features (Tensor):
                - shape [batch_size, feature_dim]
                - additional side information (e.g., logits, metadata)
            fusion_mode (str):
                - 'add'      : simple addition of feature signal
                - 'concat'   : concatenation followed by linear fusion

        Returns:
            logits (Tensor):
                - shape [batch_size, target_seq_len, vocab_size]
                - unnormalized scores for each token in the vocabulary
        """
        _, seq_len = decoder_input_ids.size()

        # 1. Token + positional embeddings
        positions = torch.arange(seq_len, device=decoder_input_ids.device).unsqueeze(0)
        token_embeddings = self.token_emb(decoder_input_ids)          # [B, T, D]
        position_embeddings = self.pos_emb(positions)                 # [1, T, D]
        x = token_embeddings + position_embeddings                   # [B, T, D]

        # 2. Project extra features
        feat_proj = self.feat_proj(extra_features)                    # [B, D]
        feat_expanded = feat_proj.unsqueeze(1).repeat(1, seq_len, 1)  # [B, T, D]
        
        # 3. Fuse by either simple addition or concatenation + linear projection
        if fusion_mode == 'add':
            # simple element-wise addition
            x = x + feat_expanded                                    # [B, T, D]
        elif fusion_mode == 'concat':
            # concatenate along feature dimension and project back
            concat = torch.cat([x, feat_expanded], dim=-1)          # [B, T, 2*D]
            x = self.fusion_linear(concat)                          # [B, T, D]
        else:
            raise ValueError("fusion_mode must be 'add' or 'concat'")
        
        # 4. Transformer decoding with cross-attention to encoder
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        out = self.transformer_decoder(
            x.transpose(0,1),                   # [seq, batch, dim]
            encoder_hidden_states.transpose(0,1),# [src_seq, batch, dim]
            tgt_mask=tgt_mask
        )
        
        logits = self.lm_head(out.transpose(0,1))
        return logits

# Training loop (conceptual)
for batch in dataloader:
    enc_out = mbart.encoder(**batch.inputs).last_hidden_state
    logits  = decoder(batch.dec_input_ids, enc_out, batch.extra_feats)
    loss    = loss_fn(logits.view(-1, logits.size(-1)), batch.labels.view(-1))
    loss.backward(); optimizer.step(); optimizer.zero_grad()
