from transformers import ViTModel, SwinModel
import torch.nn as nn


class ViTBackbone(nn.Module):
    def __init__(self, model_name="google/vit-base-patch16-224", out_dim=2304, local_files_only=False):
        super().__init__()
        self.backbone = ViTModel.from_pretrained(model_name, local_files_only=local_files_only)
        self.hidden_dim = self.backbone.config.hidden_size  # 768
        self.out_dim = out_dim
        self.proj = nn.Linear(self.hidden_dim, self.out_dim)
        # inizializzazione lineare (opzionale, ma utile)
        nn.init.xavier_uniform_(self.proj.weight, gain=nn.init.calculate_gain("relu"))
        if self.proj.bias is not None:
            nn.init.constant_(self.proj.bias, 0.0)

    def forward(self, x):
        """
        x:
          - (B, T, C, H, W)  (batch, time, channels, H, W)
        Restituisce:
          - feats: (B, out_dim, T)
        """
        # Normalizziamo i casi di input
        if x.dim() != 5:
            raise ValueError("ViTBackbone expects a 5D tensor (B,T,C,H,W) or (B,C,T,H,W). Got: {}".format(x.shape))

        if x.size(1) in (1, 3):  # (B, C, T, H, W)
            x = x.permute(0, 2, 1, 3, 4)  # -> (B, T, C, H, W)

        B, T, C, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)  # (B*T, C, H, W)

        outputs = self.backbone(x)
        cls_tokens = outputs.last_hidden_state[:, 0, :]  # (B*T, hidden_dim)
        cls_tokens = cls_tokens.view(B, T, self.hidden_dim)  # (B, T, D)
        proj = self.proj(cls_tokens)  # (B, T, out_dim)
        feats = proj.permute(0, 2, 1).contiguous()  # (B, out_dim, T)

        return feats


class SwinBackbone(nn.Module):
    def __init__(self, model_name="microsoft/swin-base-patch4-window7-224", out_dim=2304, local_files_only=True):
        super().__init__()
        self.backbone = SwinModel.from_pretrained(model_name, local_files_only=local_files_only)
        self.hidden_dim = self.backbone.config.hidden_size  # Extract hidden size from the model's config
        self.out_dim = out_dim
        self.proj = nn.Linear(self.hidden_dim, self.out_dim)
        # Initialize the projection layer
        nn.init.xavier_uniform_(self.proj.weight, gain=nn.init.calculate_gain("relu"))
        if self.proj.bias is not None:
            nn.init.constant_(self.proj.bias, 0.0)

    def forward(self, x):
        if x.dim() != 5:
            raise ValueError("Swin Backbone expects a 5D tensor (B,T,C,H,W) or (B,C,T,H,W). Got: {}".format(x.shape))

        if x.size(1) in (1, 3):  # (B, C, T, H, W)
            x = x.permute(0, 2, 1, 3, 4)  # -> (B, T, C, H, W)

        B, T, C, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)  # (B*T, C, H, W)
        outputs = self.backbone(x)
        cls_tokens = outputs.last_hidden_state[:, 0, :]  # Extract [CLS] token (B*T, hidden_dim)
        cls_tokens = cls_tokens.view(B, T, self.hidden_dim)  # Reshape to (B, T, hidden_dim)
        proj = self.proj(cls_tokens)  # Project to (B, T, out_dim)
        feats = proj.permute(0, 2, 1).contiguous()  # (B, out_dim, T)
        return feats