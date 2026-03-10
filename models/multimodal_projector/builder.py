import torch
import torch.nn as nn
from typing import List, Dict, Any

from ..ocr_encoder.builder import build_ocr_embedding

class MLPTokenCompressor(nn.Module):
    def __init__(self, input_tokens=256, output_tokens=64):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_tokens, output_tokens),
            nn.GELU(),
            nn.Linear(output_tokens, output_tokens)
        )

    def forward(self, x):
        """
        x: [B, 256, D]
        return: [B, 64, D]
        """

        x = x.transpose(1, 2)      # [B, D, 256]
        x = self.mlp(x)            # [B, D, 64]
        x = x.transpose(1, 2)      # [B, 64, D]

        return x
    
class OCRCrossAttention(nn.Module):
    def __init__(self, hidden_size, num_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            hidden_size, num_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, vision_tokens, ocr_tokens):
        o2, _ = self.attn(ocr_tokens, vision_tokens, vision_tokens)
        ocr_tokens = self.norm(ocr_tokens + o2)
        return ocr_tokens


class MLP(nn.Module):
    def __init__(self, config=None):
        super(MLP, self).__init__()
        inc, ouc = config.mm_hidden_size, config.hidden_size
        self.fc1 = nn.Linear(inc, (inc + ouc) // 2)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear((inc + ouc) // 2, ouc)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x

class OCRVisionProjector(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_prefix_tokens = getattr(config, "num_prefix_tokens", 24)

        self.vision_mlp = MLP(config)
        self.vision_token_compressor = MLPTokenCompressor(input_tokens=195, output_tokens=64)
        self.ocr_embedding = build_ocr_embedding(config)
        self.ocr_cross_attention = OCRCrossAttention(self.hidden_size)

        self.prefix_tokens = nn.Parameter(
            torch.randn(self.num_prefix_tokens, self.hidden_size)
        )

    def forward(
        self,
        vision_feats: torch.Tensor,
        ocr_info: List[Dict[str, Any]],
    ) -> torch.Tensor:
        B = vision_feats.size(0)
        device = vision_feats.device

        vision_tokens = self.vision_mlp(vision_feats)       # (B, N_v, H)

        assert ocr_info is not None, "ocr_info cannot be None for OCRVisionProjector."

        ocr_tokens = self.ocr_embedding(ocr_info).to(device)  # (B, N_o, H)

        ocr_tokens = self.ocr_cross_attention(vision_tokens, ocr_tokens)
        vision_tokens = self.vision_token_compressor(vision_tokens)  # (B, 64, H)
        image_tokens = torch.cat([vision_tokens, ocr_tokens], dim=1)  # (B, N_v + N_o, H)

        prefix = self.prefix_tokens.unsqueeze(0).expand(B, -1, -1)
        image_tokens = torch.cat([prefix, image_tokens], dim=1)

        return image_tokens

class VisionProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mlp = MLP(config)

        self.prefix_tokens = nn.Parameter(
            torch.randn(getattr(config, "num_prefix_tokens", 24), config.hidden_size)
        )

    def forward(self, vision_feats, ocr_info=None):
        B = vision_feats.size(0)
        prefix = self.prefix_tokens.unsqueeze(0).expand(B, -1, -1)
        vision_feats = self.mlp(vision_feats)
        vision_feats = torch.cat([prefix, vision_feats], dim=1)
        return vision_feats

def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'mlp_prefix')

    if getattr(config, 'ocr_path', None) is not None:
        print("Using OCRVisionProjector as vision projector.")
        return OCRVisionProjector(config)

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)
    elif projector_type == 'mlp_prefix':
        return VisionProjector(config)
    raise ValueError(f'Unknown projector type: {projector_type}')