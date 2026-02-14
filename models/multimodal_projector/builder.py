import torch
import torch.nn as nn
from typing import Optional, Union, List

from ..ocr_encoder.builder import build_ocr_embedding

def interleave_ratio(vision_tokens, ocr_tokens, v_ratio=1, o_ratio=1):
    B, Nv, H = vision_tokens.shape
    _, No, _ = ocr_tokens.shape

    v_idx = 0
    o_idx = 0
    chunks = []

    while v_idx < Nv or o_idx < No:
        if v_idx < Nv:
            chunks.append(vision_tokens[:, v_idx:v_idx+v_ratio])
            v_idx += v_ratio

        if o_idx < No:
            chunks.append(ocr_tokens[:, o_idx:o_idx+o_ratio])
            o_idx += o_ratio

    return torch.cat(chunks, dim=1)

class CrossModalFusion(nn.Module):
    def __init__(self, hidden_size, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()

        self.cross_attn_v = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn_o = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )

        self.norm1_v = nn.LayerNorm(hidden_size)
        self.norm1_o = nn.LayerNorm(hidden_size)

        # FFN
        mlp_hidden = int(hidden_size * mlp_ratio)

        self.mlp_v = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_size),
            nn.Dropout(dropout),
        )

        self.mlp_o = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_size),
            nn.Dropout(dropout),
        )

        self.norm2_v = nn.LayerNorm(hidden_size)
        self.norm2_o = nn.LayerNorm(hidden_size)

    def forward(self, v, o):
        # Cross Attention
        v2, _ = self.cross_attn_v(v, o, o)
        o2, _ = self.cross_attn_o(o, v, v)

        v = self.norm1_v(v + v2)
        o = self.norm1_o(o + o2)

        # MLP block
        v2 = self.mlp_v(v)
        o2 = self.mlp_o(o)

        v = self.norm2_v(v + v2)
        o = self.norm2_o(o + o2)

        return v, o

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
        self.ocr_embedding = build_ocr_embedding(config)

        # self.cross_modal_fusion = CrossModalFusion(self.hidden_size)

        self.prefix_tokens = nn.Parameter(
            torch.randn(self.num_prefix_tokens, self.hidden_size)
        )

    def forward(
        self,
        vision_feats: torch.Tensor,                         # (B, N_v, D_v)
        image_ids: Optional[Union[torch.Tensor, List[int]]] = None,
    ) -> torch.Tensor:

        B = vision_feats.size(0)
        device = vision_feats.device

        vision_tokens = self.vision_mlp(vision_feats)       # (B, N_v, H)

        assert image_ids is not None, "image_ids must be provided for OCRVisionProjector."
        if torch.is_tensor(image_ids):
            image_ids = image_ids.tolist()

        ocr_tokens = self.ocr_embedding(image_ids).to(device)  # (B, N_o, H)

        # vision_tokens, ocr_tokens = self.cross_modal_fusion(vision_tokens, ocr_tokens)
        # image_tokens = torch.cat([vision_tokens, ocr_tokens], dim=1)  # (B, N_v + N_o, H)
        image_tokens = interleave_ratio(vision_tokens, ocr_tokens)  # (B, N_v + N_o, H)

        prefix = self.prefix_tokens.unsqueeze(0).expand(B, -1, -1)
        image_tokens = torch.cat([prefix, image_tokens], dim=1)

        return image_tokens


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'mlp2x_gelu')

    if getattr(config, 'ocr_path', None) is not None:
        print("Using OCRVisionProjector as vision projector.")
        return OCRVisionProjector(config)

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    elif projector_type.startswith('mlp'):
        return MLP(config)
    raise ValueError(f'Unknown projector type: {projector_type}')