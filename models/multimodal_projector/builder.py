import torch
import torch.nn as nn
from typing import Optional, Union, List

from ..ocr_encoder.builder import build_ocr_embedding


class ModalityGate(nn.Module):
    """Gates between two modality embeddings in language hidden space.

    Both img_emb and ocr_emb are expected to have last dim = config.hidden_size.
    """

    def __init__(self, config=None):
        super().__init__()
        hidden_dim = config.hidden_size // 2

        self.gate = nn.Sequential(
            nn.Linear(config.hidden_size * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, config.hidden_size),
            nn.Sigmoid(),
        )

        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, img_emb, ocr_emb):
        x = torch.cat([img_emb, ocr_emb], dim=-1)
        gate = self.gate(x)

        fused = gate * img_emb + (1.0 - gate) * ocr_emb
        fused = self.out_proj(fused)

        return fused



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
    """
    Question-conditioned OCR–Vision projector for text-based VQA.

    Output token order:
        [prefix tokens] + [vision tokens] + [fused OCR tokens]

    Shape:
        (B, N_prefix + N_vision + N_ocr, hidden_size)
    """

    def __init__(self, config):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_prefix_tokens = getattr(config, "num_prefix_tokens", 24)

        # Vision projection
        self.vision_mlp = MLP(config)

        # OCR embedding
        self.ocr_embedding = build_ocr_embedding(config)

        # Question-guided OCR selection
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.ocr_proj = nn.Linear(self.hidden_size, self.hidden_size)

        # OCR -> Vision attention
        self.attn_scale = self.hidden_size ** -0.5

        # Evidence-aware fusion gate (per OCR token)
        self.evidence_gate = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.Sigmoid()
        )

        # Prefix tokens
        self.prefix_tokens = nn.Parameter(
            torch.randn(self.num_prefix_tokens, self.hidden_size)
        )

    def forward(
        self,
        vision_feats: torch.Tensor,                         # (B, N_v, D_v)
        image_ids: Optional[Union[torch.Tensor, List[int]]] = None,
        question_embeds: Optional[torch.Tensor] = None,     # (B, L_q, H)
    ) -> torch.Tensor:

        B = vision_feats.size(0)
        device = vision_feats.device

        vision_tokens = self.vision_mlp(vision_feats)       # (B, N_v, H)

        if image_ids is None:
            image_tokens = vision_tokens
        else:
            if torch.is_tensor(image_ids):
                image_ids = image_ids.tolist()

            ocr_tokens = self.ocr_embedding(image_ids).to(device)  # (B, N_o, H)

            if question_embeds is not None:
                q = question_embeds.mean(dim=1)             # (B, H)
                q_proj = self.q_proj(q)                      # (B, H)
                ocr_proj = self.ocr_proj(ocr_tokens)         # (B, N_o, H)

                # (B, N_o)
                ocr_scores = torch.einsum("bnh,bh->bn", ocr_proj, q_proj)
                ocr_attn = ocr_scores.softmax(dim=-1)

                ocr_tokens = ocr_tokens * ocr_attn.unsqueeze(-1)

            # (B, N_o, N_v)
            attn_scores = torch.einsum(
                "bnh,bmh->bnm", ocr_tokens, vision_tokens
            ) * self.attn_scale

            attn_weights = attn_scores.softmax(dim=-1)
            vision_ctx = attn_weights @ vision_tokens        # (B, N_o, H)

            gate = self.evidence_gate(
                torch.cat([ocr_tokens, vision_ctx], dim=-1)
            )                                                # (B, N_o, H)

            ocr_fused = ocr_tokens + gate * vision_ctx       # (B, N_o, H)

            image_tokens = torch.cat([vision_tokens, ocr_fused], dim=1)

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