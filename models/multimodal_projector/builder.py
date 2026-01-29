import torch
import torch.nn as nn

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
    """Projector that fuses vision features with OCR embeddings when available.

    - Vision features (from vision tower) are first projected to language hidden size.
    - OCR features are built once from global OCR features via build_ocr_embedding.
    - Both are mean-pooled to a single token per image and fused with a ModalityGate.
    """

    def __init__(self, config=None):
        super().__init__()
        self.vision_mlp = MLP(config)
        self.ocr_embedding = build_ocr_embedding(config)
        self.modality_gate = ModalityGate(config)

    def forward(self, vision_feats: torch.Tensor, image_ids=None) -> torch.Tensor:
        vision_proj = self.vision_mlp(vision_feats)
        if image_ids is None:
            return vision_proj
        if isinstance(image_ids, torch.Tensor):
            image_id_list = image_ids.detach().cpu().tolist()
        else:
            image_id_list = image_ids
        ocr_features = self.ocr_embedding(image_id_list)

        vision_token = vision_proj.mean(dim=1, keepdim=True)
        ocr_token = ocr_features.mean(dim=1, keepdim=True)

        fused = self.modality_gate(vision_token, ocr_token)
        return fused

def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'mlp2x_gelu')

    if getattr(config, 'ocr_path', None) is not None:
        return OCRVisionProjector(config)

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    elif projector_type.startswith('mlp'):
        return MLP(config)
    raise ValueError(f'Unknown projector type: {projector_type}')