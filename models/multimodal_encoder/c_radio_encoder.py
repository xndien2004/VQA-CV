import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoModel, CLIPImageProcessor, AutoConfig


class CRadioVisionTower(nn.Module):
    def __init__(self, vision_tower: str, args=None, delay_load: bool = False):
        super().__init__()

        self.is_loaded = False
        self.vision_tower_name = vision_tower
        self.select_layer = -2
        self.select_feature = getattr(args, "mm_vision_select_feature", "patch") if args is not None else "patch"

        if not delay_load:
            self.load_model()
            self.init_layer_fusion()
        elif getattr(args, "unfreeze_mm_vision_tower", False):
            self.load_model()
            self.init_layer_fusion()
        else:
            self.cfg_only = AutoConfig.from_pretrained(self.vision_tower_name, trust_remote_code=True)

    def infer_hidden_size(self):
        cfg = self.config
        for k in ["hidden_size", "embed_dim", "vision_hidden_size", "dim", "width"]:
            if hasattr(cfg, k):
                v = getattr(cfg, k)
                if v is not None:
                    return v

        if hasattr(self, "vision_tower"):
            for _, p in self.vision_tower.named_parameters():
                if p.ndim == 2:
                    return p.shape[-1]

        raise ValueError(f"Cannot infer hidden_size. Config keys: {cfg.to_dict().keys()}")

    def infer_num_hidden_layers(self):
        cfg = self.config
        for k in ["num_hidden_layers", "depth", "num_layers", "n_layer", "encoder_layers"]:
            if hasattr(cfg, k):
                v = getattr(cfg, k)
                if v is not None:
                    return v

        if hasattr(self, "vision_tower"):
            try:
                dummy = torch.zeros(1, 3, 224, 224, device=self.device, dtype=self.dtype)
                outs = self.vision_tower(dummy, output_hidden_states=True)
                if hasattr(outs, "hidden_states") and outs.hidden_states is not None:
                    return len(outs.hidden_states) - 1
            except Exception:
                pass

        raise ValueError(f"Cannot infer num_hidden_layers. Config keys: {cfg.to_dict().keys()}")

    def init_layer_fusion(self):
        hidden_size = self.infer_hidden_size()
        num_hidden_layers = self.infer_num_hidden_layers()
        num_layers = num_hidden_layers + 1

        self.fusion_linears = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)
        ])

        self.fusion_layernorms = nn.ModuleList([
            nn.LayerNorm(hidden_size) for _ in range(num_layers)
        ])

        self.fusion_alphas = nn.ParameterList([
            nn.Parameter(torch.ones(1)) for _ in range(num_layers)
        ])

    def load_model(self, device_map=None):
        if self.is_loaded:
            print(f"{self.vision_tower_name} already loaded, skipping.")
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = AutoModel.from_pretrained(
            self.vision_tower_name,
            trust_remote_code=True,
            device_map=device_map
        )

        self.vision_tower.requires_grad_(False)
        self.is_loaded = True

        if not hasattr(self, "fusion_linears") or not hasattr(self, "fusion_layernorms"):
            self.init_layer_fusion()

    def feature_select(self, image_forward_outs):
        hidden_states = getattr(image_forward_outs, "hidden_states", None)

        if hidden_states is None:
            last = getattr(image_forward_outs, "last_hidden_state", None)
            return last

        features_sum = None
        for i, layer_features in enumerate(hidden_states):
            processed = self.fusion_layernorms[i](
                self.fusion_linears[i](layer_features)
            ) * self.fusion_alphas[i]

            if features_sum is None:
                features_sum = processed
            else:
                features_sum = features_sum + processed

        return features_sum

    @torch.no_grad()
    def forward(self, images):
        if not self.is_loaded:
            self.load_model()

        if isinstance(images, Image.Image):
            pixel = self.image_processor(images, return_tensors="pt")["pixel_values"].to(
                device=self.device, dtype=self.dtype
            )
            outs = self.vision_tower(pixel, output_hidden_states=True)
            image_features = self.feature_select(outs).to(dtype=self.dtype)
            return image_features

        if isinstance(images, list):
            image_features = []
            for img in images:
                if isinstance(img, Image.Image):
                    pixel = self.image_processor(img, return_tensors="pt")["pixel_values"].to(
                        device=self.device, dtype=self.dtype
                    )
                else:
                    pixel = img.to(device=self.device, dtype=self.dtype).unsqueeze(0)

                outs = self.vision_tower(pixel, output_hidden_states=True)
                feat = self.feature_select(outs).to(dtype=self.dtype)
                image_features.append(feat)

            return image_features

        outs = self.vision_tower(
            images.to(device=self.device, dtype=self.dtype),
            output_hidden_states=True
        )
        image_features = self.feature_select(outs).to(dtype=self.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return next(self.vision_tower.parameters()).device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.infer_hidden_size()