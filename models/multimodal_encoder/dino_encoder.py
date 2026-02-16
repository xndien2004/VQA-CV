import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, AutoImageProcessor


class DINOVisionTower(nn.Module):
    """
    Vision encoder using DINOv3 models from HuggingFace.

    Supported examples:
        - facebook/dinov3-convnext-base-pretrain-lvd1689m
        - facebook/dinov3-convnext-large-pretrain-lvd1689m
        - facebook/dinov3-convnext-small-pretrain-lvd1689m
        - facebook/dinov3-convnext-tiny-pretrain-lvd1689m
    """

    def __init__(self, vision_tower, args=None, delay_load=False):
        super().__init__()

        self.is_loaded = False
        self.vision_tower_name = vision_tower

        self.select_layer = getattr(args, "mm_vision_select_layer", -2) if args else -2
        self.select_feature = getattr(args, "mm_vision_select_feature", "patch") if args else "patch"

        if not delay_load:
            self.load_model()
            self.init_layer_fusion()
        else:
            self.cfg_only = AutoConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print(f"{self.vision_tower_name} is already loaded, skipping.")
            return

        self.image_processor = AutoImageProcessor.from_pretrained(self.vision_tower_name)

        self.vision_tower = AutoModel.from_pretrained(
            self.vision_tower_name,
            device_map=device_map
        )

        # freeze backbone
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def _infer_hidden_size(self):
        cfg = self.config

        if hasattr(cfg, "hidden_size"):
            return cfg.hidden_size

        if hasattr(cfg, "hidden_sizes"):
            hs = cfg.hidden_sizes
            if isinstance(hs, (list, tuple)):
                return hs[-1]
            return int(hs)

        raise AttributeError(
            "DINO config has no hidden_size/hidden_sizes to infer hidden dimension."
        )

    def init_layer_fusion(self):
        """
        Initialize per-layer fusion modules:
            Linear + LayerNorm + alpha
        """
        num_layers = self.config.num_hidden_layers + 1
        hidden_size = self._infer_hidden_size()

        self.fusion_linears = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)
        ])

        self.fusion_layernorms = nn.ModuleList([
            nn.LayerNorm(hidden_size) for _ in range(num_layers)
        ])

        self.fusion_alphas = nn.ParameterList([
            nn.Parameter(torch.ones(1)) for _ in range(num_layers)
        ])

    def feature_select(self, image_forward_outs):
        """
        Take hidden_states and fuse all layers into a single feature tensor.
        """
        hidden_states = image_forward_outs.hidden_states
        out = 0.0

        for i, layer_features in enumerate(hidden_states):
            processed = self.fusion_layernorms[i](
                self.fusion_linears[i](layer_features)
            ) * self.fusion_alphas[i]

            out = out + processed

        return out

    @torch.no_grad()
    def forward(self, images):
        if isinstance(images, list):
            image_features = []
            for image in images:
                outs = self.vision_tower(
                    image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                    output_hidden_states=True,
                )
                feat = self.feature_select(outs).to(image.dtype)
                image_features.append(feat)

            return image_features

        outs = self.vision_tower(
            images.to(device=self.device, dtype=self.dtype),
            output_hidden_states=True,
        )

        image_features = self.feature_select(outs).to(images.dtype)
        return image_features

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        return self.cfg_only

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def hidden_size(self):
        return self._infer_hidden_size()

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)
