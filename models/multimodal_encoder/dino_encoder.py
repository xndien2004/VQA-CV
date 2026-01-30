import torch
import torch.nn as nn

from transformers import AutoModel, AutoConfig, AutoImageProcessor


class DINOVisionTower(nn.Module):
    '''
    Docstring for DINOVisionTower
    Vision encoder using DINO model.
    Args:
        vision_tower (str): Name of the pretrained DINO model.
        args: Additional arguments.
        delay_load (bool): Whether to delay loading the model.

    model supports:
        - facebook/dinov3-convnext-base-pretrain-lvd1689m 
        - facebook/dinov3-convnext-large-pretrain-lvd1689m 
        - facebook/dinov3-convnext-small-pretrain-lvd1689m 
        - facebook/dinov3-convnext-tiny-pretrain-lvd1689m
    '''
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = -2

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = AutoConfig.from_pretrained(self.vision_tower_name)

    def load_model(self):
        self.image_processor = AutoImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = AutoModel.from_pretrained(self.vision_tower_name)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]

        return image_features

    @torch.no_grad()
    def forward(self, images):
        if isinstance(images, list):
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(
                    image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                    output_hidden_states=True,
                )
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(
                images.to(device=self.device, dtype=self.dtype),
                output_hidden_states=True,
            )
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    def _infer_hidden_size(self):
        cfg = self.config
        if hasattr(cfg, "hidden_size"):
            return cfg.hidden_size
        if hasattr(cfg, "hidden_sizes"):
            hidden_sizes = cfg.hidden_sizes
            if isinstance(hidden_sizes, (list, tuple)):
                return hidden_sizes[-1]
            return int(hidden_sizes)
        raise AttributeError("DINO config has no hidden_size/hidden_sizes to infer vision hidden dimension.")

    @property
    def hidden_size(self):
        return self._infer_hidden_size()

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2