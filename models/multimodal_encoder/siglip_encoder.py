import torch
import torch.nn as nn
from transformers import SiglipVisionModel, Siglip2VisionModel, SiglipImageProcessor, SiglipVisionConfig

class SiglipVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = -2
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
    
        if not delay_load:
            self.load_model()
            self.init_layer_fusion() 
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
            self.init_layer_fusion() 
        else:
            self.cfg_only = SiglipVisionConfig.from_pretrained(self.vision_tower_name)

    def init_layer_fusion(self):
        num_layers = self.config.num_hidden_layers + 1
        hidden_size = self.config.hidden_size

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
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return
        
        self.image_processor = SiglipImageProcessor.from_pretrained(self.vision_tower_name)
        if "naflex" in self.vision_tower_name.lower():
            self.vision_tower = Siglip2VisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
            print("Loaded SigLIP-Naflex vision model.")
        else:
            self.vision_tower = SiglipVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
            print("Loaded SigLIP vision model.")
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True
        self.is_model_type = True if "naflex" in self.vision_tower_name.lower() else False
        # Ensure fusion layers are initialized after the model is loaded
        if not hasattr(self, 'fusion_linears') or not hasattr(self, 'fusion_layernorms'):
            self.init_layer_fusion()

    def feature_select(self, image_forward_outs):
        hidden_states = image_forward_outs.hidden_states

        features_sum = 0.0
        for i, layer_features in enumerate(hidden_states):
            processed = self.fusion_layernorms[i](
                self.fusion_linears[i](layer_features)
            ) * self.fusion_alphas[i]

            features_sum = features_sum + processed

        return features_sum

    def _convert_to_patches(self, pixel_values, patch_size):
        """
        Convert image tensor to patches.

        Args:
            pixel_values (torch.Tensor): Input tensor of shape [bs, channels, height, width]
            patch_size (int): Size of each patch

        Returns:
            torch.Tensor: Patches of shape [bs, num_patches, channels * patch_size * patch_size]
        """
        batch_size, channels, height, width = pixel_values.shape
        num_patches_height = height // patch_size
        num_patches_width = width // patch_size

        patches = pixel_values.reshape(
            batch_size, channels,
            num_patches_height, patch_size,
            num_patches_width, patch_size
        )

        patches = patches.permute(0, 2, 4, 3, 5, 1)

        patches = patches.reshape(
            batch_size,
            num_patches_height * num_patches_width,
            patch_size * patch_size * channels
        )

        return patches
    
    def preprocess(self, images):
        pixel_values = images
        batch_size, _, height, width = pixel_values.shape

        cfg = self.config
        if not hasattr(cfg, "patch_size"):
            raise ValueError("Vision config missing patch_size")

        patch_size = cfg.patch_size

        if height % patch_size != 0 or width % patch_size != 0:
            raise ValueError(
                f"Image size {height}x{width} not divisible by patch_size {patch_size}"
            )

        num_patches_h = height // patch_size
        num_patches_w = width // patch_size
        num_patches = num_patches_h * num_patches_w

        spatial_shapes = torch.tensor(
            [[num_patches_h, num_patches_w]],
            device=pixel_values.device
        ).expand(batch_size, -1)

        attention_mask = torch.ones(
            batch_size, num_patches,
            device=pixel_values.device,
            dtype=torch.long
        )

        pixel_values = self._convert_to_patches(pixel_values, patch_size)
        return pixel_values, spatial_shapes, attention_mask


    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                if self.is_model_type:
                    pixel_values, spatial_shapes, attention_mask = self.preprocess(image.unsqueeze(0))
                    image_forward_out = self.vision_tower(
                        pixel_values=pixel_values,
                        pixel_attention_mask=attention_mask,
                        spatial_shapes=spatial_shapes,
                        output_hidden_states=True
                    )
                else:
                    image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            if self.is_model_type:
                pixel_values, spatial_shapes, attention_mask = self.preprocess(images)
                image_forward_outs = self.vision_tower(
                    pixel_values=pixel_values,
                    pixel_attention_mask=attention_mask,
                    spatial_shapes=spatial_shapes,
                    output_hidden_states=True
                )
            else:
                image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
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

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2