

def build_vision_encoder(mm_vision_tower: str, args):
    if "siglip" in mm_vision_tower.lower():
        from .siglip_encoder import SiglipVisionTower
        vision_encoder = SiglipVisionTower(mm_vision_tower, args, delay_load=True)
    elif "dino" in mm_vision_tower.lower():
        from .dino_encoder import DINOVisionTower
        vision_encoder = DINOVisionTower(mm_vision_tower, args, delay_load=True)
    else:
        raise NotImplementedError(f"Vision encoder {mm_vision_tower} is not implemented.")
    return vision_encoder