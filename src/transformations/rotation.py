import torch
from torchvision.transforms.v2.functional import rotate

from . import register_transformation
from .transformation import Transformation

@register_transformation("rotation")
class Rotation(Transformation):
    
    @staticmethod
    def apply(images: torch.Tensor, rot_angle: float):
        
        if len(images.shape) != 4:
            raise ValueError(f"Expected image with 4 dimensions (B, C, H, W), got {images.shape}")
        
        return torch.stack([
            rotate(img, angle=rot_angle) for img in images
        ])