import torch
from torchvision.transforms.v2.functional import rotate

from transformations.transformation import Transformation

class Rotation(Transformation):
    
    @staticmethod
    def apply(images: torch.Tensor, rot_angle: float):
        
        if len(images.shape) != 4:
            raise ValueError(f"Expected image with 4 dimensions (B, C, H, W), got {images.shape}")
        
        return torch.stack([
            rotate(img, angle=rot_angle) for img in images
        ])