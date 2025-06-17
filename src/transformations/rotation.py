import torch
from torchvision.transforms.v2.functional import rotate

from src.transformations.transformation import Transformation

class Rotation(Transformation):
    
    @staticmethod
    def apply(images: torch.Tensor, rot_angle: float):
        super().apply(images)
        return torch.stack([
            rotate(img, angle=rot_angle) for img in images
        ])