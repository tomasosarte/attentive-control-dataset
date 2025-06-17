import torch
from torchvision.transforms.v2.functional import rotate

from src.transformations.invariance import Transformation

class Rotation(Transformation):
    
    @staticmethod
    def apply(image: torch.Tensor, rot_angle: float):
        super().apply(image)
        rotated_image = rotate(image, angle=rot_angle) 
        return rotated_image