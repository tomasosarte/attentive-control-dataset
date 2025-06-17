import torch
from torchvision.transforms.v2.functional import affine

from . import register_transformation
from .transformation import Transformation

@register_transformation("translation")
class Translation(Transformation):

    @staticmethod
    def apply(images: torch.Tensor, translate: tuple[float, float]):

        if len(images.shape) != 4:
            raise ValueError(f"Expected image with 4 dimensions (B, C, H, W), got {images.shape}")

        dx, dy = translate
        return torch.stack([
            affine(img, angle=0.0, translate=[dx, dy], scale=1.0, shear=[0.0, 0.0])
            for img in images
        ])
