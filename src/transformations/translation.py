import torch
from torchvision.transforms.v2.functional import affine
from src.transformations.transformation import Transformation

class Translation(Transformation):

    @staticmethod
    def apply(images: torch.Tensor, translate: tuple[float, float]):
        super().apply(images)  # Check shape

        dx, dy = translate
        return torch.stack([
            affine(img, angle=0.0, translate=[dx, dy], scale=1.0, shear=[0.0, 0.0])
            for img in images
        ])
