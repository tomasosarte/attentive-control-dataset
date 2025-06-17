import torch
from torchvision.transforms.v2.functional import affine
from src.transformations.invariance import Transformation

class Translation(Transformation):

    @staticmethod
    def apply(image: torch.Tensor, translate: tuple[float, float]):
        super().apply(image)  # Check shape

        dx, dy = translate
        translated_image = affine(
            image,
            angle=0.0,             # No rotation
            translate=[dx, dy],    # x, y translation in pixels
            scale=1.0,             # No scaling
            shear=[0.0, 0.0]       # No shearing
        )

        return translated_image
