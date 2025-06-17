import torch

class Transformation:

    @staticmethod
    def apply(images: torch.Tensor, **kwargs):
        if len(images.shape) != 4:
            raise ValueError(f"Expected image with 4 dimensions (B, C, H, W), got {images.shape}")
        return images 