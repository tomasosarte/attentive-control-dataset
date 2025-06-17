import torch

class Transformation:
    
    def apply(self, image: torch.Tensor, **kwargs):
        if len(image.shape) != 3:
            raise ValueError(f"Expected image with 3 dimensions (C, H, W), got {image.shape}")
        return image 