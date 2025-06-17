import os
import yaml
import click
import numpy as np
from math import isclose

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from src.transformations.rotation import Rotation
from src.transformations.translation import Translation
from src.transformations.transformation import Transformation

from config.config_classes import Config, TransformationConfig, RotationConfig, TranslationConfig

TRANSFORM_CLASSES: dict[str, Transformation] = {
    'rotation': Rotation,
    'translation': Translation,
}

def load_config(path: str) -> Config:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return Config(**data)


def get_kwargs(transformation: TransformationConfig):
    if isinstance(transformation, TranslationConfig):
        return {'translate': (transformation.x, transformation.y)}
    elif isinstance(transformation, RotationConfig):
        return {'rot_angle': transformation.angle}
    else:
        raise ValueError(f"Unknown transformation type: {transformation}")

@click.command()
@click.option('-c', '--config', type=click.Path(exists=True), required=True, help='Path to the YAML configuration file.')
def generate(config):

    # Load YAML config
    config = load_config(config)

    # Sanity checks
    if len(config.proportions) != len(config.transformations):
        raise ValueError(f"Proportions should be a least of the same length as the transformations")
    
    if isclose(sum(config.proportions), 1.0, rel_tol=1e-9):
        raise ValueError(f"Proportions should add up to 1.0")
    
    # Make dirs
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Load dataset
    to_tensor = transforms.ToTensor()
    if config.dataset == 'MNIST':
        dataset = datasets.MNIST(root=config.data_dir, train=True, download=True, transform=to_tensor)
    elif config.dataset == 'CIFAR10':
        dataset = datasets.CIFAR10(root=config.data_dir, train=True, download=True, transform=to_tensor)
    else:
        raise ValueError(f"Unsupported dataset: {config.dataset}")

    # Generate splits
    num_samples = len(dataset)
    indices = np.random.permutation(num_samples)  
    proportion_sizes = [int(p * num_samples) for p in config.proportions]
    proportion_sizes[-1] = num_samples - sum(proportion_sizes[:-1])  

    splits = []
    start = 0
    for size in proportion_sizes:
        end = start + size
        splits.append(indices[start:end])
        start = end
    
    # Create subsets
    subsets = [Subset(dataset, split) for split in splits]

    # Containers for final dataset
    all_images = []
    all_labels = []

    # Add original dataset
    if config.include_original:
        full_loader = DataLoader(dataset, batch_size=64, shuffle=False)
        for batch_images, batch_labels in full_loader:
            all_images.append(batch_images)
            all_labels.append(batch_labels)

    # Apply transformations to each subset
    for subset, transform_config in zip(subsets, config.transformations):
        loader = DataLoader(subset, batch_size=64, shuffle=False)

        transform_name = transform_config.__class__.__name__.replace("Config", "").lower()
        transform_cls = TRANSFORM_CLASSES[transform_name]
        kwargs = get_kwargs(transform_config)

        # Apply batched transformation
        for batch_images, batch_labels in loader:
            transformed_batch = transform_cls.apply(batch_images, **kwargs)
            all_images.append(transformed_batch)
            all_labels.append(batch_labels)

    # Final concatenation
    all_images_tensor = torch.cat(all_images, dim=0)
    all_labels_tensor = torch.cat(all_labels, dim=0)

    # Save dataset
    output_path = os.path.join(config.output_dir, f"{config.dataset.lower()}_augmented.pt")
    torch.save({
        'images': all_images_tensor,
        'labels': all_labels_tensor,
    }, output_path)

    print(f"Saved augmented dataset with {len(all_images_tensor)} samples to {output_path}")

if __name__ == '__main__':
    generate()