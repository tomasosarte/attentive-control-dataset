import os
import click
import numpy as np
from math import isclose
import inspect

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from transformations import TRANSFORMATIONS

from config.config_classes import Config, TransformationConfig, load_config


def load_dataset(name: str, data_dir: str, to_tensor: transforms.Compose):
    """Dynamically load a torchvision dataset.

    Parameters are inferred based on the dataset constructor signature. The
    dataset must be available in ``torchvision.datasets``.
    """
    dataset_cls = getattr(datasets, name, None)
    if dataset_cls is None:
        raise ValueError(f"Unsupported dataset: {name}")

    kwargs = {"root": data_dir, "transform": to_tensor}
    sig = inspect.signature(dataset_cls)
    if "download" in sig.parameters:
        kwargs["download"] = True
    if "train" in sig.parameters:
        kwargs["train"] = True
    if "split" in sig.parameters:
        kwargs["split"] = "train"

    return dataset_cls(**kwargs)

def get_kwargs(transformation: TransformationConfig) -> dict:
    """Convert a transformation configuration to kwargs."""
    return transformation.to_kwargs()

@click.command()
@click.option('-c', '--config', type=click.Path(exists=True), required=True, help='Path to the YAML configuration file.')
def generate(config):

    # Load YAML config
    config = load_config(config)

    # Sanity checks
    if len(config.proportions) != len(config.transformations):
        raise ValueError(f"Proportions should be a least of the same length as the transformations")
    
    sum_prop = sum(config.proportions)
    if not isclose(sum_prop, 1.0, rel_tol=1e-6):
        raise ValueError(f"Proportions should add up to 1.0 and add up to: {sum_prop}")
    
    # Make dirs
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Load dataset
    to_tensor = transforms.ToTensor()
    dataset = load_dataset(config.dataset, config.data_dir, to_tensor)

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
    transform_metadata = []

    # Add original dataset
    if config.include_original:
        full_loader = DataLoader(dataset, batch_size=64, shuffle=False)
        for batch_images, batch_labels in full_loader:
            all_images.append(batch_images)
            all_labels.append(batch_labels)
            transform_metadata.extend([{'type': 'original', 'params': {}} for _ in range(batch_images.size(0))])

    # Apply transformations to each subset
    for subset, transform_config in zip(subsets, config.transformations):
        loader = DataLoader(subset, batch_size=64, shuffle=False)
        transform_name = transform_config.type
        transform_cls = TRANSFORMATIONS[transform_name]
        kwargs = get_kwargs(transform_config)

        # Apply batched transformation
        for batch_images, batch_labels in loader:
            transformed_batch = transform_cls.apply(batch_images, **kwargs)
            all_images.append(transformed_batch)
            all_labels.append(batch_labels)
            transform_metadata.extend([{'type': transform_name, 'params': kwargs} for _ in range(batch_images.size(0))])

    # Final concatenation
    all_images_tensor = torch.cat(all_images, dim=0)
    all_labels_tensor = torch.cat(all_labels, dim=0)

    # Save dataset
    output_path = os.path.join(config.output_dir, f"{config.dataset.lower()}_augmented.pt")
    torch.save({
        'images': all_images_tensor,
        'labels': all_labels_tensor,
        'metadata': transform_metadata
    }, output_path)

    print(f"Saved augmented dataset with {len(all_images_tensor)} samples to {output_path}")

if __name__ == '__main__':
    generate()