import os
import yaml
import click
from math import isclose

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from src.transformations.rotation import Rotation
from src.transformations.translation import Translation

from config.config_classes import Config

TRANSFORM_CLASSES = {
    'rotation': Rotation,
    'translation': Translation,
}

def load_config(path: str) -> Config:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return Config(**data)

@click.command()
@click.option('--config', type=click.Path(exists=True), required=True, help='Path to the YAML configuration file.')
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

if __name__ == '__main__':
    generate()
