# 🧪 Control Dataset for Testing Equivariance Sensitivity in CNNs

This repository provides a **modular dataset generator** for evaluating how convolutional neural networks (CNNs) generalize under transformation biases—such as rotations and translations. It is especially useful for assessing **attentive group convolutions**, as introduced by [Romero et al., 2020].

## 🎯 Purpose

Group-equivariant CNNs (G-CNNs) ensure equivariance to transformations like rotations and translations, but they treat all transformations equally—even if some are irrelevant or misleading for the task. In contrast, **Attentive G-CNNs** learn to prioritize the most informative transformations.

This dataset generator enables controlled experimentation by allowing researchers to explicitly define which transformations occur and with what frequency. It is designed to test whether models that learn to **attend to useful transformation types** achieve better generalization under non-uniform, realistic transformation distributions.

## ❓ Research Question

> Does introducing attention over group transformations improve the generalization performance of CNNs, especially when the training data contains imbalanced transformation distributions?

## ⚙️ Dataset Generator

A typical configuration is defined in a YAML file like this:

```yaml
dataset: MNIST
data_dir: ./data
output_dir: ./transformed
transformations:
  - type: rotation
    angle: 90
  - type: translation
    x: 5
    y: 0
proportions: [0.7, 0.3]
include_original: true
```

- `dataset`: Name of the torchvision dataset to use (e.g., MNIST).
- `transformations`: List of transformations to apply, each with its parameters.
- `proportions`: Proportions (summing to 1) to apply each transformation to the dataset.
- `include_original`: Whether to include unmodified images in the output dataset.

The generator randomly shuffles the original dataset and applies transformations according to the specified proportions. Each image’s **label is preserved**.

## 🌀 Supported Transformations

Each image is modified using one of the following, sampled according to the provided proportions:

| Transformation | Description                     |
|----------------|---------------------------------|
| `rotation`     | Rotates image by a fixed angle  |
| `translation`  | Shifts image in X and Y axes    |

## 🖼️ Examples

| Original | Rotated (90°) | Translated (+5, 0) |
|----------|----------------|--------------------|
| ![](figures/number_1.png) | ![](figures/number_1_rotated.png) | ![](figures/number_1_translated.png) |

## 🚀 Usage

```bash
python3 src/cli.py -c config.yaml
```

This will:
- Load the dataset (e.g., MNIST),
- Apply the transformations as configured,
- Save the transformed dataset in the specified output directory.

## 📚 Reference

If you use this tool or build on it, please cite:

> Romero, A., et al. (2020). *Attentive Group Equivariant Convolutional Networks*. Advances in Neural Information Processing Systems (NeurIPS).