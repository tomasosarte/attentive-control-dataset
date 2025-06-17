# Control Dataset for Testing Equivariance Sensitivity in CNNs

This repository provides a modular dataset and generator for evaluating how well convolutional neural networks generalize under transformation biases (e.g., rotations, translations). The dataset is designed for controlled experimentation, particularly in the context of evaluating **attentive group convolutions** as proposed by [Romero et al., 2020].

## 🧠 Motivation

Group-equivariant CNNs (G-CNNs) enforce equivariance to transformations like rotations and translations. However, they treat all transformations uniformly, regardless of their importance to the task. Attentive G-CNNs address this by learning **which transformations are informative**.

To evaluate this, we introduce a control dataset where **the type and frequency of each transformation is explicitly configurable**. This allows us to test whether models with attention over group transformations generalize better under realistic, imbalanced distributions of transformations.

## ❓ Research Question

> Does introducing attention over group transformations improve the model's generalization ability, particularly when invariance types are imbalanced in the data?

## 📦 Dataset Design

### Transformations Included

Each image in the dataset is modified by one of the following transformations, sampled with configurable proportions:

- **Rotation** — Random angles, e.g., sampled from 15° increments.
- **Horizontal Translation** — Shift along the x-axis.
- **Vertical Translation** — Shift along the y-axis.

All transformations retain the original class label.

### Example

| Original | Rotated | Translated |
|----------|---------|------------|
| ![](figures/number_1.png) | ![](figures/number_1_rotated.png) | ![](figures/number_1_translated.png) |

### Generation Process

You can customize the dataset generation with three parameters:

1. **Transformations** – A list of transformation types to apply.
   _Example:_ `["Rotation", "TranslationH", "TranslationV"]`

2. **Parameters** – Dictionary specifying parameters for each transformation.
   _Example:_
   ```python
   {
       "Rotation": {"angle_range": [0, 360], "step": 15},
       "TranslationH": {"shift_px": [-5, 5]},
       "TranslationV": {"shift_px": [-5, 5]}
   }
   ```

3. **Dataset** – Provide the name of any dataset available in
   `torchvision.datasets` (e.g., `MNIST`, `CIFAR10`, `ImageFolder`). For custom
   folders use `ImageFolder` and set `data_dir` to your dataset root.

With these options defined in a YAML config file you can run:

```bash
python -m src.cli -c path/to/config.yaml
```

