# SegFusionFormer — Trained Model Checkpoints

## Overview

This directory documents the trained deep learning
models used in the SegFusionFormer research framework.

The implementation includes two principal models:

1. U-Net for lung-field segmentation.
2. SegFusionFormer for multi-label thoracic disease
   classification.

Large trained checkpoints are not committed directly
to this Git repository.

---

## 1. Lung Segmentation Model

**Checkpoint:** `best_unet.pth`

**Architecture:** U-Net

**Purpose:** Lung-field segmentation from chest X-rays.

### Model Configuration

- Framework: PyTorch
- Input: Grayscale chest radiograph
- Input resolution: 256 × 256
- Output: Lung segmentation probability map
- Binary segmentation threshold: 0.5

### Checkpoint Information

The saved checkpoint contains:

- Training epoch
- Model state dictionary
- Optimizer state dictionary
- Validation Dice
- Validation loss

The checkpoint structure has been verified using PyTorch.

### Implementation

[U-Net Segmentation Notebook](../notebooks/03_Lung_Segmentation.ipynb)

---

## 2. SegFusionFormer Classification Model

**Checkpoint:** `best_segfusionformer.pth`

**Architecture:** Segmentation-Guided Hybrid CNN–Transformer

**Purpose:** Multi-label thoracic disease classification.

### Main Components

- Segmentation-guided attention
- EfficientNet-B0
- MaxViT-Tiny
- Adaptive feature fusion
- Multi-label classification head

### Model Input

The classification framework combines chest X-ray
information with lung segmentation guidance.

### Model Output

The model generates probabilities for 14 thoracic
disease categories.

### Implementation

[Classification Notebook](../notebooks/04_Disease_Classification.ipynb)

---

## 3. Expected Checkpoint Structure

For local execution, the checkpoints may be organized as:

```text
SegFusionFormer/
|
|-- models/
|   |-- best_unet.pth
|   |-- best_segfusionformer.pth
|   |-- README.md
|
|-- notebooks/
|-- config.py
|-- requirements.txt
```

Existing notebooks may require local checkpoint-path
configuration before execution.

---

## 4. Checkpoint Availability

The trained checkpoints are maintained separately
from the public Git repository.

Checkpoint download links have not yet been published.

The checkpoint distribution method will be documented
after confirming the applicable journal review and
research-sharing requirements.

---

## 5. Reproducibility

Reproducing the model predictions requires:

1. The corresponding model architecture.
2. The correct trained checkpoint.
3. The appropriate Python dependencies.
4. The required preprocessing configuration.
5. The relevant dataset inputs.

The research notebooks provide the principal
implementation details.

For further information, refer to:

[Reproducibility Guide](../docs/REPRODUCIBILITY.md)

---

## 6. Research Disclaimer

These models were developed for academic research.

The framework is not clinically validated and must
not be used independently for medical diagnosis.