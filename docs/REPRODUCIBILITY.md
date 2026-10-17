# SegFusionFormer: Reproducibility Guide

## Segmentation-Guided Explainable Hybrid CNN–Transformer Framework for Multi-Label Thoracic Disease Classification

**Project Type:** M.Tech Research Project  
**Research Area:** Deep Learning, Medical Image Analysis, Explainable AI  
**Institution:** Karunya Institute of Technology and Sciences, India  
**Author:** Solomon Rajan P

---

## 1. Introduction

This document describes the implementation environment,
dataset requirements, execution workflow, and reproducibility
considerations for the SegFusionFormer research framework.

SegFusionFormer integrates:

- U-Net lung segmentation
- Segmentation-guided attention
- EfficientNet-B0 convolutional feature extraction
- MaxViT-Tiny transformer-based feature extraction
- Adaptive feature fusion
- Multi-label thoracic disease classification
- Grad-CAM explainability

The implementation is organized into eight Jupyter notebooks.

This guide describes the available research implementation
and identifies additional materials required to reproduce
the experimental results.

---

## 2. Research Implementation

The proposed framework consists of two principal stages.

### Stage 1: Lung Segmentation

A U-Net architecture is used to generate lung-field
segmentation masks from chest radiographs.

The segmentation network accepts single-channel
grayscale images of resolution 256 × 256.

The implementation uses an encoder-decoder architecture
with skip connections.

### Stage 2: Disease Classification

The classification framework combines:

1. Chest X-ray input
2. Lung segmentation mask
3. Mask-guided attention
4. EfficientNet-B0 features
5. MaxViT-Tiny features
6. Adaptive feature fusion
7. Multi-label classification

The classification input resolution is 224 × 224.

The model predicts probabilities for 14 thoracic
disease categories.

---

## 3. Development Environment

### 3.1 Programming Language

Python 3.11 is recommended for the documented
local development environment.

### 3.2 Deep Learning Framework

The project uses PyTorch.

Additional libraries include:

- Torchvision
- TIMM
- OpenCV
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Pillow
- tqdm
- JupyterLab
- ipykernel

The dependencies are listed in the repository's
requirements.txt file.

### 3.3 Locally Verified Environment

The following configuration was used during
local repository verification:

| Component | Configuration |
|-----------|---------------|
| Operating System | macOS |
| Hardware | Apple Silicon MacBook Air |
| Python | 3.11.17 |
| PyTorch | 2.14.1 |
| Conda Environment | segfusionformer |
| Apple MPS Backend | Available |
| Development Environment | Visual Studio Code |
| Notebook Execution | Jupyter |

This configuration was verified for package imports,
dataset loading, and selected U-Net inference operations.

It is not presented as a complete reproduction of
all training and evaluation experiments.

---

## 4. Repository Structure

```text
SegFusionFormer/
│
├── notebooks/
│   ├── 01_Data_Exploration.ipynb
│   ├── 02_Preprocessing.ipynb
│   ├── 03_Lung_Segmentation.ipynb
│   ├── 04_Disease_Classification.ipynb
│   ├── 05_Explainable_AI_Analysis.ipynb
│   ├── 06_Performance_Evaluation.ipynb
│   ├── 07_Final_Inference_and_Demo.ipynb
│   └── 08_Final_Validation_and_Cleanup.ipynb
│
├── docs/
│   ├── DATASET.md
│   └── REPRODUCIBILITY.md
│
├── config.py
├── requirements.txt
├── README.md
└── .gitignore
```

Large datasets and model checkpoints are maintained
outside the public Git repository.

---

## 5. Installation Instructions

### Step 1: Clone the Repository

```bash
git clone https://github.com/solomonrajannadar/SegFusionFormer.git

cd SegFusionFormer
```

### Step 2: Create a Python Environment

Install Conda if it is not already available.

Create the environment:

```bash
conda create -n segfusionformer python=3.11 -y
```

Activate it:

```bash
conda activate segfusionformer
```

### Step 3: Install Dependencies

```bash
python -m pip install --upgrade pip

python -m pip install -r requirements.txt
```

The current requirements.txt lists the main packages
without fixed version constraints.

Exact dependency locking remains necessary for
fully repeatable environment reconstruction.

### Step 4: Verify Installation

```bash
python -c "import torch, torchvision, timm, cv2, numpy, pandas, sklearn; print('Dependencies imported successfully')"
```

The command should complete without import errors.

---

## 6. Dataset Requirements

### 6.1 NIH ChestX-ray14

The primary disease classification dataset is
NIH ChestX-ray14.

It contains:

- 112,120 chest radiographs
- 30,805 patients
- 14 thoracic disease labels
- Image-level metadata

Official dataset source:

https://nihcc.app.box.com/v/ChestXray-NIHCC

Additional information is provided in:

[Dataset Documentation](DATASET.md)

### 6.2 Lung Segmentation Dataset

The U-Net segmentation experiment uses CHNCXR
images with corresponding lung segmentation masks.

The segmentation dataset is separate from
the NIH ChestX-ray14 classification dataset.

The segmentation notebook contains the relevant
training and validation procedures.

Dataset access, redistribution permissions, and
annotation provenance must be verified separately.

---

## 7. Dataset Directory Configuration

The project expects the following local organization:

```text
SegFusionFormer/
│
├── dataset/
│   ├── images/
│   ├── lung_masks/
│   ├── Data_Entry_2017.csv
│   ├── BBox_List_2017.csv
│   ├── train_val_list.txt
│   └── test_list.txt
│
├── segmentation_dataset/
│   ├── images/
│   └── masks/
│
├── notebooks/
├── config.py
└── requirements.txt
```

The dataset folders are excluded from Git tracking.

### External Dataset Storage

The central configuration supports an external
classification dataset directory through the
SEGFUSION_DATA_DIR environment variable.

For example:

```bash
export SEGFUSION_DATA_DIR="/absolute/path/to/dataset"
```

Alternatively, a local symbolic link may be used.

The segmentation training dataset and several
existing notebooks still require local path
configuration before execution on another system.

---

## 8. Jupyter Notebook Execution Order

The recommended logical execution sequence is:

| Order | Notebook | Purpose |
|-------|----------|---------|
| 01 | Data Exploration | Dataset analysis |
| 02 | Preprocessing | Image preparation |
| 03 | Lung Segmentation | U-Net segmentation |
| 04 | Disease Classification | SegFusionFormer training |
| 05 | Explainable AI Analysis | Grad-CAM and feature analysis |
| 06 | Performance Evaluation | Classification evaluation |
| 07 | Final Inference and Demo | Prediction demonstration |
| 08 | Final Validation and Cleanup | Project verification |

The notebooks document the research workflow.

Some notebooks contain absolute local file paths,
previously generated outputs, and checkpoint dependencies.

Consequently, the sequence should not yet be treated
as a fully automated end-to-end execution pipeline.

---

## 9. Lung Segmentation

### Architecture

The segmentation component uses U-Net.

Input:

- Single-channel grayscale chest X-ray
- Image resolution: 256 × 256

Output:

- Lung segmentation probability map
- Binary lung mask

### Training

The segmentation notebook includes training and
validation procedures using annotated lung images.

The recorded best checkpoint is:

```text
best_unet.pth
```

### NIH Lung Mask Generation

The trained segmentation model is applied to
NIH ChestX-ray14 images to generate lung masks.

The generated masks use the naming format:

```text
00000001_000.png

00000001_000_mask.png
```

The original project contains matching mask
filenames for all 112,120 NIH chest radiographs.

Filename matching alone does not establish
anatomical segmentation accuracy.

The existing mask-generation procedure skips
mask files that are already present.

The precise checkpoint provenance of those existing
masks has not been independently established.

---

## 10. Disease Classification

The classification stage implements SegFusionFormer.

### Backbone Networks

- EfficientNet-B0
- MaxViT-Tiny

### Main Components

- Segmentation-guided attention
- CNN feature extraction
- Transformer feature extraction
- Adaptive feature fusion
- Multi-label classification head

### Model Output

The network produces 14 disease probabilities
for each chest radiograph.

The classification notebook contains the training
configuration, loss computation, optimization,
and checkpoint-saving procedures.

The recorded classification checkpoint is:

```text
best_segfusionformer.pth
```

---

## 11. Experimental Evaluation

The project implementation includes:

- AUROC evaluation
- Precision
- Recall
- F1-score
- Confusion matrices
- Precision-recall analysis
- Class-wise performance analysis

The evaluation notebook contains the associated
metric calculations and visualization procedures.

### Data Partitioning

The classification notebook contains an initial
image-level splitting procedure and a later
patient-wise splitting procedure.

Some downstream analysis notebooks also construct
image-level random splits.

Before using the repository as evidence of exact
experimental reproducibility, the partition used
for each reported manuscript result must be identified.

Patient-wise evaluation should prevent the same
patient from appearing in training and held-out
evaluation partitions.

Results from different partitioning protocols
must not be presented as directly equivalent.

---

## 12. Explainable Artificial Intelligence

The implementation includes Grad-CAM-based
visual explanations.

The explainability notebook contains procedures
for examining model predictions and feature
contributions.

The analysis includes:

- Grad-CAM heatmaps
- Prediction visualization
- Segmentation-mask visualization
- Mask-guided attention analysis
- Adaptive fusion analysis

Explainability visualizations support qualitative
model interpretation.

They should not be interpreted as independent
clinical validation.

---

## 13. Trained Model Checkpoints

The original research implementation uses:

```text
models/
├── best_unet.pth
└── best_segfusionformer.pth
```

The checkpoints are not included in the public
Git repository.

Execution of checkpoint-dependent notebooks
requires access to the corresponding trained weights.

The checkpoint distribution method and usage
instructions will be documented separately.

---

## 14. Reproducibility Status

### Available Materials

- Eight research notebooks
- Model architecture implementations
- Training and evaluation code
- Dataset documentation
- Python dependency list
- Centralized project configuration

### Locally Verified Operations

- Python environment setup
- Dependency imports
- NIH metadata loading
- Image-file availability
- Image-mask filename matching
- U-Net checkpoint loading
- Selected U-Net inference operations

### Additional Requirements for Exact Reproduction

- Verified package versions
- Accessible model checkpoints
- Complete dataset access
- Segmentation annotation provenance
- Configuration of remaining local paths
- Exact data-partition information
- Identification of the checkpoints associated
  with reported manuscript results

The presence of code and saved notebook outputs
does not establish that all experiments have been
independently reproduced.

---

## 15. Research Limitations

SegFusionFormer is an academic research framework.

Segmentation quality may vary across datasets
and image acquisition conditions.

The U-Net validation performance recorded on
its original segmentation dataset must not be
interpreted as equivalent performance on NIH
ChestX-ray14 without independent evaluation.

The research implementation is not clinically
validated and is not intended for independent
medical diagnosis.

---

## 16. Author

**Solomon Rajan P**

M.Tech Computer Science and Engineering

Karunya Institute of Technology and Sciences

Coimbatore, Tamil Nadu, India

GitHub:

https://github.com/solomonrajannadar

---

## 17. Research Code Availability

The research implementation is available at:

https://github.com/solomonrajannadar/SegFusionFormer

Raw medical imaging datasets and large model
checkpoints are not distributed directly through
the repository.

Users should consult the dataset documentation
and checkpoint-access information.

---

**Project:** SegFusionFormer

**Document Type:** Research Reproducibility Guide

**Purpose:** Supporting documentation for academic
research review and experimental inspection.
