# SegFusionFormer

### Segmentation-Guided Explainable Hybrid CNN–Transformer Framework for Multi-Label Thoracic Disease Classification

**SOLOMON RAJAN P (PRK25CS1005) | M.Tech Research Project | Computer Science and Engineering**

**Karunya Institute of Technology and Sciences, Coimbatore, India**

---

## 1. Project Overview

SegFusionFormer is a deep learning research framework developed for automated multi-label thoracic disease classification from chest radiographs.

The framework combines lung segmentation, mask-guided attention, convolutional neural networks, vision transformers, and explainable artificial intelligence.

It is designed to investigate whether anatomical guidance and complementary deep learning features can improve thoracic disease classification while providing interpretable visual explanations.

The implementation uses the NIH ChestX-ray14 dataset and predicts 14 thoracic disease labels.

## 2. Research Motivation

Automated thoracic disease classification is challenging because:

- Multiple abnormalities may occur in the same radiograph.
- Disease manifestations can be subtle or overlapping.
- Medical imaging datasets often exhibit class imbalance.
- Deep learning predictions can be difficult to interpret.
- Models may learn irrelevant background information.

SegFusionFormer investigates segmentation-guided feature learning and adaptive fusion of CNN and transformer representations to address these challenges.

## 3. Proposed Architecture

The framework contains the following principal stages:

1. Chest X-ray preprocessing
2. Lung segmentation using U-Net
3. Segmentation-guided attention
4. CNN feature extraction using EfficientNet-B0
5. Transformer feature extraction using MaxViT-Tiny
6. Adaptive feature fusion
7. Multi-label disease classification
8. Explainable AI using Grad-CAM

   For the proposed system architecture and detailed
component descriptions, see:

**[SegFusionFormer Architecture Diagram](docs/ARCHITECTURE.md)**

### Architecture Workflow

```text
                  Input Chest X-ray
                          |
                          v
                  Image Preprocessing
                          |
                          v
                   U-Net Segmentation
                          |
                          v
                    Binary Lung Mask
                          |
                          v
                  Mask-Guided Attention
                          |
              +-----------+-----------+
              |                       |
              v                       v
        EfficientNet-B0          MaxViT-Tiny
         CNN Features         Transformer Features
              |                       |
              +-----------+-----------+
                          |
                          v
                 Adaptive Feature Fusion
                          |
                          v
                  Classification Head
                          |
                          v
                14 Disease Probabilities
                          |
                          v
                  Prediction Analysis
                          |
                          v
                Grad-CAM Explainability
```

## 4. Dataset

### NIH ChestX-ray14

The NIH ChestX-ray14 dataset is used for multi-label thoracic disease classification.

The dataset includes frontal chest radiographs with disease labels derived from radiology reports.

### Disease Categories

The framework predicts the following 14 labels:

1. Atelectasis
2. Cardiomegaly
3. Effusion
4. Infiltration
5. Mass
6. Nodule
7. Pneumonia
8. Pneumothorax
9. Consolidation
10. Edema
11. Emphysema
12. Fibrosis
13. Pleural Thickening
14. Hernia

The dataset is not included in this repository.

Users must obtain the dataset separately and follow its applicable access and usage requirements.

## 5. Methodology

### 5.1 Data Exploration

Initial dataset exploration examines the available chest radiographs, labels, and disease distributions.

### 5.2 Image Preprocessing

The preprocessing workflow includes image resizing, intensity normalization, and contrast enhancement using CLAHE.

The classification stage uses images resized to 224 × 224 pixels.

### 5.3 Lung Segmentation

A U-Net architecture is implemented for lung-field segmentation.

The segmentation network is trained using a combination of Binary Cross-Entropy and Dice loss.

The segmentation stage uses images resized to 256 × 256 pixels.

### 5.4 Segmentation-Guided Attention

The lung mask is combined with the chest radiograph to construct a two-channel input.

A learnable attention module uses the mask to guide attention toward the lung regions.

The enhanced image is subsequently supplied to the classification backbone.

### 5.5 Hybrid CNN–Transformer Classification

Two complementary feature extraction branches are implemented:

**EfficientNet-B0**

Extracts convolutional image features.

**MaxViT-Tiny**

Extracts features using a vision transformer architecture.

### 5.6 Adaptive Feature Fusion

An adaptive fusion module combines the features from EfficientNet-B0 and MaxViT-Tiny.

Learnable fusion weights determine the relative contribution of each feature branch.

The fused representation is used for multi-label disease prediction.

### 5.7 Explainable Artificial Intelligence

Grad-CAM is incorporated to visualize image regions contributing to model predictions.

The implementation also provides:

- Lung-mask visualization
- Mask-guided attention visualization
- Grad-CAM heatmaps
- Adaptive fusion weight analysis

## 6. Repository Structure

```text
SegFusionFormer/
|
|-- README.md
|-- .gitignore
|
|-- notebooks/
|   |-- 01_Data_Exploration.ipynb
|   |-- 02_Preprocessing.ipynb
|   |-- 03_Lung_Segmentation.ipynb
|   |-- 04_Disease_Classification.ipynb
|   |-- 05_Explainable_AI_Analysis.ipynb
|   |-- 06_Performance_Evaluation.ipynb
|   |-- 07_Final_Inference_and_Demo.ipynb
|   |-- 08_Final_Validation_and_Cleanup.ipynb
|   |-- README.md
```

Additional documentation, dependency specifications, and reproducibility resources are being prepared.

## 7. Jupyter Notebook Workflow

| Notebook | Description |
|----------|-------------|
| 01_Data_Exploration | Dataset exploration and label analysis |
| 02_Preprocessing | Image preprocessing and enhancement |
| 03_Lung_Segmentation | U-Net lung segmentation |
| 04_Disease_Classification | SegFusionFormer architecture and model training |
| 05_Explainable_AI_Analysis | Grad-CAM and fusion analysis |
| 06_Performance_Evaluation | Classification performance evaluation |
| 07_Final_Inference_and_Demo | End-to-end inference and visualization |
| 08_Final_Validation_and_Cleanup | Model validation and project checks |

## 8. Technology Stack

**Programming Language**
- Python

**Deep Learning**
- PyTorch
- Torchvision
- TIMM

**Medical Image Processing**
- OpenCV
- NumPy
- Pillow

**Data Analysis**
- Pandas
- Scikit-learn

**Visualization**
- Matplotlib
- Seaborn

**Development Environment**
- Jupyter Notebook

Exact dependency versions and installation instructions will be added after environment verification.

## 9. Model Evaluation

The implementation includes the following evaluation metrics:

- Area Under the Receiver Operating Characteristic Curve (AUROC)
- Precision
- Recall
- F1-score
- Confusion matrices
- Precision–Recall curves
- Class-wise performance analysis

The final reported metrics will be documented after verification of the patient-disjoint evaluation protocol.

## 10. Reproducibility Status

The research notebooks are available in this repository.

The following improvements are being prepared:

- Standardized dataset paths
- Verified Python dependency versions
- Consistent patient-wise data partitioning
- Reproducible training and evaluation instructions
- Model checkpoint documentation
- Verified experimental results

The current notebooks contain local file paths and require configuration before execution on another system.

## 11. Research Applications

This project explores:

- Medical image classification
- Lung-field segmentation
- Hybrid CNN–Transformer learning
- Segmentation-guided attention
- Multi-label prediction
- Explainable artificial intelligence
- Adaptive feature fusion

**Research Disclaimer:** This framework is intended for academic research and experimentation. It is not clinically validated and must not be used independently for medical diagnosis.

## 12. Future Work

Potential research extensions include:

- External dataset validation
- Cross-dataset generalization
- Additional explainability evaluation
- Model optimization
- Robustness analysis
- Comparative studies against established architectures

## 13. Author

**Solomon Rajan P**

M.Tech – Computer Science and Engineering

Karunya Institute of Technology and Sciences

Coimbatore, Tamil Nadu, India

**GitHub:** [solomonrajannadar](https://github.com/solomonrajannadar)

## 14. Acknowledgements

This research project was developed as part of the M.Tech Computer Science and Engineering programme at Karunya Institute of Technology and Sciences.

The project uses publicly released research datasets and established open-source deep learning frameworks.

## 15. Citation and Code Availability

### Citation

If you use SegFusionFormer in your research,
please cite this software repository.

Citation metadata is available in:

[CITATION.cff](CITATION.cff)

The associated journal manuscript citation
will be added when publication details are available.

### Code Availability

The SegFusionFormer research implementation is available at:

https://github.com/solomonrajannadar/SegFusionFormer

The repository includes:

- Eight Jupyter research notebooks
- Model architecture implementations
- Training and evaluation procedures
- Explainable AI implementation
- Experimental figures and numerical results
- Dataset documentation
- Reproducibility instructions

### Data Availability

The NIH ChestX-ray14 dataset can be obtained
from its official distribution source.

The dataset is not redistributed in this repository.

### Model Availability

Trained model checkpoints are maintained separately
from the public GitHub repository.

See [Model Documentation](models/README.md)
for checkpoint information.

### Reproducibility

See the [Reproducibility Guide](docs/REPRODUCIBILITY.md)
for implementation details, setup instructions,
and current reproducibility limitations.

## Project Review Status

SegFusionFormer is an academic M.Tech research implementation
for multi-label thoracic disease classification using
NIH ChestX-ray14.

The repository contains eight research notebooks,
saved experimental figures, numerical results,
and reproducibility documentation.

### Recorded Experimental Results

- Macro AUROC: 0.8179
- Micro AUROC: 0.8358
- Macro F1-score: 0.3224
- Micro F1-score: 0.3469
- U-Net validation Dice: 0.9601

Classification results originate from the original
image-level validation experiment using 22,424 images.

The original classification split contains patient overlap.
These results are not independent test-set performance.

The U-Net Dice score was recorded on a separate
CHNCXR segmentation validation dataset.

A patient-disjoint dataset partition was subsequently
prepared, but the reported classification metrics
were not generated using that partition.

Some NIH lung masks are incomplete, and the original
checkpoints and datasets are not distributed in GitHub.

The project is intended for academic research
and demonstration, not clinical diagnosis.

---

**Project:** SegFusionFormer

**Research Area:** Deep Learning | Medical Imaging | Explainable AI

**Status:** Research implementation and reproducibility documentation in progress.
