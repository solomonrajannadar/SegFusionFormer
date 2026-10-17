# Dataset Documentation

## 1. Dataset Overview

SegFusionFormer uses the NIH ChestX-ray14 dataset
for multi-label thoracic disease classification.

The dataset contains 112,120 frontal chest radiographs
from 30,805 patients.

It provides image-level annotations for 14 thoracic
disease categories.

## 2. Dataset Source

Dataset Name: NIH ChestX-ray14

Provider: National Institutes of Health (NIH)

Official Download:
https://nihcc.app.box.com/v/ChestXray-NIHCC

The dataset is not included in this GitHub repository.

Users must download it separately.

## 3. Required Files

The dataset preparation process requires:

- Chest radiograph images
- Data_Entry_2017.csv
- BBox_List_2017.csv (optional bounding-box metadata)

Some dataset releases use the filename
Data_Entry_2017_v2020.csv.

If a different metadata filename is downloaded,
the configured CSV path must be updated accordingly.

## 4. Expected Directory Structure

Notebook 01 expects the following local structure:

```text
SegFusionFormer/
|
|-- dataset/
|   |-- images/
|   |   |-- 00000001_000.png
|   |   |-- 00000001_001.png
|   |   |-- ...
|   |
|   |-- Data_Entry_2017.csv
|   |-- BBox_List_2017.csv
|
|-- notebooks/
|   |-- 01_Data_Exploration.ipynb
|   |-- 02_Preprocessing.ipynb
|   |-- ...
|
|-- docs/
|   |-- DATASET.md
|
|-- README.md
|-- requirements.txt
```

The original dataset image archives must be extracted
and organized into the expected images directory.

## 5. Data Preparation

The preprocessing workflow includes:

1. Dataset metadata loading
2. Image availability verification
3. Disease-label analysis
4. Image resizing
5. Intensity normalization
6. Contrast enhancement
7. Multi-label target preparation

The classification pipeline uses 224 × 224 images.

The lung segmentation pipeline uses 256 × 256 images.

## 6. Lung Segmentation Data

The NIH ChestX-ray14 dataset provides disease labels
and bounding-box annotations for selected abnormalities.

These bounding boxes are not equivalent to
pixel-wise lung segmentation masks.

The segmentation training stage requires suitable
lung-mask annotations from a separate source.

The source, license, and preparation of the lung-mask
dataset must be documented before the complete
segmentation pipeline is considered reproducible.

## 7. Training and Evaluation Splits

Patient-wise data partitioning is important for
medical imaging research.

Images belonging to the same patient must not appear
in both training and held-out evaluation partitions.

The final training and evaluation workflow must use
a documented, consistent patient-disjoint split.

The repository's final evaluation results will be
documented after this procedure is verified.

## 8. Data Privacy and Storage

Raw dataset files are excluded from Git tracking
through the project's .gitignore configuration.

Do not upload:

- Full medical imaging datasets
- Identifiable patient information
- Private credentials
- Restricted medical records
- Unnecessary large intermediate files

## 9. Dataset Citation

Wang et al., "ChestX-Ray8: Hospital-Scale Chest X-Ray
Database and Benchmarks on Weakly-Supervised Classification
and Localization of Common Thorax Diseases,"
Proceedings of the IEEE Conference on Computer Vision
and Pattern Recognition (CVPR), 2017.

## 10. Research Disclaimer

SegFusionFormer is an academic research implementation.

The dataset contains automatically derived disease labels
and must not be treated as a clinically validated
diagnostic reference.

The framework is not intended for independent
clinical diagnosis.
