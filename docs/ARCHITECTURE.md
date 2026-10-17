# SegFusionFormer — System Architecture

## Proposed Architecture

SegFusionFormer is a segmentation-guided, explainable
hybrid CNN–Transformer framework for multi-label
thoracic disease classification.

```mermaid
flowchart TD
    A["Input Chest X-ray"] --> B["Image Preprocessing"]

    B --> C["U-Net Lung Segmentation"]
    C --> D["Binary Lung Mask"]

    B --> E["Preprocessed Chest X-ray"]
    D --> F["Two-Channel Input: X-ray + Lung Mask"]
    E --> F

    F --> G["Mask-Guided Attention"]
    G --> H["Attended Image"]

    H --> I["EfficientNet-B0 CNN Branch"]
    H --> J["MaxViT-Tiny Transformer Branch"]

    I --> K["Adaptive Feature Fusion"]
    J --> K

    K --> L["Multi-Label Classification Head"]
    L --> M["Sigmoid Activation"]
    M --> N["14 Thoracic Disease Probabilities"]

    I -.-> O["Grad-CAM Explainability"]
    N -.-> O

    style A fill:#E3F2FD,stroke:#1565C0,color:#000
    style C fill:#E8F5E9,stroke:#2E7D32,color:#000
    style G fill:#FFF3E0,stroke:#EF6C00,color:#000
    style I fill:#E3F2FD,stroke:#1565C0,color:#000
    style J fill:#E8F5E9,stroke:#2E7D32,color:#000
    style K fill:#F3E5F5,stroke:#7B1FA2,color:#000
    style N fill:#FFEBEE,stroke:#C62828,color:#000
    style O fill:#E0F7FA,stroke:#00838F,color:#000
```

## Architecture Components

### 1. Input and Preprocessing

Chest radiographs are loaded and preprocessed
before segmentation and classification.

### 2. Lung Segmentation

A U-Net architecture generates lung-field
segmentation masks.

### 3. Segmentation-Guided Attention

The chest X-ray and its corresponding lung mask
are combined to guide feature learning toward
anatomically relevant regions.

### 4. Hybrid Feature Extraction

EfficientNet-B0 extracts convolutional features,
while MaxViT-Tiny extracts complementary
transformer-based representations.

### 5. Adaptive Feature Fusion

The framework combines the extracted features
using a learnable adaptive fusion mechanism.

### 6. Multi-Label Classification

The classification head produces probabilities
for 14 thoracic disease categories.

### 7. Explainable AI

Grad-CAM provides visual explanations of
model predictions.

## Implementation

The corresponding implementations are available in:

- `notebooks/03_Lung_Segmentation.ipynb`
- `notebooks/04_Disease_Classification.ipynb`
- `notebooks/05_Explainable_AI_Analysis.ipynb`

This diagram is a conceptual representation of
the implemented research architecture.
