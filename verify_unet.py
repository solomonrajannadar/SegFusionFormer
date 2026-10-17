# ==========================================================
# SegFusionFormer
# U-Net Segmentation Verification and Confidence Analysis
#
# M.Tech Computer Science and Engineering
# Karunya Institute of Technology and Sciences
#
# Purpose:
# 1. Load the trained U-Net model
# 2. Generate fresh lung segmentation predictions
# 3. Compare fresh predictions with existing masks
# 4. Analyze prediction confidence
# 5. Visualize original images and segmentation results
#
# This script does NOT modify existing dataset files.
# ==========================================================


# ==========================================================
# 1. IMPORT REQUIRED LIBRARIES
# ==========================================================

import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from config import IMAGE_DIR, DATASET_DIR


print("=" * 65)
print("SEGFUSIONFORMER - U-NET SEGMENTATION VERIFICATION")
print("=" * 65)


# ==========================================================
# 2. U-NET ARCHITECTURE
# ==========================================================

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        return self.block(x)


# ==========================================================
# 3. DOWN-SAMPLING BLOCK
# ==========================================================

class Down(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.block = nn.Sequential(

            nn.MaxPool2d(2),

            DoubleConv(
                in_channels,
                out_channels
            )
        )

    def forward(self, x):

        return self.block(x)


# ==========================================================
# 4. UP-SAMPLING BLOCK
# ==========================================================

class Up(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels,
            in_channels // 2,
            kernel_size=2,
            stride=2
        )

        self.conv = DoubleConv(
            in_channels,
            out_channels
        )

    def forward(self, x1, x2):

        x1 = self.up(x1)

        # Match spatial dimensions if necessary
        if x1.shape[2:] != x2.shape[2:]:

            x1 = nn.functional.interpolate(
                x1,
                size=x2.shape[2:],
                mode="bilinear",
                align_corners=False
            )

        # Concatenate skip connection
        x = torch.cat(
            [x2, x1],
            dim=1
        )

        return self.conv(x)


# ==========================================================
# 5. COMPLETE U-NET MODEL
# ==========================================================

class UNet(nn.Module):

    def __init__(self):

        super().__init__()

        # Encoder
        self.inc = DoubleConv(1, 32)

        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)

        # Decoder
        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)
        self.up4 = Up(64, 32)

        # Output layer
        self.outc = nn.Conv2d(
            32,
            1,
            kernel_size=1
        )

    def forward(self, x):

        # Encoder
        x1 = self.inc(x)

        x2 = self.down1(x1)

        x3 = self.down2(x2)

        x4 = self.down3(x3)

        x5 = self.down4(x4)

        # Decoder
        x = self.up1(x5, x4)

        x = self.up2(x, x3)

        x = self.up3(x, x2)

        x = self.up4(x, x1)

        return self.outc(x)


# ==========================================================
# 6. LOAD TRAINED U-NET CHECKPOINT
# ==========================================================

MODEL_PATH = (
    Path.home()
    / "Desktop/Mtech_final_project/SegFusionFormer"
    / "models/best_unet.pth"
)

if not MODEL_PATH.is_file():

    raise FileNotFoundError(
        f"U-Net checkpoint not found: {MODEL_PATH}"
    )


# Use CPU for reproducible diagnostic inference
device = torch.device("cpu")

print("\nLoading trained U-Net checkpoint...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

# Initialize model
model = UNet().to(device)

# Load trained weights
model.load_state_dict(
    checkpoint["model_state_dict"],
    strict=True
)

# Set evaluation mode
model.eval()

print("\nModel loaded successfully!")

print(
    "Saved Epoch:",
    checkpoint.get("epoch", "Not available")
)

print(
    "Validation Dice:",
    checkpoint.get("val_dice", "Not available")
)

print(
    "Validation Loss:",
    checkpoint.get("val_loss", "Not available")
)

print("Device:", device)


# ==========================================================
# 7. DEFINE TEST IMAGES
# ==========================================================

test_images = [

    # Training samples
    "00014825_001.png",
    "00001569_001.png",
    "00023305_000.png",

    # Validation samples
    "00003406_004.png",
    "00014089_001.png",
    "00012086_009.png"

]
MASK_DIR = DATASET_DIR / "lung_masks"

IMAGE_SIZE = 256

THRESHOLD = 0.5


print("\n" + "=" * 65)
print("DATASET CONFIGURATION")
print("=" * 65)

print("Images Directory:", IMAGE_DIR)

print("Masks Directory:", MASK_DIR)

print("Number of Test Images:", len(test_images))

print("Image Size:", IMAGE_SIZE)

print("Prediction Threshold:", THRESHOLD)


# ==========================================================
# 8. CREATE VISUALIZATION FIGURE
# ==========================================================

fig, axes = plt.subplots(

    nrows=len(test_images),

    ncols=4,

    figsize=(16, 12)

)


# ==========================================================
# 9. RUN U-NET INFERENCE
# ==========================================================

for i, filename in enumerate(test_images):

    print("\n" + "=" * 65)
    print(f"PROCESSING IMAGE {i + 1}: {filename}")
    print("=" * 65)

    # ------------------------------------------------------
    # 9.1 Locate image and existing mask
    # ------------------------------------------------------

    image_path = IMAGE_DIR / filename

    mask_path = (
        MASK_DIR /
        f"{Path(filename).stem}_mask.png"
    )

    # ------------------------------------------------------
    # 9.2 Read original image
    # ------------------------------------------------------

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE
    )

    # Read existing segmentation mask
    saved_mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:

        raise FileNotFoundError(
            f"Chest X-ray not found: {image_path}"
        )

    if saved_mask is None:

        raise FileNotFoundError(
            f"Existing mask not found: {mask_path}"
        )

    # ------------------------------------------------------
    # 9.3 Resize image and mask
    # ------------------------------------------------------

    image = cv2.resize(
        image,
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    saved_mask = cv2.resize(
        saved_mask,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_NEAREST
    )

    # ------------------------------------------------------
    # 9.4 Normalize image
    # ------------------------------------------------------

    input_image = (
        image.astype(np.float32) / 255.0
    )

    # Convert image to PyTorch tensor
    tensor = torch.from_numpy(
        input_image
    ).unsqueeze(0).unsqueeze(0)

    tensor = tensor.to(device)

    # ------------------------------------------------------
    # 9.5 Generate fresh U-Net prediction
    # ------------------------------------------------------

    with torch.inference_mode():

        logits = model(tensor)

        probabilities = torch.sigmoid(logits)

    probability_mask = (
        probabilities
        .squeeze()
        .cpu()
        .numpy()
    )


    # ======================================================
    # 10. U-NET PREDICTION CONFIDENCE ANALYSIS
    # ======================================================

    print("\nU-NET PREDICTION CONFIDENCE")

    print("-" * 50)

    print(
        "Minimum Probability:",
        round(float(probability_mask.min()), 4)
    )

    print(
        "Maximum Probability:",
        round(float(probability_mask.max()), 4)
    )

    print(
        "Mean Probability:",
        round(float(probability_mask.mean()), 4)
    )


    # ======================================================
    # 11. THRESHOLD ANALYSIS
    # ======================================================

    print("\nLUNG COVERAGE AT DIFFERENT THRESHOLDS")

    print("-" * 50)

    thresholds = [
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7
    ]

    for threshold in thresholds:

        binary_prediction = (
            probability_mask >= threshold
        )

        coverage = (
            binary_prediction.mean() * 100
        )

        print(
            f"Threshold {threshold:.1f}: "
            f"{coverage:.2f}%"
        )


    # ======================================================
    # 12. FINAL BINARY PREDICTION
    # ======================================================

    # Use the original threshold of 0.5
    fresh_mask = (
        probability_mask >= THRESHOLD
    )

    # Convert saved mask to binary
    saved_binary = (
        saved_mask > 127
    )


    # ======================================================
    # 13. CALCULATE MASK COVERAGE
    # ======================================================

    saved_coverage = (
        saved_binary.mean() * 100
    )

    fresh_coverage = (
        fresh_mask.mean() * 100
    )

    print("\nMASK COVERAGE COMPARISON")

    print("-" * 50)

    print(
        f"Existing Mask Coverage: "
        f"{saved_coverage:.2f}%"
    )

    print(
        f"Fresh Mask Coverage: "
        f"{fresh_coverage:.2f}%"
    )


    # ======================================================
    # 14. CALCULATE MASK AGREEMENT IoU
    # ======================================================

    intersection = np.logical_and(
        saved_binary,
        fresh_mask
    ).sum()

    union = np.logical_or(
        saved_binary,
        fresh_mask
    ).sum()

    if union > 0:

        mask_iou = intersection / union

        print(
            f"Saved/Fresh Mask IoU: "
            f"{mask_iou:.4f}"
        )

    else:

        mask_iou = float("nan")

        print(
            "Saved/Fresh Mask IoU: N/A "
            "(both masks are empty)"
        )

    # Important:
    # This IoU measures agreement between predictions.
    # It is NOT ground-truth segmentation accuracy.


    # ======================================================
    # 15. VISUALIZATION
    # ======================================================

    # ------------------------------------------------------
    # Column 1: Original chest X-ray
    # ------------------------------------------------------

    axes[i, 0].imshow(
        image,
        cmap="gray"
    )

    axes[i, 0].set_title(
        f"Original X-ray\n{filename}",
        fontsize=10
    )


    # ------------------------------------------------------
    # Column 2: Existing lung mask
    # ------------------------------------------------------

    axes[i, 1].imshow(
        image,
        cmap="gray"
    )

    existing_overlay = np.ma.masked_where(
        ~saved_binary,
        saved_binary.astype(float)
    )

    axes[i, 1].imshow(
        existing_overlay,
        cmap="autumn",
        alpha=0.4,
        vmin=0,
        vmax=1
    )

    axes[i, 1].set_title(
        f"Existing Mask\n"
        f"Coverage: {saved_coverage:.2f}%",
        fontsize=10
    )


    # ------------------------------------------------------
    # Column 3: Fresh U-Net prediction
    # ------------------------------------------------------

    axes[i, 2].imshow(
        image,
        cmap="gray"
    )

    fresh_overlay = np.ma.masked_where(
        ~fresh_mask,
        fresh_mask.astype(float)
    )

    axes[i, 2].imshow(
        fresh_overlay,
        cmap="autumn",
        alpha=0.4,
        vmin=0,
        vmax=1
    )

    axes[i, 2].set_title(
        f"Fresh U-Net Prediction\n"
        f"Coverage: {fresh_coverage:.2f}%",
        fontsize=10
    )


    # ------------------------------------------------------
    # Column 4: Raw prediction probability
    # ------------------------------------------------------

    axes[i, 3].imshow(
        probability_mask,
        cmap="viridis",
        vmin=0,
        vmax=1
    )

    axes[i, 3].set_title(
        "U-Net Prediction Probability",
        fontsize=10
    )


    # ------------------------------------------------------
    # Remove axis labels
    # ------------------------------------------------------

    for ax in axes[i]:

        ax.axis("off")


    print(
        f"\nImage {i + 1} processed successfully."
    )


# ==========================================================
# 16. DISPLAY COMPLETE VISUALIZATION
# ==========================================================

plt.suptitle(
    "SegFusionFormer - U-Net Segmentation Verification",
    fontsize=15,
    fontweight="bold"
)

plt.tight_layout(
    rect=[0, 0, 1, 0.96]
)

plt.show()


# ==========================================================
# 17. FINAL REPORT
# ==========================================================

print("\n" + "=" * 65)

print("SEGFUSIONFORMER - SEGMENTATION DIAGNOSTIC COMPLETED")

print("=" * 65)

print("Total Images Tested:", len(test_images))

print("Model:", "U-Net")

print("Checkpoint:", MODEL_PATH.name)

print("Prediction Threshold:", THRESHOLD)

print("Image Resolution:", f"{IMAGE_SIZE} x {IMAGE_SIZE}")

print("\nAll diagnostic operations completed successfully.")

print("Existing images and masks were not modified.")