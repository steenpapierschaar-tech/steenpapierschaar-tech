# Data Augmentation Report: Enhancing a Limited Dataset

This report details how data augmentation was used to improve a rock-paper-scissors gesture classification model, given a small initial dataset.

## 1. Original Dataset

The starting point was a set of 255 images: 85 each of rock, paper, and scissors gestures. These were captured using a Raspberry Pi camera, with a white background and controlled lighting. The images are 240×320 pixels, in RGB format, and stored as JPEGs. The dataset was split into 80% for training (204 images) and 20% for validation (51 images).

### Goals for the Dataset
- Sufficient examples of each gesture
- Balanced classes (rock, paper, scissors)
- Compact size

## 2. Motivation for Augmentation

The limited size of the dataset raised concerns about overfitting. The model might not generalize well to new images with different lighting, angles, or hand positions. Data augmentation was used to create additional training samples from the existing images, helping the model learn more robust features.

### Key Concerns
- Overfitting
- Lack of diversity
- Real-world variations

## 3. Augmentation Methods

To address these concerns, the following transformations were applied:

1. **Spatial Transformations**
   - Rotation (±10 degrees): Introduces slight angle variations.
   - Flips: Handles left/right gestures and upside-down views.
   - Translation (±10%): Shifts the gesture in different directions.
   - Zoom (90–110%): Simulates varying distances.
   - Shear (±10%): Adjusts perspective.

2. **Color and Lighting Adjustments**
   - Brightness (±10%): Adapts to different lighting.
   - Contrast (±10%): Highlights or softens details.
   - Saturation & Hue (±10%): Reflects color temperature changes.
   - Sharpness (90–110%): Mimics focus differences.

## 4. Implementation

TensorFlow and Keras preprocessing layers were used for real-time data augmentation. Instead of creating a large library of augmented images, the transformations were integrated directly into the training loop. During each epoch (10 in total, with a batch size of 32), every image could appear differently.

### Advantages of Real-Time Augmentation
- Conserves storage space
- Ensures variety in each epoch
- Streamlines the training pipeline

## 5. Rationale

Each transformation was chosen to address specific limitations:

- Rotation/Flips/Shear: Handle different hand orientations and camera angles.
- Translation/Zoom: Account for hand distance and positioning.
- Color/Lighting Adjustments: Reflect lighting differences and camera calibration.
- Sharpness: Handle focus or motion blur.

## 6. Before and After Examples

A custom script (augmentation.py) displays original images alongside their augmented versions. This helps visualize the transformations.

```bash
python DL-application/src/augmentation.py
```

## 7. Outcomes

- Improved generalization
- Memory efficiency
- Enhanced robustness

In conclusion, data augmentation effectively expanded the dataset and improved the model's performance on rock-paper-scissors classification tasks.
