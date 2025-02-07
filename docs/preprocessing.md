# Image Preprocessing for CNNs: A Student Report

This report outlines the image preprocessing steps implemented to prepare images for a Convolutional Neural Network (CNN) used for rock-paper-scissors gesture recognition.

## 1. Preprocessing Steps

The following steps were implemented to optimize images for CNN training:

### 1.1 Image Size
- **Target Resolution**: 96x128 pixels
- **Rationale**: A smaller image size reduces computational complexity and memory requirements, which is crucial for deployment on a Raspberry Pi. While resizing does lose some fine details, the essential features for gesture recognition (hand shape and basic contours) are preserved.

### 1.2 Color Depth
- **Format**: RGB color images (3 channels)
- **Rationale**: The decision was made to retain RGB color information rather than converting to grayscale. Color can provide additional discriminative features, such as skin tone or lighting variations, which might aid in gesture recognition.

### 1.3 Image Enhancement
- **Technique**: Contrast normalization using Keras' AutoContrast layer
- **Rationale**: This technique enhances the contrast across the image, improving feature visibility and helping to handle varying lighting conditions. It clarifies the patterns for the CNN by normalizing the intensity distribution.

### 1.4 Normalization
- **Method**: Pixel values scaled to the range [0, 1]
- **Rationale**: Normalization is crucial for CNN training as it standardizes the input data range. This improves training stability, prevents numerical issues, and helps the model converge faster.

## 2. Impact on CNN Complexity and Feature Extraction

The preprocessing pipeline reduces CNN complexity and improves feature extraction in several ways:

- **Reduced Image Size**: Decreases the number of parameters the CNN needs to learn, reducing computational load and memory requirements.
- **Normalization**: Ensures that all pixel values are within a similar range, preventing any single pixel from dominating the learning process.
- **Contrast Enhancement**: Improves the visibility of important features, making it easier for the CNN to learn relevant patterns.

## 3. Preprocessing Pipeline

The preprocessing pipeline is implemented using Keras layers for seamless integration with the model:

```python
def preprocess_image(inputs):
    # 1. Initial enhancement
    x = keras.layers.AutoContrast(value_range=(0, 255))(inputs)
    
    # 2. Resize to target dimensions
    x = keras.layers.Resizing(96, 128)(x)
    
    # 3. Normalize pixel values
    x = keras.layers.Rescaling(1.0 / 255)(x)
    
    return x
```

### Step-by-Step Explanation
1. **AutoContrast**: The AutoContrast layer normalizes the contrast across the image, enhancing feature visibility.
2. **Resizing**: The Resizing layer resizes the image to 96x128 pixels, reducing computational complexity.
3. **Rescaling**: The Rescaling layer normalizes pixel values to the range [0, 1], improving training stability.

## 4. Conclusion

The preprocessing pipeline optimizes images for CNN training by reducing complexity, improving feature extraction, and ensuring data consistency. These steps are crucial for achieving good performance on the rock-paper-scissors gesture recognition task, especially when deploying on resource-constrained devices like the Raspberry Pi.
# Image Preprocessing Pipeline

## Overview
The preprocessing pipeline is designed to optimize image data for CNN processing while maintaining essential gesture recognition features. The pipeline addresses both computational efficiency and model performance through carefully selected preprocessing steps.

## Preprocessing Steps

### 1. Image Size Optimization
- **Initial Resolution**: 240x320 pixels (captured from Raspberry Pi camera)
- **Target Resolution**: 96x128 pixels
- **Rationale**:
  - Reduces computational complexity
  - Maintains sufficient detail for gesture recognition
  - Balances memory usage with model performance
  - Preserves aspect ratio to avoid distortion

### 2. Color Processing
- **Input Format**: RGB color images
- **Depth**: 3 channels, 8-bit per channel
- **Processing**:
  - Maintains full color information
  - No conversion to grayscale as color may provide useful features
  - RGB format aligns with standard CNN architectures

### 3. Image Enhancement
- **Contrast Normalization**:
  - AutoContrast layer normalizes contrast across images
  - Improves feature visibility
  - Helps handle varying lighting conditions

### 4. Data Normalization
- **Pixel Value Scaling**:
  - Input range: 0-255 (8-bit color depth)
  - Output range: 0-1 (float)
  - Implementation: Values divided by 255
- **Benefits**:
  - Standardizes input data range
  - Improves training stability
  - Helps prevent numerical issues in model calculations

## Pipeline Implementation
The preprocessing pipeline is implemented using Keras layers for efficient integration with the model:

```python
def preprocess_image(inputs):
    # 1. Initial enhancement
    x = keras.layers.AutoContrast(value_range=(0, 255))(inputs)
    
    # 2. Resize to target dimensions
    x = keras.layers.Resizing(96, 128)(x)
    
    # 3. Normalize pixel values
    x = keras.layers.Rescaling(1.0 / 255)(x)
    
    return x
```

## Pipeline Benefits
- **Computational Efficiency**:
  - Reduced memory footprint through resizing
  - Optimized for real-time processing
  - Efficient GPU utilization

- **Model Performance**:
  - Standardized input format
  - Enhanced feature visibility
  - Consistent data scaling

- **Deployment Considerations**:
  - Pipeline integrated into model architecture
  - Consistent preprocessing across training and inference
  - Suitable for embedded deployment on Raspberry Pi
