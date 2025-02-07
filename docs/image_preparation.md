# Image Preparation for Convolutional Neural Networks

## Image Preparation Steps

The system prepares images through these key steps:

1. **Resizing Images**
   - Original size: 240×320 pixels
   - New size: 96×128 pixels (makes files 84% smaller)
   - Preserves clear hand shapes while reducing memory use

2. **Color Handling**
   - Keeps color information (RGB)
   - Avoids converting to black-and-white to preserve details

3. **Image Improvements**
   - Adjusts contrast automatically
   - Adds variations through flips, rotations, and zooms
   - Changes brightness/contrast/saturation by ±20%
   - Sharpens edges slightly (±10%)

4. **Normalization**
   - Scales pixel values between 0 and 1
   - Uses batch normalization during training

## Simplifying the Processing

The preparation steps make the system more efficient by:
- Reducing image size: 84% fewer pixels means faster processing
- Applying transformations before resizing: Keeps important details
- Standardizing values: Helps the model learn faster

## How It Works

The image preparation happens in 3 main steps:

1. **Loading Images**
   - Gets photos from folders (240×320 pixels, color)
   - Uses folder names to label rock/paper/scissors

2. **Improving Images**
   - Fixes contrast automatically
   - Adds variations through flips, rotations, and zooms
   - Adjusts brightness/contrast/colors
   - Shrinks images to 96×128 pixels
   - Changes pixel numbers to 0-1 range

3. **Training Setup**
   - Uses special layers to keep values stable
   - Stores processed images for faster training
   - Uses efficient number formats to speed up training

## Checking the System

The program includes tools to verify it works correctly:
- Side-by-side view of original vs processed images
- Tests with 99 sample images
- Live preview of changes
- Shows processing is 84% faster with smaller images

Key numbers:
- Original image size: 230,400 numbers
- New image size: 36,864 numbers 
- First layer calculations reduced by 193,000

<!-- INSERT_BEFORE_AFTER_IMAGE_GRID_HERE -->
