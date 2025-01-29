import sys
from pathlib import Path
import cv2 as cv

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

import keras
import matplotlib.pyplot as plt
from src.config import config
from src.create_dataset import create_dataset

def apply_augmentation(inputs):
    """Apply image augmentation pipeline to input layer"""
    # Start with original image dimensions
    x = inputs
    
    # 1. Initial preprocessing
    x = keras.layers.AutoContrast(value_range=(0, 255))(x)
    
    # 2. Spatial augmentations (needs original image dimensions)
    x = keras.layers.RandomFlip(mode="horizontal_and_vertical")(x)
    x = keras.layers.RandomRotation(factor=config.RANDOM_ROTATION)(x)
    x = keras.layers.RandomShear(
        x_factor=config.RANDOM_SHEAR_X,
        y_factor=config.RANDOM_SHEAR_Y
    )(x)
    x = keras.layers.RandomTranslation(
        height_factor=config.RANDOM_TRANSLATION,
        width_factor=config.RANDOM_TRANSLATION
    )(x)
    x = keras.layers.RandomZoom(
        height_factor=config.RANDOM_ZOOM,
        width_factor=config.RANDOM_ZOOM
    )(x)
    
    # 3. Color augmentations
    x = keras.layers.RandomColorJitter(
        value_range=(0, 255),
        brightness_factor=config.RANDOM_BRIGHTNESS,
        contrast_factor=config.RANDOM_CONTRAST,
        saturation_factor=config.RANDOM_SATURATION,
        hue_factor=config.RANDOM_HUE,
    )(x)
    x = keras.layers.RandomSharpness(
        factor=config.RANDOM_SHARPNESS
    )(x)
    
    # 4. Final preprocessing
    x = keras.layers.Resizing(config.TARGET_SIZE[0], config.TARGET_SIZE[1])(x)
    x = keras.layers.Rescaling(1.0 / 255)(x)
    
    return x

def preview_images(dataset, num_samples=3):
    """Preview images with keyboard navigation"""
    # Build augmentation model
    inputs = keras.Input(shape=(None, None, 3))
    outputs = apply_augmentation(inputs)
    augmentation_model = keras.Model(inputs, outputs)

    # Get sample images
    for images, _ in dataset.take(1):
        sample_images = images[:num_samples]
        augmented_images = augmentation_model.predict(sample_images)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.canvas.manager.set_window_title('Augmentation Preview - Press → for next, ← for previous, ESC to quit')
    current_idx = 0

    def update_display(idx):
        """Update displayed images"""
        ax1.clear()
        ax2.clear()
        
        # Original image (convert back from float32 if needed)
        orig_img = sample_images[idx].numpy().astype('uint8')
        ax1.imshow(orig_img)
        ax1.set_title(f"Original Image {idx+1}/{len(sample_images)}")
        ax1.axis('off')
        
        # Augmented image (already scaled to 0-1 by Rescaling)
        ax2.imshow(augmented_images[idx])
        ax2.set_title("Augmented Image")
        ax2.axis('off')
        
        plt.tight_layout()
        plt.draw()

    def on_key(event):
        """Handle key presses"""
        nonlocal current_idx
        if event.key == 'right':
            current_idx = (current_idx + 1) % len(sample_images)
        elif event.key == 'left':
            current_idx = (current_idx - 1) % len(sample_images)
        elif event.key in ['escape', 'q']:
            plt.close()
            return
        update_display(current_idx)

    # Connect keyboard events
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    # Initial display
    update_display(0)
    plt.show()

def test_augmentation():
    """Test augmentation pipeline on sample images"""
    # Create dataset
    train_ds, _ = create_dataset()
    
    # Show original vs augmented images
    preview_images(train_ds, num_samples=99)

if __name__ == "__main__":
    test_augmentation()