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
    """Apply image augmentation pipeline to input layer
    
    The augmentation pipeline follows a specific order to ensure optimal results:
    1. Initial preprocessing - Enhance image contrast
    2. Spatial transformations - Applied before resizing to preserve quality
    3. Color adjustments - Modify image appearance
    4. Final preprocessing - Resize and normalize for model input
    
    Args:
        inputs: A tensor of shape (batch_size, height, width, channels)
        
    Returns:
        A tensor of the same shape as inputs, with random augmentations applied
    """
    x = inputs
    
    # 1. Initial preprocessing
    # AutoContrast normalizes the contrast across the image
    # value_range=(0, 255) specifies the input image range
    x = keras.layers.AutoContrast(value_range=(0, 255))(x)
    
    # 2. Spatial augmentations
    # Applied before resizing to maintain image quality
    # Each operation is random and only applied during training (training=True)
    
    # Randomly flip image horizontally and/or vertically
    x = keras.layers.RandomFlip(mode="horizontal_and_vertical")(x)
    
    # Randomly rotate image by up to ±RANDOM_ROTATION * 360 degrees
    x = keras.layers.RandomRotation(factor=config.RANDOM_ROTATION)(x)
    
    # Apply random shearing transformation
    # Shear factor of 0.2 means up to 20% shear in each direction
    x = keras.layers.RandomShear(
        x_factor=config.RANDOM_SHEAR_X,
        y_factor=config.RANDOM_SHEAR_Y
    )(x)
    
    # Randomly translate image in height and width
    # Factor of 0.2 means up to 20% translation in any direction
    x = keras.layers.RandomTranslation(
        height_factor=config.RANDOM_TRANSLATION,
        width_factor=config.RANDOM_TRANSLATION
    )(x)
    
    # Randomly zoom in/out
    # Factor of 0.2 means zoom range of 80% to 120%
    x = keras.layers.RandomZoom(
        height_factor=config.RANDOM_ZOOM,
        width_factor=config.RANDOM_ZOOM
    )(x)
    
    # 3. Color augmentations
    # Randomly adjust color properties while maintaining natural look
    x = keras.layers.RandomColorJitter(
        value_range=(0, 255),  # Input image range
        brightness_factor=config.RANDOM_BRIGHTNESS,  # ±20% brightness
        contrast_factor=config.RANDOM_CONTRAST,      # ±20% contrast
        saturation_factor=config.RANDOM_SATURATION,  # ±20% saturation
        hue_factor=config.RANDOM_HUE,               # ±20% hue
    )(x)
    
    # Randomly adjust image sharpness
    # Factor of 0.2 means sharpness range of 80% to 120%
    x = keras.layers.RandomSharpness(
        factor=config.RANDOM_SHARPNESS
    )(x)
    
    # 4. Final preprocessing
    # Resize to target dimensions for model input
    x = keras.layers.Resizing(config.TARGET_SIZE[0], config.TARGET_SIZE[1])(x)
    # Normalize pixel values to 0-1 range
    x = keras.layers.Rescaling(1.0 / 255)(x)
    
    return x

def preview_images(dataset, num_samples=3):
    """Preview original and augmented images with interactive keyboard navigation
    
    This function creates an interactive matplotlib window showing original images
    and their augmented versions side by side. Use keyboard controls to navigate:
    - Right arrow (→): Next image
    - Left arrow (←): Previous image
    - ESC or Q: Quit preview
    
    Args:
        dataset: tf.data.Dataset containing images
        num_samples: Number of images to sample from dataset (default: 3)
    """
    # Build augmentation model using Keras Functional API
    # None values in shape allow for variable image dimensions
    inputs = keras.Input(shape=(None, None, 3))
    outputs = apply_augmentation(inputs)
    augmentation_model = keras.Model(inputs, outputs)

    # Extract sample images from dataset
    # Using take(1) to get first batch, then slice to get num_samples
    for images, _ in dataset.take(1):
        sample_images = images[:num_samples]
        # Apply augmentation in training mode to enable random transformations
        augmented_images = augmentation_model(sample_images, training=True)

    # Create matplotlib figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.canvas.manager.set_window_title('Augmentation Preview - Press → for next, ← for previous, ESC to quit')
    current_idx = 0  # Track current image index

    def update_display(idx):
        """Update the matplotlib display with new images
        
        This function updates both subplots with the original and augmented
        versions of the image at the given index. The original image is
        converted back to uint8 format (0-255 range) while the augmented
        image remains in float format (0-1 range) as output by the model.
        
        Args:
            idx: Index of the image pair to display
        """
        ax1.clear()
        ax2.clear()
        
        # Original image (convert back from float32 if needed)
        # numpy() converts tensor to numpy array
        # astype('uint8') ensures proper 0-255 range for display
        orig_img = sample_images[idx].numpy().astype('uint8')
        ax1.imshow(orig_img)
        ax1.set_title(f"Original Image {idx+1}/{len(sample_images)}")
        ax1.axis('off')
        
        # Augmented image (already scaled to 0-1 by Rescaling)
        # imshow automatically handles the 0-1 float range
        ax2.imshow(augmented_images[idx])
        ax2.set_title("Augmented Image")
        ax2.axis('off')
        
        plt.tight_layout()
        plt.draw()

    def on_key(event):
        """Handle keyboard navigation events
        
        Processes keyboard events for image navigation:
        - Right arrow: Move to next image
        - Left arrow: Move to previous image
        - ESC/Q: Close the preview window
        
        The current_idx cycles through the available images using modulo
        operation to wrap around at the end of the list.
        
        Args:
            event: Matplotlib keyboard event object
        """
        nonlocal current_idx
        if event.key == 'right':
            current_idx = (current_idx + 1) % len(sample_images)
        elif event.key == 'left':
            current_idx = (current_idx - 1) % len(sample_images)
        elif event.key in ['escape', 'q']:
            plt.close()
            return
        update_display(current_idx)

    # Connect keyboard handler to the figure
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    # Show initial image pair
    update_display(0)
    plt.show()

def test_augmentation():
    """Test the augmentation pipeline on sample images
    
    This function demonstrates the augmentation pipeline by:
    1. Creating a dataset from the image directory
    2. Showing a preview window with original and augmented image pairs
    
    The preview loads 99 sample images and allows interactive navigation
    through them to verify augmentation effects. Each image shown will
    have random augmentations applied in real-time.
    """
    # Create training dataset (validation set is unused here)
    train_ds, _ = create_dataset()
    
    # Launch interactive preview with 99 sample images
    # High sample count helps verify augmentation variety
    preview_images(train_ds, num_samples=99)

if __name__ == "__main__":
    """
    Main entry point - runs the augmentation preview tool
    
    When this script is run directly, it will:
    1. Load the image dataset
    2. Open an interactive preview window
    3. Allow navigation through augmented image pairs
    """
    test_augmentation()
