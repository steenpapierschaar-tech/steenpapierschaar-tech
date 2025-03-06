import sys
from pathlib import Path
import keras
import tensorflow as tf
import matplotlib.pyplot as plt

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

# Import local modules
from src.config import config

# Define constants at module level
AUTOTUNE = tf.data.AUTOTUNE

def apply_augmentation(dataset):
    """Apply augmentation to a dataset.
    
    Applies a series of augmentation operations in order:
    1. Initial preprocessing - Enhance image contrast
    2. Spatial transformations - Applied before resizing
    3. Color adjustments - Modify image appearance
    4. Normalization - Scale pixel values to [0, 1] range
    
    Args:
        dataset (tf.data.Dataset): Dataset to augment
        
    Returns:
        tf.data.Dataset: Augmented dataset with pixel values in [0, 1] range
    """
    def build_model():
        """Build the augmentation model"""
        inputs = keras.Input(shape=(None, None, 3))
        x = inputs
        
        # 1. Initial preprocessing
        x = keras.layers.AutoContrast(value_range=(0, 255))(x)
        
        # 2. Spatial augmentations
        x = keras.layers.RandomFlip(mode="horizontal_and_vertical")(x)
        x = keras.layers.RandomRotation(factor=config.RANDOM_ROTATION)(x)
        x = keras.layers.RandomShear(
            x_factor=(0, config.RANDOM_SHEAR_X),
            y_factor=(0, config.RANDOM_SHEAR_Y)
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
        x = keras.layers.RandomSharpness(factor=config.RANDOM_SHARPNESS)(x)
        
        # 4. Normalization
        x = keras.layers.Rescaling(scale=1./255)(x)
        
        return keras.Model(inputs, x)
    
    # Build model and apply to dataset
    model = build_model()
    return dataset.map(
        lambda x, y: (model(x, training=True), y),
        num_parallel_calls=AUTOTUNE
    )

def preview_augmentation_pipeline(num_samples: int = 3) -> None:
    """Preview the augmentation pipeline on images from the dataset
    
    Shows original and augmented versions of the same images side by side.
    Use keyboard controls to navigate:
    - Right arrow (→): Next image
    - Left arrow (←): Previous image
    - ESC or Q: Quit preview
    
    Args:
        num_samples (int): Number of images to preview (default: 3)
    """
    # Create dataset
    dataset = dir_to_dataset(Path(config.DATASET_ROOT_DIR))
    preview_ds = dataset.take(1)
    
    # Get one batch of images
    batch = next(iter(preview_ds))
    images = batch[0][:num_samples]  # Take requested number of samples
    
    # Create single-batch dataset from preview images
    preview_images_ds = tf.data.Dataset.from_tensor_slices((
        images, tf.zeros(len(images))  # Dummy labels
    )).batch(len(images))
    
    # Apply augmentation
    augmented_ds = apply_augmentation(preview_images_ds)
    augmented_images = next(iter(augmented_ds))[0]  # Get first (and only) batch
    
    # Setup visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.canvas.manager.set_window_title('Augmentation Preview - Press → for next, ← for previous, ESC to quit')
    current_idx = 0

    def update_display(idx):
        """Update the matplotlib display with new images"""
        ax1.clear()
        ax2.clear()
        
        # Display original image
        orig_img = images[idx].numpy().astype('uint8')
        ax1.imshow(orig_img)
        ax1.set_title("Original Image")
        ax1.axis('off')
        
        # Display augmented version of the same image
        ax2.imshow(augmented_images[idx])
        ax2.set_title("Augmented Version")
        ax2.axis('off')
        
        plt.tight_layout()
        plt.draw()

    def on_key(event):
        """Handle keyboard navigation events"""
        nonlocal current_idx
        if event.key == 'right':
            current_idx = (current_idx + 1) % len(images)
        elif event.key == 'left':
            current_idx = (current_idx - 1) % len(images)
        elif event.key in ['escape', 'q']:
            plt.close()
            return
        update_display(current_idx)

    fig.canvas.mpl_connect('key_press_event', on_key)
    update_display(0)
    plt.show()

def dir_to_dataset(directory: Path, for_preview=False):
    """Load images from a directory into a dataset.
    
    Args:
        directory (Path): Path to directory containing image classes in subdirectories
        
    Returns:
        tf.data.Dataset: Dataset containing images and labels
    """
    # Start with an uncached dataset
    dataset = keras.utils.image_dataset_from_directory(
        directory=str(directory),
        labels="inferred",
        label_mode="categorical",
        color_mode="rgb",
        batch_size=config.BATCH_SIZE,
        image_size=config.TARGET_SIZE,
        shuffle=True,
        seed=config.RANDOM_STATE,
        interpolation="bilinear",
        verbose=config.VERBOSE,
    )
    
    return dataset

def merge_datasets(*datasets):
    """Merge multiple datasets into a single dataset.
    
    Args:
        *datasets: Variable number of tf.data.Dataset objects
        
    Returns:
        tf.data.Dataset: Merged dataset
    """
    if not datasets:
        raise ValueError("At least one dataset must be provided")
    
    return datasets[0].concatenate(*datasets[1:]) if len(datasets) > 1 else datasets[0]

def split_dataset(dataset):
    """Split a dataset into training and validation sets."""
    dataset_size = tf.data.experimental.cardinality(dataset).numpy()
    train_size = int(dataset_size * (1 - config.VALIDATION_SPLIT))
    
    train_ds = dataset.take(train_size)
    val_ds = dataset.skip(train_size)
    
    return train_ds, val_ds

def rescale_dataset(dataset):
    """Apply rescaling to a dataset without augmentation.
    
    Args:
        dataset (tf.data.Dataset): Dataset to rescale
        
    Returns:
        tf.data.Dataset: Rescaled dataset with pixel values in [0, 1] range
    """
    return dataset.map(
        lambda x, y: (tf.cast(x, tf.float32) / 255.0, y),
        num_parallel_calls=AUTOTUNE
    )

def create_validation_dataset():
    """Create a validation dataset from the external dataset directory.
    
    Returns:
        tf.data.Dataset: Validation dataset with categorical labels
        and pixel values in [0, 1] range.
    """
    # Create validation dataset using existing infrastructure
    val_ds = dir_to_dataset(Path(config.DATASET_EXTERNAL_DIR))
    
    # Apply rescaling without augmentation
    val_ds = rescale_dataset(val_ds)
    
    # Configure dataset for performance
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    return val_ds

def create_dataset(augment_train=True, additional_dirs=None):
    """Create and return datasets for training and validation.
    
    Args:
        augment_train (bool): Whether to apply augmentation to training data (default: True)
        additional_dirs (list[Path], optional): Additional directories to merge into training set
        
    Returns:
        tuple[tf.data.Dataset, tf.data.Dataset]: Training and validation datasets
        with pixel values in [0, 1] range.
    """
    if config.EXTERNAL_DATASET_USAGE:
        train_ds = dir_to_dataset(Path(config.DATASET_EXTERNAL_DIR))
        val_ds = dir_to_dataset(Path(config.DATASET_EXTERNAL_DIR))
    else:
        dataset = dir_to_dataset(Path(config.DATASET_ROOT_DIR))
        train_ds, val_ds = split_dataset(dataset)
    
    # Merge additional datasets if provided
    if additional_dirs:
        additional_datasets = [dir_to_dataset(path) for path in additional_dirs]
        train_ds = merge_datasets(train_ds, *additional_datasets)
    
    if augment_train:
        train_ds = apply_augmentation(train_ds)  # Includes rescaling
    else:
        train_ds = rescale_dataset(train_ds)
    
    # Always rescale validation dataset
    val_ds = rescale_dataset(val_ds)
    
    return train_ds, val_ds

if __name__ == "__main__":

    ds_train, ds_val = create_dataset()

    # Print dataset information
    print(f"Training dataset: {ds_train}")
    print(f"Validation dataset: {ds_val}")

    # Show augmentation preview
    preview_augmentation_pipeline(num_samples=config.BATCH_SIZE)
