import keras
import tensorflow as tf
from src.config import config

def create_dataset():
    """
    Use Keras API to create a dataset from the images
    https://keras.io/api/data_loading/image/
    """
        
    train_ds, val_ds = keras.utils.image_dataset_from_directory(
        directory=config.DATASET_ROOT_DIR,
        labels="inferred",      # Use directory names as labels
        label_mode="categorical",
        color_mode="rgb",
        batch_size=config.BATCH_SIZE,
        image_size=config.IMAGE_DIMS,
        shuffle=True,
        seed=config.RANDOM_STATE,
        validation_split=config.VALIDATION_SPLIT,
        subset="both",
        interpolation="bilinear",
        verbose=config.VERBOSE
    )
    
    # Apply autotuning optimizations
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds

if __name__ == "__main__":
    """
    Test functions in this file
    """
    
    # Create dataset
    ds_train, ds_val = create_dataset()
    
    # Print dataset information
    print(f"Training dataset: {ds_train}")
    print(f"Validation dataset: {ds_val}")