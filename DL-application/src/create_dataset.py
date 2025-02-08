import os
import keras
import tensorflow as tf
from src.config import config

def create_dataset():
    """
    Use Keras API to create a dataset from the images.
    If EXTERNAL_DATASET_USAGE is True, uses external dataset for validation.
    Otherwise uses validation split from main dataset.
    """
    
    if config.EXTERNAL_DATASET_USAGE:

        # Create training dataset from main directory
        train_ds = keras.utils.image_dataset_from_directory(
            directory=config.DATASET_ROOT_DIR,
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

        # Create validation dataset from external directory
        val_ds = keras.utils.image_dataset_from_directory(
            directory=config.DATASET_EXTERNAL_DIR,
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
    else:
        # Original behavior - use validation split
        train_ds, val_ds = keras.utils.image_dataset_from_directory(
            directory=config.DATASET_ROOT_DIR,
            labels="inferred",
            label_mode="categorical",
            color_mode="rgb",
            batch_size=config.BATCH_SIZE,
            image_size=config.TARGET_SIZE,
            shuffle=True,
            seed=config.RANDOM_STATE,
            validation_split=config.VALIDATION_SPLIT,
            subset="both",
            interpolation="bilinear",
            verbose=config.VERBOSE,
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

    ds_train, ds_val = create_dataset()

    # Print dataset information
    print(f"Training dataset: {ds_train}")
    print(f"Validation dataset: {ds_val}")
