import numpy as np
import tensorflow as tf
import keras
import autokeras as ak

# Import custom modules
from src.training_callbacks import ringring_callbackplease
from src.config import config
from src.tensorboard import TensorboardLauncher

def main():
    """Main function for training an AutoKeras image classification model.
    
    This script:
    1. Sets up training environment (mixed precision, random seed)
    2. Creates datasets from image directories
    3. Builds an AutoKeras model architecture
    4. Performs automated architecture search
    5. Evaluates and exports the best model
    """
    
    # Enable mixed precision training for better performance on compatible GPUs
    # This allows some operations to use float16 instead of float32
    keras.mixed_precision.set_global_policy("mixed_float16")

    # Set global random seed for reproducibility
    # This ensures consistent results across runs
    keras.utils.set_random_seed(config.RANDOM_STATE)

    # Clear any existing Keras session to free up memory
    # This helps prevent memory leaks between training runs
    keras.utils.clear_session(free_memory=True)

    # Initialize and start TensorBoard for visualization
    # This allows monitoring of training metrics in real-time
    tensorboard = TensorboardLauncher(config.LOGS_DIR)
    tensorboard.start_tensorboard()
    
    # Create training and validation datasets
    # If EXTERNAL_DATASET_USAGE is True, use separate directories
    # If False, split single directory using validation_split
    if config.EXTERNAL_DATASET_USAGE:
        # Create training dataset from main directory
        train_ds = ak.image_dataset_from_directory(
            directory=config.DATASET_ROOT_DIR,
            batch_size=config.BATCH_SIZE,
            image_size=(config.TARGET_SIZE[0], config.TARGET_SIZE[1]),
            shuffle=True,
            seed=config.RANDOM_STATE,
        )

        # Create validation dataset from external directory
        val_ds = ak.image_dataset_from_directory(
            directory=config.DATASET_EXTERNAL_DIR,
            batch_size=config.BATCH_SIZE,
            image_size=(config.TARGET_SIZE[0], config.TARGET_SIZE[1]),
            shuffle=True,
            seed=config.RANDOM_STATE,
        )
    else:
        # Create training dataset with validation split
        train_ds = ak.image_dataset_from_directory(
            directory=config.DATASET_ROOT_DIR,
            validation_split=config.VALIDATION_SPLIT,
            subset='training',
            batch_size=config.BATCH_SIZE,
            image_size=(config.TARGET_SIZE[0], config.TARGET_SIZE[1]),
            shuffle=True,
            seed=config.RANDOM_STATE,
        )

        # Create validation dataset from the same split
        val_ds = ak.image_dataset_from_directory(
            directory=config.DATASET_ROOT_DIR,
            validation_split=config.VALIDATION_SPLIT,
            subset='validation',
            batch_size=config.BATCH_SIZE,
            image_size=(config.TARGET_SIZE[0], config.TARGET_SIZE[1]),
            shuffle=True,
            seed=config.RANDOM_STATE,
        )

    # Define model architecture using AutoKeras building blocks

    # Input layer: Handles image input with automatic shape inference
    input = ak.ImageInput()

    # Normalization layer: Scales pixel values to improve training
    # Automatically computes mean and standard deviation from data
    normal = ak.Normalization()(input)

    # Data augmentation: Increases effective dataset size
    # - horizontal_flip: Randomly flip images horizontally
    # - vertical_flip: Randomly flip images vertically
    augment = ak.ImageAugmentation(
        horizontal_flip=True,
        vertical_flip=True,
    )(normal)

    # Convolutional block: Main feature extraction
    # AutoKeras will search for optimal CNN architecture
    # May include multiple conv layers, pooling, dropout, etc.
    convblock = ak.ConvBlock()(augment)

    # Classification head: Final layers for classification
    # Automatically adjusts to number of classes in dataset
    head = ak.ClassificationHead()(convblock)

    # Create AutoModel for architecture search
    # - Tries different architectures within the defined structure
    # - Keeps track of best performing models
    # - Optimizes hyperparameters automatically
    auto_model = ak.AutoModel(
        inputs=input,
        outputs=head,
        project_name=config.OUTPUT_DIR,
        max_trials=config.MAX_TRIALS,
        seed=config.RANDOM_STATE,
        directory=config.OUTPUT_DIR,
    )

    # Train model with architecture search
    # - Tries different model configurations
    # - Uses early stopping and other callbacks
    # - Keeps best performing model
    auto_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.EPOCHS,
        callbacks=ringring_callbackplease(),
    )
    
    # Test model on validation set
    # Returns predictions for all validation samples
    predict = auto_model.predict(val_ds)
    print(predict)
    
    # Evaluate model performance
    # Returns metrics like accuracy, loss
    eval = auto_model.evaluate(val_ds)
    print(eval)

    # Export the best model found during search
    # Can be loaded later using standard Keras API
    export = auto_model.export_model()
    print(export)

if __name__ == "__main__":
    main()
