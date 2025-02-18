# Third-party imports
import tensorflow as tf
import keras
import autokeras as ak

# Import custom modules
from src.training_callbacks import ringring_callbackplease
from src.config import config
from src.tensorboard import TensorboardLauncher
from src.create_dataset import create_dataset
from src.create_plots import (
    plot_dataset_distribution,
    plot_training_history,
    plot_confusion_matrix,
    plot_metrics_comparison,
    plot_bias_variance,
    plot_metric_gap_analysis
)


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
    train_ds, val_ds = create_dataset()

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
        project_name="AutoKeras_Image_Classification",
        max_trials=5,
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
    
    # Generate plots showing dataset and model performance
    plot_dataset_distribution()  # Dataset balance visualization
    plot_training_history(config.CSV_LOG_PATH)  # Training progress
    plot_confusion_matrix(export, val_ds)  # Classification performance
    plot_metrics_comparison(export, val_ds)  # Precision/Recall analysis
    plot_bias_variance(config.CSV_LOG_PATH)  # Bias-Variance tradeoff
    plot_metric_gap_analysis(config.CSV_LOG_PATH)  # Overfitting analysis

if __name__ == "__main__":
    main()
