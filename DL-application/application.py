"""
Rock Paper Scissors Image Classifier using Deep Learning

This application implements a Convolutional Neural Network (CNN) to classify images
into three categories: rock, paper, or scissors. The model uses multiple
convolutional layers for feature extraction followed by dense layers for classification.

Key Components:
- CNN Architecture: Extracts visual features from images
- Data Augmentation: Improves model generalization by creating variations of training images
- Regularization: Prevents overfitting using techniques like dropout and L2 regularization
- Performance Monitoring: Various plots and metrics to analyze model behavior
"""

import keras
from src.config import config
from src.create_dataset import create_dataset
from src.create_plots import (
    plot_bias_variance,
    plot_confusion_matrix,
    plot_dataset_distribution,
    plot_metric_gap_analysis,
    plot_metrics_comparison,
    plot_training_history,
)
from src.tensorboard import TensorboardLauncher


def build_model():
    """
    Creates the CNN model architecture for image classification.

    Architecture Overview:
    1. Input Layer: Accepts RGB images of size specified in config
    2. Multiple Convolutional Blocks: Each containing:
       - Conv2D: Extracts visual features using sliding filters
       - BatchNormalization: Stabilizes learning
       - MaxPooling: Reduces spatial dimensions
    3. Dense Layers: Final classification based on extracted features

    Returns:
        keras.Model: Compiled model ready for training
    """
    # Input layer: Shape is (height, width, 3 color channels)
    inputs = keras.layers.Input(shape=(config.TARGET_SIZE[0], config.TARGET_SIZE[1], 3))
    # First Convolutional Block
    # - 224 filters: Number of different features to detect
    # - (7,7) kernel: Size of the sliding window
    # - leaky_relu: Prevents "dying ReLU" problem
    # - L2 regularization: Prevents overfitting by penalizing large weights
    x = keras.layers.Conv2D(
        224,
        (7, 7),
        activation="leaky_relu",
        padding="valid",
        kernel_regularizer=keras.regularizers.l2(0.0001633),
    )(inputs)

    # Normalization and Pooling
    # - BatchNormalization: Normalizes the layer's inputs, stabilizing training
    # - MaxPooling: Reduces image size while keeping important features
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0)(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Second Convolutional Block
    # Similar structure to first block, building more complex features
    x = keras.layers.Conv2D(
        224,
        (7, 7),
        activation="leaky_relu",
        padding="valid",
        kernel_regularizer=keras.regularizers.l2(0.0001633),
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Third Convolutional Block
    # Deepest convolutional layer, detecting most complex features
    x = keras.layers.Conv2D(
        224,
        (7, 7),
        activation="leaky_relu",
        padding="valid",
        kernel_regularizer=keras.regularizers.l2(0.0001633),
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Flatten: Converts 2D feature maps to 1D vector for dense layers
    x = keras.layers.Flatten()(x)

    # Dense Layer: Combines features for classification
    # - 512 neurons: Rich representation for final classification
    # - Dropout: Randomly turns off 10% of neurons to prevent overfitting
    x = keras.layers.Dense(512, activation="leaky_relu")(x)
    x = keras.layers.Dropout(0.1)(x)

    # Output Layer
    # - 3 neurons: One for each class (rock, paper, scissors)
    # - Softmax: Converts outputs to probabilities that sum to 1
    outputs = keras.layers.Dense(3, activation="softmax")(x)

    # Create model
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model


def main():
    """
    Main training pipeline for the rock-paper-scissors classifier.

    Steps:
    1. Load and prepare training/validation datasets
    2. Configure model training settings
    3. Train the model with early stopping
    4. Generate performance visualization plots
    """
    # Load and prepare datasets with augmentation
    train_ds, val_ds = create_dataset()
    tensorboard = TensorboardLauncher(config.LOGS_DIR)
    tensorboard.start_tensorboard()
    model = build_model()
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=0.00040288),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Display model architecture summary
    model.summary()

    # Early Stopping Callback
    # Stops training if the model achieves high accuracy and low loss
    # This prevents overfitting and saves training time
    class CustomCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if logs.get("val_accuracy") > 0.92 and logs.get("val_loss") < 0.3:
                print("\nReached target metrics - stopping training")
                self.model.stop_training = True

    custom_callback = CustomCallback()
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=config.MODEL_MANUAL_PATH,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
    )

    csv_logger = keras.callbacks.CSVLogger(config.CSV_LOG_PATH, append=True)

    model.fit(
        train_ds,
        epochs=config.EPOCHS,
        validation_data=val_ds,
        callbacks=[model_checkpoint_callback, csv_logger, custom_callback],
    )

    # Generate comprehensive performance analysis plots
    # These help understand model behavior and identify potential issues
    # Analyze dataset class distribution for balance
    plot_dataset_distribution()

    # Visualize training metrics over time
    plot_training_history(config.CSV_LOG_PATH)

    # Show model's classification accuracy per class
    plot_confusion_matrix(model, val_ds)

    # Compare precision and recall metrics
    plot_metrics_comparison(model, val_ds)

    # Analyze model's bias-variance tradeoff
    plot_bias_variance(config.CSV_LOG_PATH)

    # Check for signs of overfitting
    plot_metric_gap_analysis(config.CSV_LOG_PATH)


if __name__ == "__main__":
    main()
