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
from src.tensorboard import TensorboardLauncher
from src.training_callbacks import ringring_callbackplease

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
    3. Train the model with early stopping (if SKIP_TRAINING is False)
    4. Generate performance visualization plots
    """
    # Load and prepare datasets with augmentation
    train_ds, val_ds = create_dataset()
    tensorboard = TensorboardLauncher()
    tensorboard.start_tensorboard()

    # If model exists and SKIP_TRAINING is True, load it
    if config.SKIP_TRAINING_MANUAL_CNN:
        print("Loading existing model...")
        model = keras.models.load_model(config.PATH_MANUAL_CNN_MODEL)
    else:
        print("Building and training new model...")
        model = build_model()
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=0.00040288),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Precision(),
            keras.metrics.Recall(),
        ],
    )

    # Display model architecture summary
    model.summary()

    # Early Stopping Callback
    # Stops training if the model achieves high accuracy and low loss
    # This prevents overfitting and saves training time
    if not config.SKIP_TRAINING_MANUAL_CNN:
        callbacks = ringring_callbackplease(
            logs_dir=tensorboard.log_dir,
            csv_log_path=config.PATH_MANUAL_CNN_LOG,
            use_model_checkpoint=True,
            use_early_stopping=False,
            use_csv_logger=True,
            use_timeout=False,
            use_custom_callback=False,
            use_tensorboard=True
        )
        
        model.fit(
            train_ds,
            epochs=config.EPOCHS,
            validation_data=val_ds,
            callbacks=callbacks,
        )

if __name__ == "__main__":
    main()
