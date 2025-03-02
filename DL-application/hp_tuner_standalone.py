# Standard library imports
import os
import subprocess
import time
from datetime import datetime
from typing import Optional

# Third-party imports
import keras
import keras_tuner
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Configuration for the DL application
CONFIG = {
    # Directories
    "DATASET_ROOT_DIR": "photoDataset",
    "DATASET_EXTERNAL_DIR": "photoDataset_external",
    "OUTPUT_DIR": "output",
    # Image parameters
    "IMAGE_DIMS": (240, 320),
    "TARGET_SIZE": (96, 96),
    # Training parameters
    "BATCH_SIZE": 16,
    "EPOCHS": 10,
    "VALIDATION_SPLIT": 0.2,
    "RANDOM_STATE": 42,
    "MAX_TRIALS": 5,
    "MAX_EPOCH_SECONDS": 10,
    # Augmentation settings
    "AUGMENTATION_ENABLED": True,
    "RANDOM_BRIGHTNESS": 0.1,
    "RANDOM_CONTRAST": 0.6,
    "RANDOM_SATURATION": (0.4, 0.6),
    "RANDOM_HUE": 0.1,
    "RANDOM_SHARPNESS": 0.1,
    "RANDOM_SHEAR_X": 0.1,
    "RANDOM_SHEAR_Y": 0.1,
    "RANDOM_TRANSLATION": 0.1,
    "RANDOM_ZOOM": 0.1,
    "RANDOM_ROTATION": 0.1,
        # TensorBoard settings
    "TENSORBOARD_ENABLED": True,
    # Metrics
    "METRICS": ["accuracy", "precision", "recall"],
    "VERBOSE": 1,
}

# Create timestamp for output directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#output_base = os.path.join(CONFIG["OUTPUT_DIR"], f"hp_tuner_{timestamp}")
output_base = os.path.join(CONFIG["OUTPUT_DIR"], f"hp_tuner")

# Create all necessary directories
OUTPUT_DIRS = {
    "hp_tuner": {
        "model": os.path.join(output_base, "model"),
        "history": os.path.join(output_base, "history"),
        "logs": os.path.join(output_base, "logs"),
        "plots": os.path.join(output_base, "plots"),
    },
}

# Create directories
for strategy in OUTPUT_DIRS.values():
    for directory in strategy.values():
        os.makedirs(directory, exist_ok=True)


def start_tensorboard():
    """Start TensorBoard in the background"""
    if CONFIG["TENSORBOARD_ENABLED"]:
        cmd = f"tensorboard --logdir={OUTPUT_DIRS['hp_tuner']['logs']} --port=6006"
        subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("TensorBoard started. Visit http://localhost:6006")


class TimeoutCallback(keras.callbacks.Callback):
    def __init__(self, max_epoch_seconds):
        super().__init__()
        self.max_epoch_seconds = max_epoch_seconds
        self.current_epoch = 0

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch
        self.epoch_start_time = time.time()

    def on_batch_end(self, batch, logs=None):
        # Skip monitoring first epoch (epoch 0)
        if self.current_epoch == 0:
            return

        elapsed_seconds = time.time() - self.epoch_start_time
        if elapsed_seconds > self.max_epoch_seconds:
            print(
                f"\nStopping training: Epoch {self.current_epoch} took {elapsed_seconds:.2f} seconds (limit: {self.max_epoch_seconds} seconds)"
            )
            self.model.stop_training = True


class CustomCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get("val_accuracy") > 0.92 and logs.get("val_loss") < 0.3:
            print("\nReached target metrics - stopping training")
            self.model.stop_training = True


def create_callbacks():
    """Create and return training callbacks."""
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            os.path.join(OUTPUT_DIRS["hp_tuner"]["model"], "model_hp_tuner.keras"),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, min_delta=0.02, verbose=CONFIG["VERBOSE"]
        ),
        keras.callbacks.TensorBoard(
            log_dir=OUTPUT_DIRS["hp_tuner"]["logs"],
            histogram_freq=1,
            write_steps_per_second=True,
        ),
        keras.callbacks.CSVLogger(
            filename=os.path.join(OUTPUT_DIRS["hp_tuner"]["logs"], "log_hp_tuner.csv"),
            separator=",",
            append=True,
        ),
        TimeoutCallback(max_epoch_seconds=CONFIG["MAX_EPOCH_SECONDS"]),
        CustomCallback(),
    ]
    return callbacks


def apply_augmentation(dataset):
    """Apply augmentation to a dataset."""

    def build_model():
        """Build the augmentation model"""
        inputs = keras.Input(shape=(None, None, 3))
        x = inputs

        # 1. Initial preprocessing
        x = keras.layers.AutoContrast(value_range=(0, 255))(x)

        # 2. Spatial augmentations
        x = keras.layers.RandomFlip(mode="horizontal_and_vertical")(x)
        x = keras.layers.RandomRotation(factor=CONFIG["RANDOM_ROTATION"])(x)
        x = keras.layers.RandomShear(
            x_factor=(0, CONFIG["RANDOM_SHEAR_X"]),
            y_factor=(0, CONFIG["RANDOM_SHEAR_Y"]),
        )(x)
        x = keras.layers.RandomTranslation(
            height_factor=CONFIG["RANDOM_TRANSLATION"],
            width_factor=CONFIG["RANDOM_TRANSLATION"],
        )(x)
        x = keras.layers.RandomZoom(
            height_factor=CONFIG["RANDOM_ZOOM"], width_factor=CONFIG["RANDOM_ZOOM"]
        )(x)

        # 3. Color augmentations
        x = keras.layers.RandomColorJitter(
            value_range=(0, 255),
            brightness_factor=CONFIG["RANDOM_BRIGHTNESS"],
            contrast_factor=CONFIG["RANDOM_CONTRAST"],
            saturation_factor=CONFIG["RANDOM_SATURATION"],
            hue_factor=CONFIG["RANDOM_HUE"],
        )(x)
        x = keras.layers.RandomSharpness(factor=CONFIG["RANDOM_SHARPNESS"])(x)

        # 4. Normalization
        x = keras.layers.Rescaling(scale=1.0 / 255)(x)

        return keras.Model(inputs, x)

    # Build model and apply to dataset
    model = build_model()
    return dataset.map(
        lambda x, y: (model(x, training=True), y), num_parallel_calls=AUTOTUNE
    )


def create_dataset_train():
    dataset = keras.utils.image_dataset_from_directory(
        CONFIG["DATASET_ROOT_DIR"],
        labels="inferred",
        label_mode="categorical",
        color_mode="rgb",
        batch_size=CONFIG["BATCH_SIZE"],
        image_size=CONFIG["IMAGE_DIMS"],
        shuffle=True,
        seed=CONFIG["RANDOM_STATE"],
        validation_split=CONFIG["VALIDATION_SPLIT"],
        subset="training",
        follow_links=False,
    )

    return dataset


def create_dataset_val():
    dataset = keras.utils.image_dataset_from_directory(
        CONFIG["DATASET_ROOT_DIR"],
        labels="inferred",
        label_mode="categorical",
        color_mode="rgb",
        batch_size=CONFIG["BATCH_SIZE"],
        image_size=CONFIG["IMAGE_DIMS"],
        shuffle=True,
        seed=CONFIG["RANDOM_STATE"],
        validation_split=CONFIG["VALIDATION_SPLIT"],
        subset="validation",
        follow_links=False,
    )

    return dataset


def create_dataset_test():
    dataset = keras.utils.image_dataset_from_directory(
        directory=CONFIG["DATASET_EXTERNAL_DIR"],
        labels="inferred",
        label_mode="categorical",
        color_mode="rgb",
        batch_size=CONFIG["BATCH_SIZE"],
        image_size=CONFIG["IMAGE_DIMS"],
        shuffle=True,
        seed=CONFIG["RANDOM_STATE"],
        validation_split=CONFIG["VALIDATION_SPLIT"],
        subset="validation",
        follow_links=False,
    )

    return dataset


def build_model(hp):
    """Build a CNN model with tunable hyperparameters."""
    # Input layer with flexible dimensions for variable image sizes
    inputs = keras.Input(shape=(None, None, 3))

    # Skip preprocessing as it's handled in the dataset pipeline
    x = keras.layers.Resizing(CONFIG["TARGET_SIZE"][0], CONFIG["TARGET_SIZE"][1])(
        inputs
    )

    # --------------------
    # Normalization Layer
    # --------------------
    normalization_strategy = hp.Choice(
        "Normalization Type", ["standard", "layer", "batch", "none"], default="batch"
    )
    if normalization_strategy == "standard":
        x = keras.layers.Normalization()(x)
    elif normalization_strategy == "layer":
        x = keras.layers.LayerNormalization()(x)
    elif normalization_strategy == "batch":
        x = keras.layers.BatchNormalization()(x)

    # --------------------
    # Convolutional Layers
    # --------------------
    architecture_depth = hp.Int(
        "Number of Convolutional Layers", min_value=1, max_value=2, default=1
    )

    for i in range(architecture_depth):
        # Conv layer hyperparameters with GPU-optimized ranges
        feature_extractors = hp.Int(
            f"ConvBlock {i + 1}: Filters",
            min_value=32,  # Minimum for feature detection
            max_value=256,  # Maximum before diminishing returns
            step=32,  # Power of 2 for GPU optimization
            default=32,
        )
        activation_function = hp.Choice(
            f"ConvBlock {i + 1}: Activation Function",
            ["relu", "leaky_relu", "selu", "tanh", "gelu"],
            default="relu",
        )
        regularization_factor = hp.Float(
            f"ConvBlock {i + 1}: Dropout Rate",
            min_value=0.0,
            max_value=0.5,
            step=0.1,
            default=0.0,
        )
        perception_field = hp.Choice(
            f"ConvBlock {i + 1}: Kernel Size", [3, 5, 7], default=3
        )

        # Regularization hyperparameters
        regularization_method = hp.Choice(
            f"ConvBlock {i + 1}: Regularization Strategy",
            ["none", "l1", "l2", "l1_l2"],
            default="none",
        )
        regularization_strength = hp.Float(
            f"ConvBlock {i + 1}: Regularization Strength",
            min_value=1e-5,
            max_value=1e-2,
            sampling="log",
            default=1e-4,
        )

        # Create regularizer based on selected strategy
        if regularization_method == "l1":
            conv_regularizer = keras.regularizers.L1(regularization_strength)
        elif regularization_method == "l2":
            conv_regularizer = keras.regularizers.L2(regularization_strength)
        elif regularization_method == "l1_l2":
            conv_regularizer = keras.regularizers.L1L2(
                l1=regularization_strength, l2=regularization_strength
            )
        else:
            conv_regularizer = None

        # Convolutional layer construction
        if hp.Boolean(f"ConvBlock {i + 1}: Enable Batch Normalization"):
            x = keras.layers.BatchNormalization()(x)

        x = keras.layers.Conv2D(
            feature_extractors,
            perception_field,
            padding="same",
            kernel_regularizer=conv_regularizer,
        )(x)
        x = keras.layers.Activation(activation_function)(x)

        # Spatial Reduction layer
        dimensionality_reduction = hp.Choice(
            f"ConvBlock {i + 1}: Spatial Reduction",
            ["maximum", "average"],
            default="maximum",
        )
        if dimensionality_reduction == "maximum":
            x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        else:  # average
            x = keras.layers.AveragePooling2D(pool_size=(2, 2))(x)

        x = keras.layers.Dropout(regularization_factor)(x)

    # --------------------
    # Dense Layers
    # --------------------
    x = keras.layers.Flatten()(x)

    classifier_depth = hp.Int(
        "Number of Dense Layers", min_value=1, max_value=1, default=1
    )

    for i in range(classifier_depth):
        # Dense layer hyperparameters
        neuron_count = hp.Int(
            f"DenseBlock {i + 1}: Units",
            min_value=16,
            max_value=128,
            step=32,
            default=32,
        )
        regularization_factor = hp.Float(
            f"DenseBlock {i + 1}: Dropout Rate",
            min_value=0.0,
            max_value=0.5,
            step=0.1,
            default=0.0,
        )
        activation_function = hp.Choice(
            f"DenseBlock {i + 1}: Activation Function",
            ["relu", "selu", "leaky_relu"],
            default="relu",
        )

        # Regularization hyperparameters
        regularization_method = hp.Choice(
            f"DenseBlock {i + 1}: Regularization Strategy",
            ["none", "l1", "l2", "l1_l2"],
            default="none",
        )
        regularization_strength = hp.Float(
            f"DenseBlock {i + 1}: Regularization Strength",
            min_value=1e-5,
            max_value=1e-2,
            sampling="log",
            default=1e-4,
        )

        # Create regularizer based on selected strategy
        if regularization_method == "l1":
            dense_regularizer = keras.regularizers.L1(regularization_strength)
        elif regularization_method == "l2":
            dense_regularizer = keras.regularizers.L2(regularization_strength)
        elif regularization_method == "l1_l2":
            dense_regularizer = keras.regularizers.L1L2(
                l1=regularization_strength, l2=regularization_strength
            )
        else:
            dense_regularizer = None

        # Dense layer construction
        x = keras.layers.Dense(neuron_count, kernel_regularizer=dense_regularizer)(x)
        x = keras.layers.Activation(activation_function)(x)

        if hp.Boolean(f"DenseBlock {i + 1}: Enable Batch Normalization"):
            x = keras.layers.BatchNormalization()(x)

        x = keras.layers.Dropout(regularization_factor)(x)

    # --------------------
    # Learning Rate Configuration
    # --------------------
    learning_strategy = hp.Choice(
        "Learning Rate Schedule Type",
        ["constant", "step", "exponential", "polynomial", "cosine"],
        default="constant",
    )

    # Base learning rate for all strategies
    initial_learning_rate = hp.Float(
        "Initial Learning Rate",
        min_value=1e-4,
        max_value=1e-1,
        sampling="log",
        default=1e-3,
    )

    # Configure learning rate scheduler based on strategy
    if learning_strategy == "constant":
        learning_rate = initial_learning_rate
    elif learning_strategy == "step":
        decay_factor = hp.Float("Decay Rate", min_value=0.1, max_value=0.5, default=0.1)
        learning_rate = keras.optimizers.schedules.InverseTimeDecay(
            initial_learning_rate=initial_learning_rate,
            decay_rate=decay_factor,
            decay_steps=1,
            staircase=True,
        )
    elif learning_strategy == "exponential":
        decay_rate = hp.Float("Decay Rate", min_value=0.8, max_value=0.99, default=0.9)
        learning_rate = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_learning_rate,
            decay_rate=decay_rate,
            decay_steps=1,
        )
    elif learning_strategy == "polynomial":
        final_learning_rate = hp.Float(
            "Final Learning Rate",
            min_value=1e-7,
            max_value=1e-4,
            sampling="log",
            default=1e-6,
        )
        decay_power = hp.Float("Decay Power", min_value=1.0, max_value=3.0, default=1.0)
        learning_rate = keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=CONFIG["EPOCHS"],
            end_learning_rate=final_learning_rate,
            power=decay_power,
        )
    else:  # cosine
        min_learning_rate = hp.Float(
            "Minimum Learning Rate", min_value=0.0, max_value=0.2, default=0.0
        )
        learning_rate = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=CONFIG["EPOCHS"],
            alpha=min_learning_rate,
        )

    # --------------------
    # Model Compilation
    # --------------------
    # Output layer with softmax activation for classification (keep in float32 for stability)
    outputs = keras.layers.Dense(3, activation="softmax", dtype='float32')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)

    
    optimization_strategy = hp.Choice(
        "Optimizer Type", ["Adam", "RMSprop", "SGD", "AdamW"], default="Adam"
    )

    if optimization_strategy == "Adam":
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimization_strategy == "RMSprop":
        optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
    elif optimization_strategy == "AdamW":
        optimizer = keras.optimizers.AdamW(learning_rate=learning_rate)
    else:  # SGD
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            "precision",
            "recall",
        ],
    )

    return model


def main():
    # Set random seed for reproducibility
    keras.utils.set_random_seed(CONFIG["RANDOM_STATE"])

    # Clear session and memory
    keras.utils.clear_session(free_memory=True)

    # Load dataset
    train_ds = create_dataset_train()
    val_ds = create_dataset_val()
    test_ds = create_dataset_test()

    # Start TensorBoard for monitoring if enabled
    start_tensorboard()

    print("Starting hyperparameter tuning...")
    # Configure Bayesian Optimization tuner
    tuner = keras_tuner.Hyperband(
        build_model,
        objective="val_loss",
        directory=OUTPUT_DIRS["hp_tuner"]["model"],
        max_epochs=CONFIG["EPOCHS"],
        hyperband_iterations=1,
    )

    # Perform hyperparameter search
    tuner.search(
        train_ds,
        validation_data=val_ds,
        epochs=CONFIG["EPOCHS"],
        callbacks=create_callbacks(),
    )

    # Save best hyperparameters
    best_hp = tuner.get_best_hyperparameters(1)[0]
    with open(
        os.path.join(
            OUTPUT_DIRS["hp_tuner"]["model"], "hyperparameters_hp_tuner.txt"
        ),
        "w",
    ) as f:
        for param, value in best_hp.values.items():
            f.write(f"{param}: {value}\n")

    # Train final model with best hyperparameters
    best_model = tuner.hypermodel.build(best_hp)
    best_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=CONFIG["EPOCHS"],
    )
    model_path = os.path.join(OUTPUT_DIRS["hp_tuner"]["model"], "model_best_hp_tuner.keras")
    best_model.save(model_path)

    # Evaluate model on test dataset and create a confusion matrix
    test_results = best_model.evaluate(val_ds)
    print("Test results:")
    print(test_results)

    # Get predictions for test dataset
    y_pred = best_model.predict(val_ds)
    y_pred_classes = tf.argmax(y_pred, axis=1)

    # Get true labels (need to unbatch the test dataset)
    y_true = tf.concat([y for _, y in test_ds], axis=0)
    y_true_classes = tf.argmax(y_true, axis=1)

    # Calculate confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)

    # Create confusion matrix plot
    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=['paper', 'rock', 'scissors']
    )
    disp.plot(cmap='Blues', values_format='d')
    plt.title('Confusion Matrix - Test Dataset')

    # Save the plot
    plt.savefig(os.path.join(OUTPUT_DIRS["hp_tuner"]["plots"], 'confusion_matrix.png'))
    plt.close()

if __name__ == "__main__":
    main()
