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

# Configure parallel processing
AUTOTUNE = tf.data.AUTOTUNE

# Configuration for the DL application
CONFIG = {
    # Directories
    "DATASET_ROOT_DIR": "photoDataset",
    "DATASET_EXTERNAL_DIR": "photoDataset",
    "OUTPUT_DIR": "/Volumes/dnn/output",
    # Image parameters
    "IMAGE_DIMS": (240, 320),
    "TARGET_SIZE": (120, 160),
    # Training parameters
    "BATCH_SIZE": 32,
    "EPOCHS": 90,
    "VALIDATION_SPLIT": 0.2,
    "RANDOM_STATE": 42,
    "MAX_TRIALS": 10,
    "MAX_EPOCH_SECONDS": 30,  # Increased from 10 to allow more training time
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
        if logs.get("val_accuracy") > 0.95 and logs.get("val_loss") < 0.2:  # Higher threshold
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
            monitor="val_loss", patience=7, min_delta=0.01, verbose=CONFIG["VERBOSE"]
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
        # Removed ValLossEarlyStop to allow for better training
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
    
    # Apply data augmentation if enabled
    if CONFIG["AUGMENTATION_ENABLED"]:
        dataset = apply_augmentation(dataset)
    
    # Add prefetching for better performance
    dataset = dataset.prefetch(AUTOTUNE)
    
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
    
    # Add normalization layer but no augmentation
    dataset = dataset.map(
        lambda x, y: (tf.cast(x, tf.float32) / 255.0, y),
        num_parallel_calls=AUTOTUNE
    )
    
    # Add prefetching for better performance
    dataset = dataset.prefetch(AUTOTUNE)
    
    return dataset


def create_dataset_test():
    # Test dataset should use the full external dataset, not a validation split
    try:
        dataset = keras.utils.image_dataset_from_directory(
            directory=CONFIG["DATASET_EXTERNAL_DIR"],
            labels="inferred",
            label_mode="categorical",
            color_mode="rgb",
            batch_size=CONFIG["BATCH_SIZE"],
            image_size=CONFIG["IMAGE_DIMS"],
            shuffle=False,  # Don't shuffle test data to maintain order
            seed=CONFIG["RANDOM_STATE"],
            # Not using validation_split parameters for test dataset
            follow_links=False,
        )
    except ValueError as e:
        print(f"Warning when creating test dataset: {str(e)}")
        # Fallback - if external dataset requires validation_split for some reason
        dataset = keras.utils.image_dataset_from_directory(
            directory=CONFIG["DATASET_EXTERNAL_DIR"],
            labels="inferred",
            label_mode="categorical",
            color_mode="rgb",
            batch_size=CONFIG["BATCH_SIZE"],
            image_size=CONFIG["IMAGE_DIMS"],
            shuffle=False,
            seed=CONFIG["RANDOM_STATE"],
            validation_split=CONFIG["VALIDATION_SPLIT"],
            subset="validation",
            follow_links=False,
        )
        print("Using validation subset of external dataset as test dataset")
    
    # Add normalization layer but no augmentation
    dataset = dataset.map(
        lambda x, y: (tf.cast(x, tf.float32) / 255.0, y),
        num_parallel_calls=AUTOTUNE
    )
    
    # Add prefetching for better performance
    dataset = dataset.prefetch(AUTOTUNE)
    
    return dataset


def build_model(hp):
    """Build a CNN model with tunable hyperparameters."""
    # Input layer
    inputs = keras.Input(shape=(CONFIG["IMAGE_DIMS"][0], CONFIG["IMAGE_DIMS"][1], 3))

    # Resizing layer to standardize input
    x = keras.layers.Resizing(CONFIG["TARGET_SIZE"][0], CONFIG["TARGET_SIZE"][1])(
        inputs
    )

    # --------------------
    # Normalization Layer - Skip as normalization is handled in dataset pipeline
    # --------------------
    normalization_strategy = hp.Choice(
        "Normalization Type", ["layer", "batch", "none"], default="batch"
    )
    
    if normalization_strategy == "layer":
        x = keras.layers.LayerNormalization()(x)
    elif normalization_strategy == "batch":
        x = keras.layers.BatchNormalization()(x)

    # --------------------
    # Convolutional Layers
    # --------------------
    architecture_depth = hp.Int(
        "Number of Convolutional Layers", min_value=2, max_value=4, default=3
    )

    for i in range(architecture_depth):
        # Conv layer hyperparameters
        feature_extractors = hp.Int(
            f"ConvBlock {i + 1}: Filters",
            min_value=32,
            max_value=128,
            step=32,
            default=64,
        )
        activation_function = hp.Choice(
            f"ConvBlock {i + 1}: Activation Function",
            ["relu", "leaky_relu"],
            default="relu",
        )
        regularization_factor = hp.Float(
            f"ConvBlock {i + 1}: Dropout Rate",
            min_value=0.0,
            max_value=0.5,
            step=0.1,
            default=0.2,
        )
        perception_field = hp.Choice(
            f"ConvBlock {i + 1}: Kernel Size", [3, 5], default=3
        )

        # Regularization hyperparameters
        regularization_method = hp.Choice(
            f"ConvBlock {i + 1}: Regularization Strategy",
            ["none", "l2"],
            default="l2",
        )
        regularization_strength = hp.Float(
            f"ConvBlock {i + 1}: Regularization Strength",
            min_value=1e-5,
            max_value=1e-3,
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
        if hp.Boolean(f"ConvBlock {i + 1}: Enable Batch Normalization", default=True):
            x = keras.layers.BatchNormalization()(x)

        x = keras.layers.Conv2D(
            feature_extractors,
            perception_field,
            padding="same",
            kernel_regularizer=conv_regularizer,
        )(x)
        
        if activation_function == "leaky_relu":
            x = keras.layers.LeakyReLU()(x)
        else:
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
    x = keras.layers.GlobalAveragePooling2D()(x)  # Use global pooling instead of flatten

    classifier_depth = hp.Int(
        "Number of Dense Layers", min_value=1, max_value=2, default=1
    )

    for i in range(classifier_depth):
        # Dense layer hyperparameters
        neuron_count = hp.Int(
            f"DenseBlock {i + 1}: Units",
            min_value=32,
            max_value=256,
            step=32,
            default=128,
        )
        regularization_factor = hp.Float(
            f"DenseBlock {i + 1}: Dropout Rate",
            min_value=0.0,
            max_value=0.5,
            step=0.1,
            default=0.3,
        )
        activation_function = hp.Choice(
            f"DenseBlock {i + 1}: Activation Function",
            ["relu", "leaky_relu"],
            default="relu",
        )

        # Regularization hyperparameters
        regularization_method = hp.Choice(
            f"DenseBlock {i + 1}: Regularization Strategy",
            ["none", "l2"],
            default="l2",
        )
        regularization_strength = hp.Float(
            f"DenseBlock {i + 1}: Regularization Strength",
            min_value=1e-5,
            max_value=1e-3,
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
        
        if activation_function == "leaky_relu":
            x = keras.layers.LeakyReLU()(x)
        else:
            x = keras.layers.Activation(activation_function)(x)

        if hp.Boolean(f"DenseBlock {i + 1}: Enable Batch Normalization", default=True):
            x = keras.layers.BatchNormalization()(x)

        x = keras.layers.Dropout(regularization_factor)(x)

    # --------------------
    # Learning Rate Configuration
    # --------------------
    learning_strategy = hp.Choice(
        "Learning Rate Schedule Type",
        ["constant", "exponential", "cosine"],
        default="cosine",
    )

    # Base learning rate for all strategies
    initial_learning_rate = hp.Float(
        "Initial Learning Rate",
        min_value=1e-4,
        max_value=1e-2,
        sampling="log",
        default=1e-3,
    )

    # Configure learning rate scheduler based on strategy
    if learning_strategy == "constant":
        learning_rate = initial_learning_rate
    elif learning_strategy == "exponential":
        decay_rate = hp.Float("Decay Rate", min_value=0.8, max_value=0.99, default=0.9)
        learning_rate = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_learning_rate,
            decay_rate=decay_rate,
            decay_steps=100,
        )
    else:  # cosine
        min_learning_rate = hp.Float(
            "Minimum Learning Rate", min_value=1e-6, max_value=1e-4, default=1e-5
        )
        learning_rate = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=CONFIG["EPOCHS"] * 10,  # Multiply by steps per epoch
            alpha=min_learning_rate,
        )

    # --------------------
    # Model Compilation
    # --------------------
    # Output layer with softmax activation for classification
    num_classes = 3  # rock, paper, scissors
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)

    optimizer = keras.optimizers.AdamW(learning_rate=learning_rate)

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

    # First, get class names from a raw dataset before applying transformations
    raw_train_ds = keras.utils.image_dataset_from_directory(
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
    )
    class_names = raw_train_ds.class_names
    print(f"Class names: {class_names}")
    
    # Now load the datasets with all transformations
    train_ds = create_dataset_train()
    val_ds = create_dataset_val()
    test_ds = create_dataset_test()
    
    # Print information about the datasets
    print("Dataset information:")
    for ds_name, ds in [("Training", train_ds), ("Validation", val_ds), ("Test", test_ds)]:
        print(f"  {ds_name} dataset batches: {len(list(ds))}")

    # Start TensorBoard for monitoring if enabled
    start_tensorboard()

    print("Starting hyperparameter tuning...")
    # Configure Hyperband tuner
    tuner = keras_tuner.Hyperband(
        build_model,
        objective="val_loss",
        max_epochs=CONFIG["EPOCHS"],
        factor=3,
        directory=OUTPUT_DIRS["hp_tuner"]["model"],
        project_name="hyperparameter_tuning",
        seed=CONFIG["RANDOM_STATE"],
    )

    # Perform hyperparameter search - use val_ds for validation, not test_ds
    tuner.search(
        train_ds,
        validation_data=val_ds,  # Use validation set, not test set
        epochs=CONFIG["EPOCHS"],
        callbacks=create_callbacks(),
    )

    # Get top 5 best hyperparameters
    best_hps = tuner.get_best_hyperparameters(10)
    best_models = []
    
    # Save hyperparameters and train models for each of the top 5 configurations
    for idx, hp in enumerate(best_hps):
        # Save hyperparameters to file
        with open(
            os.path.join(
                OUTPUT_DIRS["hp_tuner"]["model"], f"hyperparameters_hp_tuner_model{idx+1}.txt"
            ),
            "w",
        ) as f:
            f.write(f"Model {idx+1} Hyperparameters:\n")
            for param, value in hp.values.items():
                f.write(f"{param}: {value}\n")

        # Train model with these hyperparameters
        model = tuner.hypermodel.build(hp)
    
        # Use a longer training run for each model with proper callbacks
        final_callbacks = [
            keras.callbacks.ModelCheckpoint(
                os.path.join(OUTPUT_DIRS["hp_tuner"]["model"], f"model_hp_tuner_{idx+1}.keras"),
                monitor="val_loss",
                mode="min",
                save_best_only=True,
            ),
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=10, min_delta=0.01, restore_best_weights=True
            ),
            keras.callbacks.TensorBoard(
                log_dir=os.path.join(OUTPUT_DIRS["hp_tuner"]["logs"], f"final_model_{idx+1}")
            )
        ]
        
        print(f"\nTraining model {idx+1} of 5...")
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=CONFIG["EPOCHS"] * 2,  # Double epochs for final training
            callbacks=final_callbacks,
        )
        best_models.append(model)
    
        # Save the training history plot
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title(f'Model {idx+1} Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'Model {idx+1} Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIRS["hp_tuner"]["plots"], f'training_history_model{idx+1}.png'))
        plt.close()

        # Evaluate model on both validation and test datasets
        print(f"\nEvaluating model {idx+1} on validation dataset:")
        val_results = model.evaluate(val_ds)
        for metric_name, value in zip(model.metrics_names, val_results):
            print(f"  {metric_name}: {value:.4f}")
        
        print(f"\nEvaluating model {idx+1} on test dataset:")
        test_results = model.evaluate(test_ds)
        for metric_name, value in zip(model.metrics_names, test_results):
            print(f"  {metric_name}: {value:.4f}")

        # Create confusion matrices for both validation and test datasets
        def create_confusion_matrix(dataset, dataset_name, model_idx):
            all_predictions = []
            all_labels = []
            
            # Collect predictions and labels batch by batch
            for images, labels in dataset:
                predictions = model.predict_on_batch(images)
                predictions = tf.argmax(predictions, axis=1).numpy()
                labels = tf.argmax(labels, axis=1).numpy()
                
                all_predictions.extend(predictions)
                all_labels.extend(labels)
            
            cm = confusion_matrix(all_labels, all_predictions)
            
            # Create confusion matrix plot
            plt.figure(figsize=(10, 8))
            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm,
                display_labels=class_names
            )
            disp.plot(cmap='Blues', values_format='d')
            plt.title(f'Model {model_idx+1} Confusion Matrix - {dataset_name}')
            
            # Save the plot
            plt.savefig(os.path.join(OUTPUT_DIRS["hp_tuner"]["plots"], 
                       f'confusion_matrix_model{model_idx+1}_{dataset_name.lower()}.png'))
            plt.close()
            
            return cm
        
        # Create confusion matrices for current model
        val_cm = create_confusion_matrix(val_ds, "Validation Dataset", idx)
        test_cm = create_confusion_matrix(test_ds, "Test Dataset", idx)
        
        print(f"\nModel {idx+1} Confusion Matrix - Validation Dataset:")
        print(val_cm)
        
        print(f"\nModel {idx+1} Confusion Matrix - Test Dataset:")
        print(test_cm)
    
    print("\nTraining and evaluation complete.")


if __name__ == "__main__":
    main()
