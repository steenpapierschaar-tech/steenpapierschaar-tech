# Standard library imports
import sys
from pathlib import Path

# Third-party imports
import keras
import keras_tuner

# Local application imports
from src.config import config
from src.create_dataset import create_dataset
from src.training_callbacks import ringring_callbackplease
from src.tensorboard import TensorboardLauncher
from src.augmentation import apply_augmentation

#--------------------
# Model Architecture
#--------------------
def build_model(hp):
    """Build a CNN model with tunable hyperparameters.
    
    This function constructs a Convolutional Neural Network with a dynamic 
    architecture determined by hyperparameters. The model includes:
    - Configurable number of convolutional and dense layers
    - Various normalization options
    - Flexible learning rate schedules
    - Multiple optimizer choices
    - Regularization options
    
    Args:
        hp: Keras Tuner hyperparameter object for defining search space
        
    Returns:
        keras.Model: Compiled model ready for training
    """
    # Input layer with flexible dimensions for variable image sizes
    inputs = keras.Input(shape=(None, None, 3))

    # Apply data augmentation pipeline (includes rescaling)
    x = apply_augmentation(inputs)

    #--------------------
    # Normalization Layer
    #--------------------
    normalization_strategy = hp.Choice(
        "Normalization Type",
        ["standard", "layer", "batch", "none"],
        default="batch"
    )
    if normalization_strategy == "standard":
        x = keras.layers.Normalization()(x)
    elif normalization_strategy == "layer":
        x = keras.layers.LayerNormalization()(x)
    elif normalization_strategy == "batch":
        x = keras.layers.BatchNormalization()(x)

    #--------------------
    # Convolutional Layers
    #--------------------
    architecture_depth = hp.Int(
        "Number of Convolutional Layers",
        min_value=1,
        max_value=2,
        default=1
    )

    for i in range(architecture_depth):
        # Conv layer hyperparameters with GPU-optimized ranges
        feature_extractors = hp.Int(
            f"ConvBlock {i+1}: Filters",
            min_value=32,    # Minimum for feature detection
            max_value=256,   # Maximum before diminishing returns
            step=32,         # Power of 2 for GPU optimization
            default=32
        )
        activation_function = hp.Choice(
            f"ConvBlock {i+1}: Activation Function",
            ["relu", "leaky_relu", "selu", "tanh", "gelu"],
            default="relu"
        )
        regularization_factor = hp.Float(
            f"ConvBlock {i+1}: Dropout Rate",
            min_value=0.0,
            max_value=0.5,
            step=0.1,
            default=0.0
        )
        perception_field = hp.Choice(
            f"ConvBlock {i+1}: Kernel Size",
            [3, 5, 7],
            default=3
        )

        # Regularization hyperparameters
        regularization_method = hp.Choice(
            f"ConvBlock {i+1}: Regularization Strategy",
            ["none", "l1", "l2", "l1_l2"],
            default="none"
        )
        regularization_strength = hp.Float(
            f"ConvBlock {i+1}: Regularization Strength",
            min_value=1e-5,
            max_value=1e-2,
            sampling="log",
            default=1e-4
        )

        # Create regularizer based on selected strategy
        if regularization_method == "l1":
            conv_regularizer = keras.regularizers.L1(regularization_strength)
        elif regularization_method == "l2":
            conv_regularizer = keras.regularizers.L2(regularization_strength)
        elif regularization_method == "l1_l2":
            conv_regularizer = keras.regularizers.L1L2(
                l1=regularization_strength,
                l2=regularization_strength
            )
        else:
            conv_regularizer = None

        # Convolutional layer construction
        if hp.Boolean(f"ConvBlock {i+1}: Enable Batch Normalization"):
            x = keras.layers.BatchNormalization()(x)

        x = keras.layers.Conv2D(
            feature_extractors,
            perception_field,
            padding="same",
            kernel_regularizer=conv_regularizer
        )(x)
        x = keras.layers.Activation(activation_function)(x)

        # Spatial Reduction layer
        dimensionality_reduction = hp.Choice(
            f"ConvBlock {i+1}: Spatial Reduction",
            ["maximum", "average"],
            default="maximum"
        )
        if dimensionality_reduction == "maximum":
            x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        else:  # average
            x = keras.layers.AveragePooling2D(pool_size=(2, 2))(x)

        x = keras.layers.Dropout(regularization_factor)(x)

    #--------------------
    # Dense Layers
    #--------------------
    x = keras.layers.Flatten()(x)

    classifier_depth = hp.Int(
        "Number of Dense Layers",
        min_value=1,
        max_value=1,
        default=1
    )

    for i in range(classifier_depth):
        # Dense layer hyperparameters
        neuron_count = hp.Int(
            f"DenseBlock {i+1}: Units",
            min_value=16,
            max_value=256,
            step=32,
            default=32
        )
        regularization_factor = hp.Float(
            f"DenseBlock {i+1}: Dropout Rate",
            min_value=0.0,
            max_value=0.5,
            step=0.1,
            default=0.0
        )
        activation_function = hp.Choice(
            f"DenseBlock {i+1}: Activation Function",
            ["relu", "selu", "leaky_relu"],
            default="relu"
        )

        # Regularization hyperparameters
        regularization_method = hp.Choice(
            f"DenseBlock {i+1}: Regularization Strategy",
            ["none", "l1", "l2", "l1_l2"],
            default="none"
        )
        regularization_strength = hp.Float(
            f"DenseBlock {i+1}: Regularization Strength",
            min_value=1e-5,
            max_value=1e-2,
            sampling="log",
            default=1e-4
        )

        # Create regularizer based on selected strategy
        if regularization_method == "l1":
            dense_regularizer = keras.regularizers.L1(regularization_strength)
        elif regularization_method == "l2":
            dense_regularizer = keras.regularizers.L2(regularization_strength)
        elif regularization_method == "l1_l2":
            dense_regularizer = keras.regularizers.L1L2(
                l1=regularization_strength,
                l2=regularization_strength
            )
        else:
            dense_regularizer = None

        # Dense layer construction
        x = keras.layers.Dense(
            neuron_count,
            kernel_regularizer=dense_regularizer
        )(x)
        x = keras.layers.Activation(activation_function)(x)

        if hp.Boolean(f"DenseBlock {i+1}: Enable Batch Normalization"):
            x = keras.layers.BatchNormalization()(x)

        x = keras.layers.Dropout(regularization_factor)(x)

    #--------------------
    # Learning Rate Configuration
    #--------------------
    learning_strategy = hp.Choice(
        "Learning Rate Schedule Type",
        ["constant", "step", "exponential", "polynomial", "cosine"],
        default="constant"
    )

    # Base learning rate for all strategies
    initial_learning_rate = hp.Float(
        "Initial Learning Rate",
        min_value=1e-4,
        max_value=1e-1,
        sampling="log",
        default=1e-3
    )

    # Configure learning rate scheduler based on strategy
    if learning_strategy == "constant":
        learning_rate = initial_learning_rate
    elif learning_strategy == "step":
        decay_factor = hp.Float(
            "Decay Rate",
            min_value=0.1,
            max_value=0.5,
            default=0.1
        )
        learning_rate = keras.optimizers.schedules.InverseTimeDecay(
            initial_learning_rate=initial_learning_rate,
            decay_rate=decay_factor,
            decay_steps=1,
            staircase=True
        )
    elif learning_strategy == "exponential":
        decay_rate = hp.Float(
            "Decay Rate",
            min_value=0.8,
            max_value=0.99,
            default=0.9
        )
        learning_rate = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_learning_rate,
            decay_rate=decay_rate,
            decay_steps=1
        )
    elif learning_strategy == "polynomial":
        final_learning_rate = hp.Float(
            "Final Learning Rate",
            min_value=1e-7,
            max_value=1e-4,
            sampling="log",
            default=1e-6
        )
        decay_power = hp.Float(
            "Decay Power",
            min_value=1.0,
            max_value=3.0,
            default=1.0
        )
        learning_rate = keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=config.EPOCHS,
            end_learning_rate=final_learning_rate,
            power=decay_power
        )
    else:  # cosine
        min_learning_rate = hp.Float(
            "Minimum Learning Rate",
            min_value=0.0,
            max_value=0.2,
            default=0.0
        )
        learning_rate = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=config.EPOCHS,
            alpha=min_learning_rate
        )

    #--------------------
    # Model Compilation
    #--------------------
    # Output layer with softmax activation for classification
    outputs = keras.layers.Dense(3, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)

    # Optimizer selection and configuration
    optimization_strategy = hp.Choice(
        "Optimizer Type",
        ["Adam", "RMSprop", "SGD", "AdamW"],
        default="Adam"
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
        metrics=["accuracy"]
    )

    return model

#--------------------
# Training Setup
#--------------------
def main():
    """Main training pipeline for hyperparameter tuning"""
    # Enable mixed precision for improved GPU utilization
    keras.mixed_precision.set_global_policy("mixed_float16")

    # Set random seed for reproducibility
    keras.utils.set_random_seed(config.RANDOM_STATE)

    # Clear session and memory
    keras.utils.clear_session(free_memory=True)

    # Load and prepare datasets
    train_ds, val_ds = create_dataset()

    # Initialize TensorBoard for monitoring
    tensorboard = TensorboardLauncher(config.LOGS_DIR)
    tensorboard.start_tensorboard()

    # Configure Bayesian Optimization tuner
    tuner = keras_tuner.BayesianOptimization(
        build_model,
        objective="val_loss",
        max_trials=config.MAX_TRIALS,
        directory=config.OUTPUT_DIR
    )

    # Perform hyperparameter search
    tuner.search(
        train_ds,
        validation_data=val_ds,
        epochs=config.EPOCHS,
        callbacks=ringring_callbackplease()
    )

    # Save best hyperparameters
    best_hp = tuner.get_best_hyperparameters(1)[0]
    with open(config.HYPERPARAMS_PATH, "w") as f:
        for param, value in best_hp.values.items():
            f.write(f"{param}: {value}\n")

    # Train final model with best hyperparameters
    best_model = tuner.hypermodel.build(best_hp)
    best_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.EPOCHS,
        callbacks=ringring_callbackplease()
    )
    best_model.save(config.BEST_MODEL_PATH)

if __name__ == "__main__":
    main()
