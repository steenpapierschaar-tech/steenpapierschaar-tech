import keras
import keras_tuner
from src.config import config
from src.create_dataset import create_dataset
from src.training_callbacks import ringring_callbackplease
from src.tensorboard import TensorboardLauncher
from src.augmentation import apply_augmentation


def build_model(hp):
    """Build CNN model with tunable hyperparameters"""

    # Input layer
    inputs = keras.Input(shape=(None, None, 3))

    # Apply augmentation pipeline
    x = apply_augmentation(inputs)

    # Rescale pixel values
    x = keras.layers.Rescaling(1.0 / 255)(x)

    # Normalization
    norm_choice = hp.Choice("normalization", ["standard", "layer", "batch", "none"])
    if norm_choice == "standard":
        x = keras.layers.Normalization()(x)
    elif norm_choice == "layer":
        x = keras.layers.LayerNormalization()(x)
    elif norm_choice == "batch":
        x = keras.layers.BatchNormalization()(x)

    # Convolutional layers
    n_conv_layers = hp.Int(
        "Amount of convolutional layers", min_value=1, max_value=4, default=1
    )

    for i in range(n_conv_layers):
        # Conv layer hyperparameters
        filters = hp.Int(
            f"Layer {i}: Amount of filters",
            min_value=32,
            max_value=256,
            step=32,
            default=32,
        )
        activation_function = hp.Choice(
            f"Layer {i}: Activation function",
            ["relu", "leaky_relu", "selu", "tanh", "gelu"],
            default="relu",
        )
        dropout_rate = hp.Float(
            f"Layer {i}: dropout rate",
            min_value=0.0,
            max_value=0.5,
            step=0.1,
            default=0.0,
        )
        kernel_size = hp.Choice(f"Layer {i}: kernel size", [3, 5, 7], default=3)

        # Regularizer hyperparameters
        conv_reg_type = hp.Choice(
            f"Conv Layer {i}: Regularizer type",
            ["none", "l1", "l2", "l1_l2"],
            default="none",
        )
        conv_reg_factor = hp.Float(
            f"Conv Layer {i}: Regularizer factor",
            min_value=1e-5,
            max_value=1e-2,
            sampling="log",
            default=1e-4,
        )

        # Create regularizer
        if conv_reg_type == "l1":
            conv_regularizer = keras.regularizers.L1(conv_reg_factor)
        elif conv_reg_type == "l2":
            conv_regularizer = keras.regularizers.L2(conv_reg_factor)
        elif conv_reg_type == "l1_l2":
            conv_regularizer = keras.regularizers.L1L2(
                l1=conv_reg_factor, l2=conv_reg_factor
            )
        else:
            conv_regularizer = None

        # Layer construction
        if hp.Boolean(f"Layer {i}: Use batch normalization"):
            x = keras.layers.BatchNormalization()(x)

        x = keras.layers.Conv2D(
            filters, kernel_size, padding="same", kernel_regularizer=conv_regularizer
        )(x)
        x = keras.layers.Activation(activation_function)(x)

        pool_type = hp.Choice(f"pool_{i}", ["max", "avg"])
        if pool_type == "max":
            x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        else:
            x = keras.layers.AveragePooling2D(pool_size=(2, 2))(x)

        x = keras.layers.Dropout(dropout_rate)(x)

    # Flatten
    x = keras.layers.Flatten()(x)

    # Dense layers
    n_dense_layers = hp.Int(
        "Amount of dense layers", min_value=1, max_value=3, default=1
    )

    for i in range(n_dense_layers):
        # Dense layer hyperparameters
        units = hp.Int(
            f"Dense layer {i}: Units", min_value=32, max_value=512, step=64, default=32
        )
        dropout_rate = hp.Float(
            f"Dense layer {i}: dropout rate",
            min_value=0.0,
            max_value=0.5,
            step=0.1,
            default=0.0,
        )
        activation_function = hp.Choice(
            f"Dense layer {i}: Activation function",
            ["relu", "selu", "leaky_relu"],
            default="relu",
        )

        # Regularizer hyperparameters
        dense_reg_type = hp.Choice(
            f"Dense Layer {i}: Regularizer type",
            ["none", "l1", "l2", "l1_l2"],
            default="none",
        )
        dense_reg_factor = hp.Float(
            f"Dense Layer {i}: Regularizer factor",
            min_value=1e-5,
            max_value=1e-2,
            sampling="log",
            default=1e-4,
        )

        # Create regularizer
        if dense_reg_type == "l1":
            dense_regularizer = keras.regularizers.L1(dense_reg_factor)
        elif dense_reg_type == "l2":
            dense_regularizer = keras.regularizers.L2(dense_reg_factor)
        elif dense_reg_type == "l1_l2":
            dense_regularizer = keras.regularizers.L1L2(
                l1=dense_reg_factor, l2=dense_reg_factor
            )
        else:
            dense_regularizer = None

        # Layer construction
        x = keras.layers.Dense(units, kernel_regularizer=dense_regularizer)(x)
        x = keras.layers.Activation(activation_function)(x)

        if hp.Boolean(f"Dense layer {i}: Use batch normalization"):
            x = keras.layers.BatchNormalization()(x)

        x = keras.layers.Dropout(dropout_rate)(x)

    # Output layer
    outputs = keras.layers.Dense(3, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)

    # Optimizer
    optimizer_choice = hp.Choice(
        "optimizer", ["adam", "rmsprop", "sgd", "AdamW"], default="adam"
    )

    learning_rate = 0.001

    if optimizer_choice == "adam":
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_choice == "rmsprop":
        optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
    elif optimizer_choice == "AdamW":
        optimizer = keras.optimizers.AdamW(learning_rate=learning_rate)
    else:
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def main():
    # Allow mixed precision
    keras.mixed_precision.set_global_policy("mixed_float16")

    # Set global random seed
    keras.utils.set_random_seed(config.RANDOM_STATE)

    # Clear session and memory
    keras.utils.clear_session(free_memory=True)

    # Load data
    train_ds, val_ds = create_dataset()

    # Initialize TensorBoard
    tensorboard = TensorboardLauncher(config.LOGS_DIR)
    tensorboard.start_tensorboard()

    # Create tuner
    tuner = keras_tuner.BayesianOptimization(
        build_model,
        objective="val_loss",
        max_trials=config.MAX_TRIALS,
        directory=config.OUTPUT_DIR,
    )

    # Hyperparameter search
    tuner.search(
        train_ds,
        validation_data=val_ds,
        epochs=config.EPOCHS,
        callbacks=ringring_callbackplease(),
    )

    # Save best hyperparameters
    best_hp = tuner.get_best_hyperparameters(1)[0]
    with open(config.BEST_HP_PATH, "w") as f:
        for param, value in best_hp.values.items():
            f.write(f"{param}: {value}\n")

    # Build and save best model
    best_model = tuner.hypermodel.build(best_hp)
    best_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.EPOCHS,
        callbacks=ringring_callbackplease(),
    )
    best_model.save(config.BEST_MODEL_PATH)


if __name__ == "__main__":
    main()
