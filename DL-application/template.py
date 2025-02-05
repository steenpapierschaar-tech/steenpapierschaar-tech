import keras
from src.config import config
from src.create_dataset import create_dataset
from src.training_callbacks import ringring_callbackplease
from src.tensorboard import TensorboardLauncher
from src.augmentation import apply_augmentation


def build_model():
    """Build CNN model with fixed hyperparameters"""

    # Input layer
    inputs = keras.Input(shape=(None, None, 3))

    # Apply augmentation pipeline
    x = apply_augmentation(inputs)

    # Rescale pixel values
    x = keras.layers.Rescaling(1.0 / 255)(x)

    # Normalization (default: "standard")
    x = keras.layers.Normalization()(x)

    # Layer construction
    x = keras.layers.Conv2D(filters=64, kernel_size=3)(x)
    x = keras.layers.Activation(activation="relu")(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = keras.layers.Dropout(rate=0.1)(x)

    # Flatten
    x = keras.layers.Flatten()(x)

    # Dense layers
    x = keras.layers.Dense(units=64)(x)
    x = keras.layers.Activation(activation="relu")(x)
    x = keras.layers.Dropout(rate=0.1)(x)

    # Output layer
    outputs = keras.layers.Dense(3, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)

    # Optimizer (default: "adam" with lr=0.001)
    optimizer = keras.optimizers.Adam(learning_rate=0.001)

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

    # Build and train model
    model = build_model()
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.EPOCHS,
        callbacks=ringring_callbackplease(),
    )
    model.save(config.MODEL_BEST_PATH)


if __name__ == "__main__":
    main()
