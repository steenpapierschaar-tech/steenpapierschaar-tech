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

    # Layer 0: Conv
    x = keras.layers.Conv2D(32, (3, 3), activation='gelu', padding='same',
                            kernel_regularizer=keras.regularizers.l2(0.0018481))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 1: Conv
    x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                            kernel_regularizer=keras.regularizers.l2(0.0079166))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.AveragePooling2D(pool_size=(4, 4))(x)

    # Layer 2: Conv
    x = keras.layers.Conv2D(256, (3, 3), activation='leaky_relu', padding='same',
                            kernel_regularizer=keras.regularizers.l2(0.0058566))(x)
    x = keras.layers.Dropout(0.1)(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 3: Conv
    x = keras.layers.Conv2D(256, (3, 3), activation='tanh', padding='same',
                            kernel_regularizer=keras.regularizers.l1(0.00015597))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Flatten before Dense layers
    x = keras.layers.Flatten()(x)

    # Dense Layer 0
    x = keras.layers.Dense(32, activation='relu',
                           kernel_regularizer=keras.regularizers.l1(6.6062e-05))(x)
    x = keras.layers.Dropout(0.1)(x)

    # Dense Layer 1
    x = keras.layers.Dense(288, activation='leaky_relu',
                           kernel_regularizer=keras.regularizers.l1_l2(0.00010264))(x)
    x = keras.layers.Dropout(0.2)(x)

    # Dense Layer 2
    x = keras.layers.Dense(160, activation='leaky_relu',
                           kernel_regularizer=keras.regularizers.l1(0.00013522))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.1)(x)

    # Output layer
    outputs = keras.layers.Dense(3, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)

    # Optimizer (default: "adam" with lr=0.001)
    optimizer = keras.optimizers.AdamW(learning_rate=0.01)

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
