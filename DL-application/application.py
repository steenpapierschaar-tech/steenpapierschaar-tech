
import keras
import keras_tuner
from src.config import config
from src.create_dataset import create_dataset
from src.training_callbacks import ringring_callbackplease
from src.tensorboard import TensorboardLauncher
from src.augmentation import apply_augmentation
from tensorflow import keras
from tensorflow.keras import layers, regularizers, optimizers

def build_model():
    model = keras.Sequential()
    # Layer 0: Conv
    model.add(layers.Conv2D(224, (7, 7), activation='leaky_relu', padding='same',
                            kernel_regularizer=regularizers.l2(0.0001633),
                            input_shape=(config.IMAGE_ROWS, config.IMAGE_COLS, 3)))  # Adjust input shape as needed
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # Layer 1: Conv
    model.add(layers.Conv2D(224, (7, 7), activation='leaky_relu', padding='same',
                            kernel_regularizer=regularizers.l2(0.0001633)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # Layer 2: Conv
    model.add(layers.Conv2D(224, (7, 7), activation='leaky_relu', padding='same',
                            kernel_regularizer=regularizers.l2(0.0001633)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # Flatten before Dense layers
    model.add(layers.Flatten())

    # Dense Layer 0
    model.add(layers.Dense(96, activation='leaky_relu'))
    model.add(layers.Dropout(0.1))

    # Output layer (adjust units and activation for classification)
    model.add(layers.Dense(3, activation='softmax'))  # Example: 3 classes for classification

    return model


def main():
    # Load data
    train_ds, val_ds = create_dataset()
    tensorboard = TensorboardLauncher(config.LOGS_DIR)
    tensorboard.start_tensorboard()
    model = build_model()
    model.compile(optimizer=optimizers.AdamW(learning_rate=0.00040288),
            loss='categorical_crossentropy',
            metrics=['accuracy'])

    # Print model summary
    model.summary()
    model.fit(
        train_ds,
        epochs=config.EPOCHS,
        validation_data=val_ds,
        # callbacks=ringring_callbackplease,
    )
    
if __name__ == "__main__":
    main()
