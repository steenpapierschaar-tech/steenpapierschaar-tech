
import keras
from src.config import config
from src.create_dataset import create_dataset
from src.training_callbacks import ringring_callbackplease
from src.tensorboard import TensorboardLauncher
from tensorflow import keras
from tensorflow.keras import layers, regularizers, optimizers

def build_model():
    inputs = keras.layers.Input(shape=(config.TARGET_SIZE[0], config.TARGET_SIZE[1], 3))
    x = keras.layers.Conv2D(224, (7, 7), activation='leaky_relu', padding='valid',
                     kernel_regularizer=keras.regularizers.l2(0.0001633))(inputs)
    # Layer 0
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0)(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 1: Conv
    x = keras.layers.Conv2D(224, (7, 7), activation='leaky_relu', padding='valid',
                     kernel_regularizer=keras.regularizers.l2(0.0001633))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 2: Conv
    x = keras.layers.Conv2D(224, (7, 7), activation='leaky_relu', padding='valid',
                     kernel_regularizer=keras.regularizers.l2(0.0001633))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Flatten before Dense layers
    x = keras.layers.Flatten()(x)

    # Dense Layer 0
    x = keras.layers.Dense(512, activation='leaky_relu')(x)
    x = keras.layers.Dropout(0.1)(x)

    # Output layer
    outputs = keras.layers.Dense(3, activation='softmax')(x)

    # Create model
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model


def main():
    # Load data
    train_ds, val_ds = create_dataset()
    tensorboard = TensorboardLauncher(config.LOGS_DIR)
    tensorboard.start_tensorboard()
    model = build_model()
    model.compile(optimizer=keras.optimizers.AdamW(learning_rate=0.00040288),
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
    
    # Save model
    model.save(config.MODEL_BEST_PATH)
    
if __name__ == "__main__":
    main()
