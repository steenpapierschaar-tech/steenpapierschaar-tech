
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
    model.add(layers.Conv2D(32, (3, 3), activation='gelu', padding='same',
                            kernel_regularizer=regularizers.l2(0.0018481),
                            input_shape=(config.IMAGE_ROWS,config.IMAGE_COLS, 3)))  # Adjust input shape as needed
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    # add optimizer amamw 
    

    # Layer 1: Conv
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                            kernel_regularizer=regularizers.l2(0.0079166)))
    model.add(layers.BatchNormalization())
    model.add(layers.AveragePooling2D(pool_size=(4, 4)))



    # Layer 2: Conv
    model.add(layers.Conv2D(256, (3, 3), activation='leaky_relu', padding='same',
                            kernel_regularizer=regularizers.l2(0.0058566)))
    model.add(layers.Dropout(0.1))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # Layer 3: Conv
    model.add(layers.Conv2D(256, (3, 3), activation='tanh', padding='same',
                            kernel_regularizer=regularizers.l1(0.00015597)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.4))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # Flatten before Dense layers
    model.add(layers.Flatten())

    # Dense Layer 0
    model.add(layers.Dense(32, activation='relu',
                           kernel_regularizer=regularizers.l1(6.6062e-05)))
    model.add(layers.Dropout(0.1))

    # Dense Layer 1
    model.add(layers.Dense(288, activation='leaky_relu',
                           kernel_regularizer=regularizers.l1_l2(0.00010264)))
    model.add(layers.Dropout(0.2))

    # Dense Layer 2
    model.add(layers.Dense(160, activation='leaky_relu',
                           kernel_regularizer=regularizers.l1(0.00013522)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.1))

    # Output layer (adjust units and activation for classification)
    model.add(layers.Dense(3, activation='softmax'))  # Example: 10 classes for classification

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
