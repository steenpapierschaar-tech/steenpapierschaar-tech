import keras
from src.config import config
from src.create_dataset import create_dataset
from src.create_plots import (
    plot_bias_variance,
    plot_confusion_matrix,
    plot_dataset_distribution,
    plot_metric_gap_analysis,
    plot_metrics_comparison,
    plot_training_history,
)
from src.tensorboard import TensorboardLauncher
from src.training_callbacks import ringring_callbackplease
from tensorflow import keras
from tensorflow.keras import layers, optimizers, regularizers


def build_model():
    inputs = keras.layers.Input(shape=(config.TARGET_SIZE[0], config.TARGET_SIZE[1], 3))
    x = keras.layers.Conv2D(
        224,
        (7, 7),
        activation="leaky_relu",
        padding="valid",
        kernel_regularizer=keras.regularizers.l2(0.0001633),
    )(inputs)
    # Layer 0
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0)(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 1: Conv
    x = keras.layers.Conv2D(
        224,
        (7, 7),
        activation="leaky_relu",
        padding="valid",
        kernel_regularizer=keras.regularizers.l2(0.0001633),
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 2: Conv
    x = keras.layers.Conv2D(
        224,
        (7, 7),
        activation="leaky_relu",
        padding="valid",
        kernel_regularizer=keras.regularizers.l2(0.0001633),
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Flatten before Dense layers
    x = keras.layers.Flatten()(x)

    # Dense Layer 0
    x = keras.layers.Dense(512, activation="leaky_relu")(x)
    x = keras.layers.Dropout(0.1)(x)

    # Output layer
    outputs = keras.layers.Dense(3, activation="softmax")(x)

    # Create model
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model


def main():
    # Load data
    train_ds, val_ds = create_dataset()
    tensorboard = TensorboardLauncher(config.LOGS_DIR)
    tensorboard.start_tensorboard()
    model = build_model()
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=0.00040288),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Print model summary
    model.summary()

    # Stop early if validation metrics meet criteria
    class CustomCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if logs.get("val_accuracy") > 0.92 and logs.get("val_loss") < 0.3:
                print("\nReached target metrics - stopping training")
                self.model.stop_training = True

    custom_callback = CustomCallback()
    # callbacks = ringring_callbackplease() + [custom_callback]

    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=config.MODEL_MANUAL_PATH,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
    )

    model.fit(
        train_ds,
        epochs=config.EPOCHS,
        validation_data=val_ds,
        callbacks=[model_checkpoint_callback],
    )

    # Save model
    # model.save(config.MODEL_MANUAL_PATH)

    plot_dataset_distribution()  # Dataset balance visualization
    plot_training_history(config.CSV_LOG_PATH)  # Training progress
    plot_confusion_matrix(model, val_ds)  # Classification performance
    plot_metrics_comparison(model, val_ds)  # Precision/Recall analysis
    plot_bias_variance(config.CSV_LOG_PATH)  # Bias-Variance tradeoff
    plot_metric_gap_analysis(config.CSV_LOG_PATH)  # Overfitting analysis


if __name__ == "__main__":
    main()
