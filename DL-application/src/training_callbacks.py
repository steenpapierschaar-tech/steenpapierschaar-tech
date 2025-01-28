import keras
from src.config import config
import os


def ringring_callbackplease():
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            config.MODEL_CHECKPOINT_PATH,
            monitor="val_loss",
            verbose=config.VERBOSE,
            save_best_only=True,
            mode="auto",
            save_freq="epoch",
            initial_value_threshold=0.5,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            min_delta=0.01,
            verbose=config.VERBOSE,
            restore_best_weights=True,
            start_from_epoch=10,
        ),
        keras.callbacks.TensorBoard(
            log_dir=config.LOGS_DIR,
            histogram_freq=config.TENSORBOARD_HISTOGRAM_FREQ,
            write_graph=True,
            write_images=True,
            update_freq=config.TENSORBOARD_UPDATE_FREQ,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            min_delta=0.01,
            factor=0.1,
            patience=4,
            verbose=config.VERBOSE,
            min_lr=0.00001,
        ),
        keras.callbacks.CSVLogger(
            filename=config.CSV_LOG_PATH, separator=",", append=False
        ),
        keras.callbacks.TerminateOnNaN(),
    ]

    return callbacks
