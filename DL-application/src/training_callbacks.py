import keras
from src.config import config
import os


def ringring_callbackplease():
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            config.MODEL_CHECKPOINT_PATH,
            monitor="val_loss",
            verbose=0,
            save_best_only=True,
            mode="auto",
            save_freq="epoch",
            initial_value_threshold=0.8,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=6,
            min_delta=0.01,
            verbose=config.VERBOSE,
            restore_best_weights=True,
            start_from_epoch=10,
        ),
        keras.callbacks.TensorBoard(
            log_dir=config.LOGS_DIR,
            #histogram_freq=config.TENSORBOARD_HISTOGRAM_FREQ,
            write_steps_per_second=True,
            #write_graph=True,
            #write_images=True,
            #profile_batch=(2, 2),
            update_freq=config.TENSORBOARD_UPDATE_FREQ,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            min_delta=0.1,
            factor=0.1,
            patience=3,
            verbose=config.VERBOSE,
            min_lr=0.000001,
        ),
        keras.callbacks.CSVLogger(
            filename=config.CSV_LOG_PATH, separator=",", append=False
        ),
        keras.callbacks.TerminateOnNaN(),
    ]

    return callbacks
