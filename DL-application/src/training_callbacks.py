import gc
import time

import keras
from src.config import config


class TimeoutCallback(keras.callbacks.Callback):
    def __init__(self, max_epoch_seconds):
        super().__init__()
        self.max_epoch_seconds = max_epoch_seconds
        self.current_epoch = 0

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch
        self.epoch_start_time = time.time()

    def on_batch_end(self, batch, logs=None):
        # Skip monitoring first epoch (epoch 0)
        if self.current_epoch == 0:
            return

        elapsed_seconds = time.time() - self.epoch_start_time
        if elapsed_seconds > self.max_epoch_seconds:
            print(
                f"\nStopping training: Epoch {self.current_epoch} took {elapsed_seconds:.2f} seconds (limit: {self.max_epoch_seconds} seconds)"
            )
            self.model.stop_training = True


class CustomCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get("val_accuracy") > 0.92 and logs.get("val_loss") < 0.3:
            print("\nReached target metrics - stopping training")
            self.model.stop_training = True

class MemoryCleanupCallback(keras.callbacks.Callback):
    def on_trial_end(self, logs=None):
        """Clear memory after each epoch"""
        gc.collect()  # Python garbage collection
        keras.utils.clear_session(free_memory=True)


def ringring_callbackplease(
    logs_dir=None,
    csv_log_path=None,
    use_model_checkpoint=False,
    use_early_stopping=True,
    use_csv_logger=True,
    use_timeout=True,
    use_custom_callback=False,
    use_tensorboard=False,
):
    """Create a list of callbacks for model training.

    This function configures various callbacks to monitor and improve training:
    - ModelCheckpoint: Save best model during training
    - EarlyStopping: Prevent overfitting by stopping when validation stops improving
    - TensorBoard: Visualize training metrics and model architecture
    - ReduceLROnPlateau: Adjust learning rate when training plateaus
    - CSVLogger: Save training history to CSV file
    - TerminateOnNaN: Stop training if loss becomes NaN
    - TimeoutCallback: Stop training if epoch exceeds time limit

    Returns:
        list: List of Keras callback objects
    """
    callbacks = []

    if use_model_checkpoint:
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                config.PATH_MANUAL_CNN_MODEL,
                monitor="val_loss",
                mode="min",
                save_best_only=True,
            )
        )

    if use_early_stopping:
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=10,
                min_delta=0.02,
                verbose=config.VERBOSE,
            )
        )

    if use_tensorboard:
        callbacks.append(
            keras.callbacks.TensorBoard(
                log_dir=logs_dir,
                histogram_freq=config.TENSORBOARD_HISTOGRAM_FREQ,
                write_steps_per_second=True,
                update_freq=config.TENSORBOARD_UPDATE_FREQ,
            )
        )

    if use_csv_logger:
        callbacks.append(
            keras.callbacks.CSVLogger(filename=csv_log_path, separator=",", append=True)
        )
        
        # # Stop training if NaN loss occurs
        # keras.callbacks.TerminateOnNaN(),
        
    if use_timeout:
        callbacks.append(TimeoutCallback(max_epoch_seconds=config.MAX_EPOCH_SECONDS))
        
    if use_custom_callback:
        callbacks.append(CustomCallback())

    return callbacks
