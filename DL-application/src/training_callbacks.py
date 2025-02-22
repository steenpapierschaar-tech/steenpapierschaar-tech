import keras
from src.config import config
import os
import time
import gc

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
            print(f"\nStopping training: Epoch {self.current_epoch} took {elapsed_seconds:.2f} seconds (limit: {self.max_epoch_seconds} seconds)")
            self.model.stop_training = True

class MemoryCleanupCallback(keras.callbacks.Callback):
    def on_trial_end(self, logs=None):
        """Clear memory after each epoch"""
        gc.collect()  # Python garbage collection
        keras.utils.clear_session(free_memory=True)

def ringring_callbackplease():
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
    callbacks = [
        # # Save best model during training
        # keras.callbacks.ModelCheckpoint(
        #     config.MODEL_CHECKPOINT_PATH,
        #     monitor="val_accuracy",  # Monitor validation accuracy instead of loss
        #     verbose=1,
        #     save_best_only=True,
        #     mode="max",  # Save when accuracy increases
        #     save_freq="epoch",
        # ),
        
        # Stop training when validation metrics plateau
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            min_delta=0.02,
            verbose=config.VERBOSE,
        ),
        
        # Visualize training progress
        keras.callbacks.TensorBoard(
            log_dir=config.LOGS_DIR,
            histogram_freq=config.TENSORBOARD_HISTOGRAM_FREQ,
            write_steps_per_second=True,
        #    write_graph=True,
        #    write_images=True,
            update_freq=config.TENSORBOARD_UPDATE_FREQ,
        ),
        
        # # Reduce learning rate when training plateaus
        # keras.callbacks.ReduceLROnPlateau(
        #     monitor="val_accuracy",
        #     mode="max",
        #     min_delta=0.01,
        #     factor=0.5,  # Halve the learning rate
        #     patience=5,
        #     verbose=config.VERBOSE,
        #     min_lr=1e-6,
        # ),
        
        # Log training metrics to CSV
        keras.callbacks.CSVLogger(
            filename=config.CSV_LOG_PATH,
            separator=",",
            append=False
        ),
        
        # # Stop training if NaN loss occurs
        # keras.callbacks.TerminateOnNaN(),
        
        # Stop training if epoch exceeds time limit (5 minutes default)
        TimeoutCallback(max_epoch_seconds=config.MAX_EPOCH_SECONDS),
        
        # Clear memory after each trial
        # MemoryCleanupCallback(),
    ]

    return callbacks
