import keras
from src.config import config
import os

def ringring_callbackplease():
    """Create a list of callbacks for model training.
    
    This function configures various callbacks to monitor and improve training:
    - ModelCheckpoint: Save best model during training
    - EarlyStopping: Prevent overfitting by stopping when validation stops improving
    - TensorBoard: Visualize training metrics and model architecture
    - ReduceLROnPlateau: Adjust learning rate when training plateaus
    - CSVLogger: Save training history to CSV file
    - TerminateOnNaN: Stop training if loss becomes NaN
    
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
        
        # # Stop training when validation metrics plateau
        # keras.callbacks.EarlyStopping(
        #     monitor="val_accuracy",
        #     patience=10,  # More patience to allow learning rate adjustments
        #     min_delta=0.01,
        #     verbose=config.VERBOSE,
        #     restore_best_weights=True,
        #     mode="max",
        #     start_from_epoch=5,  # Allow initial learning
        # ),
        
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
    ]

    return callbacks
