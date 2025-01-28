import keras
from config import config
import os

def ringring_callbackplease():
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            config.CHECKPOINT_MODEL_PATH,
            monitor='val_loss',
            verbose=config.VERBOSE,
            save_best_only=True,
            mode="auto",
            save_freq="epoch",
            initial_value_threshold=0.4
        ),
        
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=8,
            min_delta=0.01,
            verbose=config.VERBOSE,
            restore_best_weights=True,
            start_from_epoch=10
        ),
        
        keras.callbacks.TensorBoard(
            log_dir=config.LOGS_PATH,
            histogram_freq=config.TENSORBOARD_HISTOGRAM_FREQ,
            write_graph=True,
            write_images=True,
            update_freq='epoch'
        ),

        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            min_delta=0.01,
            factor=0.2,
            patience=4,
            verbose=config.VERBOSE,
        ),

        keras.callbacks.CSVLogger(
            filename=os.path.join(config.HISTORY_PATH, "log.csv"),
            separator=',',
            append=False
        ),
        
        keras.callbacks.TerminateOnNaN(),
        
        keras.callbacks.ProgbarLogger()
    ]
    
    return callbacks