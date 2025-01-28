import keras
from config import config

def ringring_callbackplease():
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            config.CHECKPOINT_MODEL_PATH,
            monitor='val_loss',
            verbose=config.VERBOSE,
            save_best_only=True,
            mode="auto",
            save_freq="epoch"
            initial_value_threshold=0.4
        ),
        
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=6,
            verbose=config.VERBOSE,
            restore_best_weights=True
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
            factor=1e-2,
            patience=3,
            verbose=config.VERBOSE,
            min_delta=0.01
        )
        
        keras.callbacks.LearningRateScheduler(
            
    ]
    
    return callbacks