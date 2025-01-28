import keras
from src.config import config
import os

class BestHyperparameterSaver(keras.callbacks.Callback):
    """Keras Tuner callback to save best hyperparameters after each trial"""
    def __init__(self, filepath):
        super().__init__()
        self.filepath = filepath
        self.best_score = float('inf')
        self.best_trial_id = None

    def on_trial_end(self, trial, logs=None):
        # Get the tuner object from the parent model
        tuner = self.model.optimizer.tuner if hasattr(self.model.optimizer, 'tuner') else None
        if not tuner:
            return

        # Get current best trial
        best_trials = tuner.oracle.get_best_trials(num_trials=1)
        if not best_trials:
            return
            
        current_best_trial = best_trials[0]
        
        # Only save if we have a new best trial
        if current_best_trial.trial_id != self.best_trial_id:
            self.best_score = current_best_trial.score
            self.best_trial_id = current_best_trial.trial_id
            
            content = [
                f"Best Trial ID: {current_best_trial.trial_id}",
                f"Validation Score: {current_best_trial.score:.4f}",
                "\nHyperparameters:"
            ]
            content += [f"{k}: {v}" for k, v in current_best_trial.hyperparameters.values.items()]
            
            with open(self.filepath, 'w') as f:
                f.write("\n".join(content))

def ringring_callbackplease():
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            config.MODEL_CHECKPOINT_PATH,
            monitor='val_loss',
            verbose=config.VERBOSE,
            save_best_only=True,
            mode="auto",
            save_freq="epoch",
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
            log_dir=config.LOGS_DIR,
            histogram_freq=config.TENSORBOARD_HISTOGRAM_FREQ,
            write_graph=True,
            write_images=True,
            update_freq=config.TENSORBOARD_UPDATE_FREQ
        ),

        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            min_delta=0.01,
            factor=0.1,
            patience=4,
            verbose=config.VERBOSE,
            min_lr=0.00001
        ),

        keras.callbacks.CSVLogger(
            filename=config.CSV_LOG_PATH,
            separator=',',
            append=False
        ),
        
        keras.callbacks.TerminateOnNaN(),
        
        BestHyperparameterSaver(config.HYPERPARAMS_PATH),
    ]
    
    return callbacks