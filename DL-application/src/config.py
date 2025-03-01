import datetime
import os
import inspect
from pathlib import Path

class Config:
    """Centralized configuration for the DL application with smart directory management"""

    def __init__(self):
        # General
        self.VERBOSE = 1
        self.TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # Model existence check and training control
        self.SKIP_TRAINING_MANUAL_CNN = False  # Will be updated after path initialization
        self.SKIP_TRAINING_AUTO_KERAS = False  # Will be updated after path initialization
        self.SKIP_TRAINING_HP_TUNER = False   # Will be updated after path initialization
        
        # Data Loading
        self.IMAGE_DIMS = (240, 320)
        self.IMAGE_ROWS = self.IMAGE_DIMS[0]
        self.IMAGE_COLS = self.IMAGE_DIMS[1]
        self.VALIDATION_SPLIT = 0.2
        self.RANDOM_STATE = 42
        self.EXTERNAL_DATASET_USAGE = False

        # Augmentation
        self.AUGMENTATION_ENABLED = True
        self.RANDOM_BRIGHTNESS = 0.1
        self.RANDOM_CONTRAST = 0.6
        self.RANDOM_SATURATION = (0.4, 0.6)
        self.RANDOM_HUE = 0.1
        self.RANDOM_CROP = 0.1
        self.RANDOM_FLIP = "horizontal_and_vertical"
        self.RANDOM_SHARPNESS = 0.1
        self.RANDOM_SHEAR_X = 0.1
        self.RANDOM_SHEAR_Y = 0.1
        self.RANDOM_TRANSLATION = 0.1
        self.RANDOM_ZOOM = 0.1
        self.RANDOM_ROTATION = 0.1

        # Training
        self.EPOCHS = 100
        self.BATCH_SIZE = 16
        self.TARGET_SIZE = (96, 96)
        self.MAX_TRIALS = 100
        self.MAX_EPOCH_SECONDS = 10

        # Input
        self.DATASET_ROOT_DIR = "photoDataset"
        self.INPUT_DIRECTORY = "input"
        self.DATASET_EXTERNAL_DIR = "photoDataset_external"
        #self.DATASET_EXTERNAL_DIR = "photoDataset"

        # Tensorboard
        self.TENSORBOARD_PORT = 6006
        self.TENSORBOARD_UPDATE_FREQ = "batch"
        self.TENSORBOARD_HISTOGRAM_FREQ = 1

        # Metrics and Analysis
        self.METRICS = ["accuracy", "precision", "recall"]
        self.PLOT_WINDOW_SIZE = 5  # For moving averages in plots
        self.CLASS_NAMES = ["rock", "paper", "scissors"]  # Known classes from dataset

        # Initialize paths based on the calling script
        self._initialize_paths()

    def _initialize_paths(self):
        """Initialize directory and file paths based on the calling script"""
        # Base output directory
        self.OUTPUT_DIR = "output"

        # Strategies
        self.STRAT_MANUAL_CNN = "manual_cnn"
        self.STRAT_AUTO_KERAS = "auto_keras"
        self.STRAT_HP_TUNER = "hp_tuner"
        
        # Define subdirectories
        self.OUTPUT_MANUAL_CNN = os.path.join(self.OUTPUT_DIR, self.STRAT_MANUAL_CNN)
        self.OUTPUT_AUTO_KERAS = os.path.join(self.OUTPUT_DIR, self.STRAT_AUTO_KERAS)
        self.OUTPUT_HP_TUNER = os.path.join(self.OUTPUT_DIR, self.STRAT_HP_TUNER)

        # Create sub sub directories
        self.DIR_MANUAL_CNN_MODEL = os.path.join(self.OUTPUT_MANUAL_CNN, "model")
        self.DIR_MANUAL_CNN_HISTORY = os.path.join(self.OUTPUT_MANUAL_CNN, "history")
        self.DIR_MANUAL_CNN_LOGS = os.path.join(self.OUTPUT_MANUAL_CNN, "logs")
        self.DIR_MANUAL_CNN_PLOTS = os.path.join(self.OUTPUT_MANUAL_CNN, "plots")

        self.DIR_AUTO_KERAS_MODEL = os.path.join(self.OUTPUT_AUTO_KERAS, "model")
        self.DIR_AUTO_KERAS_HISTORY = os.path.join(self.OUTPUT_AUTO_KERAS, "history")
        self.DIR_AUTO_KERAS_LOGS = os.path.join(self.OUTPUT_AUTO_KERAS, "logs")
        self.DIR_AUTO_KERAS_PLOTS = os.path.join(self.OUTPUT_AUTO_KERAS, "plots")

        self.DIR_HP_TUNER_MODEL = os.path.join(self.OUTPUT_HP_TUNER, "model")
        self.DIR_HP_TUNER_HISTORY = os.path.join(self.OUTPUT_HP_TUNER, "history")
        self.DIR_HP_TUNER_LOGS = os.path.join(self.OUTPUT_HP_TUNER, "logs")
        self.DIR_HP_TUNER_PLOTS = os.path.join(self.OUTPUT_HP_TUNER, "plots")
        
        # Define resulting paths (directory + filename)
        self.PATH_MANUAL_CNN_MODEL = os.path.join(self.DIR_MANUAL_CNN_MODEL, "model_manual_cnn.keras")
        self.PATH_MANUAL_CNN_BEST = os.path.join(self.DIR_MANUAL_CNN_MODEL, "model_best_manual_cnn.keras")
        self.PATH_MANUAL_CNN_CHECKPOINT = os.path.join(self.DIR_MANUAL_CNN_MODEL, "model_checkpoint_manual_cnn.keras")
        self.PATH_MANUAL_CNN_HYPERPARAMS = os.path.join(self.DIR_MANUAL_CNN_MODEL, "hyperparameters_manual_cnn.txt")
        self.PATH_MANUAL_CNN_HISTORY = os.path.join(self.DIR_MANUAL_CNN_HISTORY, "training_history_manual_cnn.json")
        self.PATH_MANUAL_CNN_LOG = os.path.join(self.DIR_MANUAL_CNN_LOGS, "log_manual_cnn.csv")

        self.PATH_AUTO_KERAS_MODEL = os.path.join(self.DIR_AUTO_KERAS_MODEL, "model_auto_keras.keras")
        self.PATH_AUTO_KERAS_BEST = os.path.join(self.DIR_AUTO_KERAS_MODEL, "model_best_auto_keras.keras")
        self.PATH_AUTO_KERAS_CHECKPOINT = os.path.join(self.DIR_AUTO_KERAS_MODEL, "model_checkpoint_auto_keras.keras")
        self.PATH_AUTO_KERAS_HYPERPARAMS = os.path.join(self.DIR_AUTO_KERAS_MODEL, "hyperparameters_auto_keras.txt")
        self.PATH_AUTO_KERAS_HISTORY = os.path.join(self.DIR_AUTO_KERAS_HISTORY, "training_history_auto_keras.json")
        self.PATH_AUTO_KERAS_LOG = os.path.join(self.DIR_AUTO_KERAS_LOGS, "log_auto_keras.csv")

        self.PATH_HP_TUNER_MODEL = os.path.join(self.DIR_HP_TUNER_MODEL, "model_hp_tuner.keras")
        self.PATH_HP_TUNER_BEST = os.path.join(self.DIR_HP_TUNER_MODEL, "model_best_hp_tuner.keras")
        self.PATH_HP_TUNER_CHECKPOINT = os.path.join(self.DIR_HP_TUNER_MODEL, "model_checkpoint_hp_tuner.keras")
        self.PATH_HP_TUNER_HYPERPARAMS = os.path.join(self.DIR_HP_TUNER_MODEL, "hyperparameters_hp_tuner.txt")
        self.PATH_HP_TUNER_HISTORY = os.path.join(self.DIR_HP_TUNER_HISTORY, "training_history_hp_tuner.json")
        self.PATH_HP_TUNER_LOG = os.path.join(self.DIR_HP_TUNER_LOGS, "log_hp_tuner.csv")

        # Create necessary directories
        self._create_directories()
        
        # Check if models exist and update skip training flags
        self._check_models_exist()

    def _create_directories(self):
        """Create all necessary directories for all strategies"""
        directories = [
            # Manual CNN directories
            self.DIR_MANUAL_CNN_MODEL,
            self.DIR_MANUAL_CNN_HISTORY,
            self.DIR_MANUAL_CNN_LOGS,
            self.DIR_MANUAL_CNN_PLOTS,
            
            # Auto Keras directories
            self.DIR_AUTO_KERAS_MODEL,
            self.DIR_AUTO_KERAS_HISTORY,
            self.DIR_AUTO_KERAS_LOGS,
            self.DIR_AUTO_KERAS_PLOTS,
            
            # HP Tuner directories
            self.DIR_HP_TUNER_MODEL,
            self.DIR_HP_TUNER_HISTORY,
            self.DIR_HP_TUNER_LOGS,
            self.DIR_HP_TUNER_PLOTS
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            
    def _check_models_exist(self):
        """Check if trained models already exist and set skip flags accordingly"""
        self.SKIP_TRAINING_MANUAL_CNN = os.path.exists(self.PATH_MANUAL_CNN_MODEL)
        self.SKIP_TRAINING_AUTO_KERAS = os.path.exists(self.PATH_AUTO_KERAS_MODEL)
        self.SKIP_TRAINING_HP_TUNER = os.path.exists(self.PATH_HP_TUNER_MODEL)

# Create config instance
config = Config()
