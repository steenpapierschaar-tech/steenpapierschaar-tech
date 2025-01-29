import datetime
import os

class Config:
    """Centralized configuration for the DL application"""
    
    def __init__(self):
        # General
        self.VERBOSE = 1
        self.TIMESTAMP = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        
        # Data Loading
        self.IMAGE_DIMS = (240, 320)
        self.VALIDATION_SPLIT = 0.2
        self.RANDOM_STATE = 42
        
        # Augmentation
        self.RANDOM_BRIGHTNESS = 0.2
        self.RANDOM_CONTRAST = 0.2
        self.RANDOM_SATURATION = 0.2
        self.RANDOM_HUE = 0.2
        self.RANDOM_CROP = 0.2
        self.RANDOM_FLIP = "horizontal_and_vertical"
        self.RANDOM_SHARPNESS = 0.2
        self.RANDOM_SHEAR_X = 0.2
        self.RANDOM_SHEAR_Y = 0.2
        self.RANDOM_TRANSLATION = 0.2
        self.RANDOM_ZOOM = 0.2
        self.RANDOM_ROTATION = 0.2

        # Training
        self.EPOCHS = 50
        self.BATCH_SIZE = 32
        self.TARGET_AUGMENTATION_SIZE = 1500
        self.TARGET_SIZE = (96, 128)
        self.MAX_TRIALS = 1000
        
        # Input
        self.DATASET_ROOT_DIR = 'photoDataset'
        self.INPUT_DIRECTORY = 'input'

        # Tensorboard
        self.TENSORBOARD_PORT = 6006
        self.TENSORBOARD_UPDATE_FREQ = 'epoch'  # or 'batch' for more frequent updates
        self.TENSORBOARD_HISTOGRAM_FREQ = 1

        # Define directories
        self.OUTPUT_DIRECTORY       = 'output'
        self.OUTPUT_DIR             = os.path.join(self.OUTPUT_DIRECTORY, self.TIMESTAMP)
        self.MODEL_DIR              = os.path.join(self.OUTPUT_DIR, 'model')
        self.HISTORY_DIR            = os.path.join(self.OUTPUT_DIR, 'history')
        self.LOGS_DIR               = os.path.join(self.OUTPUT_DIR, 'logs')

        # Define resulting paths (directory + filename)
        self.TRAIN_MODEL_PATH       = os.path.join(self.MODEL_DIR, 'model.keras')
        self.MODEL_BEST_PATH        = os.path.join(self.MODEL_DIR, 'model_best.keras')
        self.MODEL_CHECKPOINT_PATH  = os.path.join(self.MODEL_DIR, 'model_checkpoint.keras')
        self.HYPERPARAMS_PATH       = os.path.join(self.MODEL_DIR, 'hyperparameters.txt')
        self.TRAIN_HISTORY_PATH     = os.path.join(self.HISTORY_DIR, 'training_history.json')
        self.CSV_LOG_PATH           = os.path.join(self.HISTORY_DIR, 'log.csv')
        
        # Create all necessary directories
        self._create_directories()

    def _create_directories(self):
        """Create all necessary directories for the application"""
        directories = [
            self.OUTPUT_DIR,
            self.MODEL_DIR,
            self.HISTORY_DIR,
            self.LOGS_DIR,
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

# Create config instance
config = Config()
