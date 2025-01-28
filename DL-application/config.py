import datetime
import os

class Config:
    """Centralized configuration for the DL application"""
    
    def __init__(self):
        # General
        self.VERBOSE = 1
        self.CURRENTDATETIME = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        
        # Data Loading
        self.IMAGE_DIMS = (320, 240)
        self.VALIDATION_SPLIT = 0.2
        self.RANDOM_STATE = 42
        
        # Augmentation
        self.ROTATION_RANGE = 40
        self.SHIFT_RANGE = 0.2
        self.SHEAR_RANGE = 0.2
        self.ZOOM_RANGE = 0.2
        self.HORIZONTAL_FLIP = True
        self.FILL_MODE = 'nearest'
        
        # Training
        self.EPOCHS = 50
        self.BATCH_SIZE = 32
        self.TARGET_AUGMENTATION_SIZE = 1500
        self.TARGET_SIZE = (64, 48)
        
        # Input
        self.DATASET_ROOT_DIR = 'photoDataset'
        self.INPUT_DIRECTORY = 'input'
        
        # Output
        self.OUTPUT_DIRECTORY = 'output'
        self.MODEL_DIR = 'model'
        self.HISTORY_DIR = 'history'
        self.LOGS_DIR = 'logs'
        self.TENSORBOARD_HISTOGRAM_FREQ = 1
        self.TRAINED_MODEL_NAME = 'trained_model'
        self.TRAINING_HISTORY_FILE = 'training_history.json'
        
        # Files
        self.BEST_HP_FILENAME = 'best_hyperparameters.txt'
        self.BEST_MODEL_FILENAME = 'model_best.keras'
        self.CHECKPOINT_MODEL_FILENAME = 'model_checkpoint.keras'
        
        # define paths
        self.OUTPUT_PATH = os.path.join(self.OUTPUT_DIRECTORY)
        self.MODEL_PATH = os.path.join(self.OUTPUT_DIRECTORY, self.MODEL_DIR)
        self.HISTORY_PATH = os.path.join(self.OUTPUT_DIRECTORY, self.HISTORY_DIR)
        self.LOGS_PATH = os.path.join(self.OUTPUT_DIRECTORY, self.LOGS_DIR)
        self.TRAINED_MODEL_PATH = os.path.join(self.MODEL_PATH, f"{self.TRAINED_MODEL_NAME}.h5")
        self.TRAINING_HISTORY_PATH = os.path.join(self.HISTORY_PATH, self.TRAINING_HISTORY_FILE)
        self.DATASET_PATH = os.path.join(self.DATASET_ROOT_DIR)
        self.INPUT_PATH = os.path.join(self.INPUT_DIRECTORY)
        self.BEST_HP_PATH = os.path.join(self.OUTPUT_PATH, self.BEST_HP_FILENAME)
        self.BEST_MODEL_PATH = os.path.join(self.MODEL_PATH, self.BEST_MODEL_FILENAME)
        self.CHECKPOINT_MODEL_PATH = os.path.join(self.MODEL_PATH, self.CHECKPOINT_MODEL_FILENAME)

# Create config instance
config = Config()
