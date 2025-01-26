class Config:
    """Centralized configuration for the DL application"""
    
    # Data Loading
    IMAGE_DIMS = (32, 24)
    VALIDATION_SPLIT = 0.2
    RANDOM_STATE = 42
    
    # Augmentation
    ROTATION_RANGE = 40
    SHIFT_RANGE = 0.2
    SHEAR_RANGE = 0.2
    ZOOM_RANGE = 0.2
    HORIZONTAL_FLIP = True
    FILL_MODE = 'nearest'
    
    # Training
    EPOCHS = 50
    BATCH_SIZE = 32
    TARGET_AUGMENTATION_SIZE = 1500
    TARGET_SIZE = (150, 150)
    
    # Input
    DATASET_ROOT_DIR = 'photoDataset'
    INPUT_DIRECTORY = 'input'
    
    # Output
    OUTPUT_DIRECTORY = 'output'
    MODEL_DIR = 'model'
    HISTORY_DIR = 'history'
    LOGS_DIR = 'logs'
    TENSORBOARD_HISTOGRAM_FREQ = 1
    TRAINED_MODEL_NAME = 'trained_model.keras'
    TRAINING_HISTORY_FILE = 'training_history.json'

# Create config instance
config = Config()
