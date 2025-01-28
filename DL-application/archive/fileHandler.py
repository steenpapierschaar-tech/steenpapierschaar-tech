import os
import glob
import datetime
from config import config

# Get working directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)  # Go up one level
output_dir = os.path.join(project_root, config.OUTPUT_DIRECTORY)

def loadFiles():
    """Load all dataset image files"""

    # get the data path relative to script location
    data_path = os.path.join(project_root, config.DATASET_ROOT_DIR)

    # Show the script location
    print(f"[INFO] Script location: {script_dir}")
    
    # Show the project root
    print(f"[INFO] Project root: {project_root}")

    # grab the list of images in our data directory
    print(f"[INFO] Loading images from {data_path}...")
    p = os.path.sep.join([data_path, "**", "*.png"])

    # Building the file list
    fileList = [f for f in glob.iglob(p, recursive=True) if (os.path.isfile(f))]

    return fileList

def countFiles():
    """Count all dataset image files"""

    # get the data path relative to script location
    data_path = os.path.join(project_root, config.DATASET_ROOT_DIR)

    # grab the list of images in our data directory
    print(f"[INFO] Counting images from {data_path}...")
    p = os.path.sep.join([data_path, "*.png"])

    # Building the file list
    fileList = [f for f in glob.iglob(p) if (os.path.isfile(f))]

    return len(fileList)

def createOutputDir():
    """Create output directory structure"""
    
    # Create base output directory relative to project root
    outputDir = os.path.join(project_root, config.OUTPUT_DIRECTORY)
    if not os.path.exists(outputDir):
        os.makedirs(outputDir, exist_ok=True)
        
    # Create timestamped subdirectory
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    timestampDir = os.path.join(outputDir, timestamp)
    if not os.path.exists(timestampDir):
        os.makedirs(timestampDir, exist_ok=True)
        
    # Create required subdirectories
    os.makedirs(os.path.join(timestampDir, config.MODEL_DIR), exist_ok=True)
    os.makedirs(os.path.join(timestampDir, config.HISTORY_DIR), exist_ok=True)
    os.makedirs(os.path.join(timestampDir, config.LOGS_DIR), exist_ok=True)
        
    return timestampDir

def createAugmentedDirs(baseDir):
    """Create augmented directories for rock, paper, and scissors"""
    categories = ["rock", "paper", "scissors"]
    augDirs = {}
    for category in categories:
        augDir = os.path.join(baseDir, category)
        if not os.path.exists(augDir):
            os.makedirs(augDir, exist_ok=True)
        augDirs[category] = augDir
    return augDirs

if __name__ == "__main__":
    # Create output directory
    outputDir = createOutputDir()
    
    # Example usage of output directories
    print(f"[INFO] Created output directory structure at: {outputDir}")
    
    # Building the file list
    fileList = loadFiles()
    print(f"[INFO] Loaded {len(fileList)} files")
