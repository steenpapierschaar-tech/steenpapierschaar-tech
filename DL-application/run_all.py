"""
This script runs all three model implementations sequentially:
1. Manual Model (application.py)
2. Hyperparameter Tuner Model (hp_tuner.py)
3. AutoKeras Model (automodel.py)

The script uses config.SKIP_TRAINING flag to determine whether to run
training or just evaluation for each model.
"""
import sys
import shutil
import argparse
from pathlib import Path

# Ensure the script can import from the src directory
sys.path.append(str(Path(__file__).parent))

# Import the main functions from each script
from application import main as run_manual_model
from hp_tuner import main as run_hp_tuner
from automodel import main as run_automodel
from src.config import config

def clear_output_directory():
    """Clear the output directory if it exists and reset training flags"""
    if Path(config.OUTPUT_DIR).exists():
        shutil.rmtree(config.OUTPUT_DIR)
        print(f"Cleared output directory: {config.OUTPUT_DIR}")
    
    # Reset training flags since we're starting fresh
    config.SKIP_TRAINING_MANUAL_CNN = True
    config.SKIP_TRAINING_AUTO_KERAS = True
    config.SKIP_TRAINING_HP_TUNER = False
    
    # Recreate directory structure
    config._create_directories()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run all model implementations sequentially')
    parser.add_argument('--force', action='store_true', 
                       help='Clear output directory before starting')
    args = parser.parse_args()

    # Clear output directory if --force is given
    if args.force:
        clear_output_directory()

    # Run Manual Model
    print("\n" + "="*50)
    print("Running Manual Model (application.py)")
    print("="*50)
    run_manual_model()

    # Run Hyperparameter Tuner
    print("\n" + "="*50)
    print("Running Hyperparameter Tuner (hp_tuner.py)")
    print("="*50)
    run_hp_tuner()

    # Run AutoKeras Model
    print("\n" + "="*50)
    print("Running AutoKeras Model (automodel.py)")
    print("="*50)
    run_automodel()

    print("\n" + "="*50)
    print("All models have been processed successfully!")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
