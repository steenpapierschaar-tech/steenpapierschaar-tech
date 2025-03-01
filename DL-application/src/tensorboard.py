import subprocess
import os
from typing import Optional, Dict
from src.config import config

class TensorboardLauncher:
    def __init__(self):
        """Initialize with output directory as parent log directory"""
        self.process: Optional[subprocess.Popen] = None
        self.is_running = False
        
        # Use the parent output directory for all logs
        self.log_dir = config.OUTPUT_DIR
        
        # Define logdir mappings for each strategy
        self.strategy_logs = {
            'manual_cnn': config.DIR_MANUAL_CNN_LOGS,
            'auto_keras': config.DIR_AUTO_KERAS_LOGS,
            'hp_tuner': config.DIR_HP_TUNER_LOGS
        }

    def start_tensorboard(self):
        """Start tensorboard with all strategy logs visible"""
        if not self.is_running:
            # Create a logdir string that includes all strategy logs
            logdir_args = ",".join([
                f"{name}:{path}" 
                for name, path in self.strategy_logs.items()
            ])
            
            cmd = f"tensorboard --logdir={logdir_args} --reload_multifile=true --port=6006"
            self.process = subprocess.Popen(
                cmd.split(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            self.is_running = True
            print("TensorBoard started with all strategy logs. Open http://localhost:6006 in your browser")

    def stop_tensorboard(self):
        """Stop the tensorboard process"""
        if self.process and self.is_running:
            self.process.terminate()
            self.is_running = False
