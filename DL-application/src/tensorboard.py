import subprocess
import os
from typing import Optional

class TensorboardLauncher:
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.process: Optional[subprocess.Popen] = None
        self.is_running = False

    def start_tensorboard(self):
        if not self.is_running:
            cmd = f"tensorboard --logdir={self.log_dir} --port=6006"
            self.process = subprocess.Popen(
                cmd.split(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            self.is_running = True
            print("TensorBoard started. Open http://localhost:6006 in your browser")

    def stop_tensorboard(self):
        if self.process and self.is_running:
            self.process.terminate()
            self.is_running = False