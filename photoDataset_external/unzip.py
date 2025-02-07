"""
unzip Archive.zip, which contains images from an external dataset.
https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors

Required Libraries:
- zipfile
- os
- pathlib

pip install zipfile os pathlib
"""

import zipfile
import os
import pathlib

# Get the path to the directory containing this script
current_dir = pathlib.Path(__file__).parent.resolve()

# Path to the ZIP archive containing the external dataset
archive_path = current_dir / "Archive.zip"

# Extract zip to the current directory
with zipfile.ZipFile(archive_path, "r") as zip_ref:
    zip_ref.extractall(current_dir)