import kagglehub
import os

# Download latest version
path = kagglehub.dataset_download("drgfreeman/rockpaperscissors")
# Copy directory structure and files to current directory
dest_base = os.getcwd()
for root, dirs, files in os.walk(path):
    # Get the relative path from the source base
    rel_path = os.path.relpath(root, path)
    # Create corresponding directory in destination
    dest_dir = os.path.join(dest_base, rel_path)
    os.makedirs(dest_dir, exist_ok=True)
    # Move files maintaining structure
    for file in files:
        src = os.path.join(root, file)
        dst = os.path.join(dest_dir, file)
        os.rename(src, dst)
path = dest_base
print("Path to dataset files:", path)