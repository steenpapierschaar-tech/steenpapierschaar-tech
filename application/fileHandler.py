import os
import glob
import datetime


def loadFiles():
    """Load all dataset image files"""

    # get the data path
    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__))
    )
    data_path = os.path.join(__location__, os.pardir, "photoDataset")

    # grab the list of images in our data directory
    print("[INFO] Loading images...")
    p = os.path.sep.join([data_path, "**", "*.png"])

    # Count amount of files
    count = 0
    for filename in glob.iglob(p, recursive=True):
        count += 1
    print("[INFO] Loaded", count, "images")

    # Building the file list
    fileList = [f for f in glob.iglob(p, recursive=True) if (os.path.isfile(f))]

    return fileList

def createOutputDir(subdir=None):
    """Create timestamped output directory
    If the output directory does not exist, it will be created.
    If a custom subdirectory is provided, it will be appended to the output directory.
    """

    # Check if output directory exists
    output_dir = os.path.join(os.getcwd(), "output")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    # Create timestamped subdirectory
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_subdir = os.path.join(output_dir, timestamp)
    if subdir:
        output_subdir = os.path.join(output_subdir, subdir)
    if not os.path.exists(output_subdir):
        os.makedirs(output_subdir, exist_ok=True)
    return output_subdir


if __name__ == "__main__":
    # Building the file list
    fileList = loadFiles()

    count = 0
    
    for filename in fileList:

        print("[INFO] Loading image: {}".format(filename))

        # Count amount of files
        count += 1

    print("[INFO] Amount of images loaded:", count)
