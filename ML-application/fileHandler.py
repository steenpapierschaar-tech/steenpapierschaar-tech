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

def createOutputDir():
    """Create timestamped output directory"""

    # Check if output directory exists
    outputDir = os.path.join(os.getcwd(), "output")
    if not os.path.exists(outputDir):
        os.makedirs(outputDir, exist_ok=True)
    return outputDir

def createTimestampDir(outputDir):
    """Create timestamped output directory"""

    # Create timestamped subdirectory
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    timestampDir = os.path.join(outputDir, timestamp)
    if not os.path.exists(timestampDir):
        os.makedirs(timestampDir, exist_ok=True)
        
    return timestampDir

def createSubDir(timestampDir, subDir):
    """Create custom subdirectory in timestamped directory"""

    # Create timestamped subdirectory
    subDir = os.path.join(timestampDir, subDir)
    if not os.path.exists(subDir):
        os.makedirs(subDir, exist_ok=True)
        
    return subDir

if __name__ == "__main__":
    
    # Create output directory
    outputDir = createOutputDir()
    
    # Create timestamped subdirectory
    timestampDir = createTimestampDir(outputDir)
    
    # Create custom subdirectory
    subdir = "Test_Directory"
    subDir = createSubDir(timestampDir, subdir)
    
    # Building the file list
    fileList = loadFiles()

    count = 0
    
    for filename in fileList:

        print("[INFO] Loading image: {}".format(filename))

        # Count amount of files
        count += 1

    print("[INFO] Amount of images loaded:", count)
