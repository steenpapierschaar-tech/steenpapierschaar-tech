import os
import glob
import datetime


def loadFiles(folderName):
    """Load all dataset image files"""

    # get the data path
    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__))
    )
    data_path = os.path.join(__location__, os.pardir, folderName)

    # grab the list of images in our data directory
    print(f"[INFO] Loading images for {data_path}...")
    p = os.path.sep.join([data_path, "**", "*.png"])

    # Building the file list
    fileList = [f for f in glob.iglob(p, recursive=True) if (os.path.isfile(f))]

    return fileList

def countFiles(folderName):
    """Count all dataset image files"""

    # get the data path
    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__))
    )
    data_path = os.path.join(__location__, os.pardir, folderName)

    # grab the list of images in our data directory
    print(f"[INFO] Counting images for {data_path}...")
    p = os.path.sep.join([data_path, "**", "*.png"])

    # Building the file list
    fileList = [f for f in glob.iglob(p, recursive=True) if (os.path.isfile(f))]

    return len(fileList)

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
    
    # Create timestamped subdirectory
    timestampDir = createTimestampDir(outputDir)
    
    # Create custom subdirectory
    subdir = "Test_Directory"
    subDir = createSubDir(timestampDir, subdir)
    
    # Create augmented directories
    baseDir = os.path.join(os.getcwd(), "photoDataset")
    augDirs = createAugmentedDirs(baseDir)
    
    # Example usage of augDirs
    for category, augDir in augDirs.items():
        print(f"[INFO] Augmented directory for {category}: {augDir}")
    
    # Building the file list
    fileList = loadFiles()

    count = 0
    
    for filename in fileList:

        print("[INFO] Loading image: {}".format(filename))

        # Count amount of files
        count += 1

    print("[INFO] Amount of images loaded:", count)
