import os
import glob


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


if __name__ == "__main__":
    # Building the file list
    fileList = loadFiles()

    count = 0
    
    for filename in fileList:

        print("[INFO] Loading image: {}".format(filename))

        # Count amount of files
        count += 1

    print("[INFO] Amount of images loaded:", count)
