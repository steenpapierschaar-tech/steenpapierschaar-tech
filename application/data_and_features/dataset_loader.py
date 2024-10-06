import os
import glob

def load_files():
    """ Load all dataset image files """

    # get the data path
    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__)))
    data_path = os.path.join(__location__, os.pardir, os.pardir, 'photo_dataset')
    
    # grab the list of images in our data directory
    print("[INFO] loading images...")
    p = os.path.sep.join([data_path, '**', '*.png'])
    
    # Count amount of files
    count = 0
    for filename in glob.iglob(p, recursive=True):
        count += 1
    print("Amount of files: ", count)
    
    # Building the file list
    file_list = [f for f in glob.iglob(p, recursive=True) if (os.path.isfile(f))]
    
    return file_list
