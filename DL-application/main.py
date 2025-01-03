import cv2 as cv
from preprocessing import preprocessingPipeline as pipe
# from preprocessing import grayscale, rescaleImages, histogramEqualization
from dataAugmentation import augmentData
from fileHandler import loadFiles, countFiles
from modelDesign import createModel

if __name__ == "__main__":
    
    # File handling
    filelist = loadFiles("photoDataset")

    # Count the number of images
    datasetSize = len(filelist)
    print(f"[INFO] Amount of images loaded: {datasetSize}")

    # Augment data by generating new images
    target_size = 500
    if len(filelist) < target_size:
        filelist = augmentData(filelist, target_size)
    
    datasetSize = len(filelist)
    print(f"[INFO] Dataset size: {datasetSize}")
    
    # Design CNN model
    
    
    # Compile model
    
    # Parameter tuning
    
    # Train model
    
    # Transfer learning
    
    # Performance analysis
    
    # Model analysis
    
    # Visualize CNN model
