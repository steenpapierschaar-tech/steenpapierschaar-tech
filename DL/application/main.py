import cv2 as cv
from preprocessing import preprocessingPipeline as pipe
from preprocessing import grayscale, rescaleImages, histogramEqualization
from dataAugmentation import dataAugmentation
from fileHandler import loadFiles
from modelDesign import createModel

def main():
    imageDataSet = loadFiles("photoDataset")
    pipe() 

if __name__ == "__main__":
    
    # File handling
    filelist = loadFiles("photoDataset")
    
    for file in filelist:
        print(file)
        
        # Preprocessing
        
        img = cv.imread(file)
        
        img = grayscale(img)
        
        img = histogramEqualization(img)
        
        img = rescaleImages(img)

    # Data augmentation
    
    ### If dataset < 5000 images, use data augmentation
    if len(filelist) < 5000:
        filelist = dataAugmentation(filelist)
    
    # Design CNN model
    
    
    # Compile model
    
    # Parameter tuning
    
    # Train model
    
    # Transfer learning
    
    # Performance analysis
    
    # Model analysis
    
    # Visualize CNN model
    
    main()