import os
import glob
import cv2 as cv
import numpy as np
from segment import *
from extract import *
from sklearn.utils import Bunch
from datasetLoader import loadFiles

def initDataMatrix():
    """
    Initialize data matrix with correct number of features
    """
    featureNames = ['area', 'contourLength', 'ConvexHullLength', 'ConvexityDefects','compactness']
    
    featureExtraNames = ['area', 'perimeter', 'aspect_ratio', 'extent']
    
    data = np.empty((0,len(featureNames)), len(featureExtraNames),float)
    
    return data

def analyzeImage(image):
    """
    Analyze image and return features
    """
    
    # get mask from image
    imageMask = prepareImage(image)
    
    # get largest contour
    contour = getLargestContour(imageMask)
    
    # get features from contour
    features = getFeatures(contour)
    
    # get simple features as extra
    featuresExtra = getSimpleContourFeatures(contour)
    
    # get the label of the image
    label = image.split(os.path.sep)[-2]
    
    return features, featuresExtra, label
    
def datasetBuilder(features, featuresExtra, label):
    """
    Build dataset from features
    """
    
    # get unique labels
    target = []
    
    unique_labels = np.unique(label)
    
    # extract label from folder name and stor
    label = filename.split(os.path.sep)[-2]
    target.append(label)

    # append features to data matrix
    data = np.append(data, np.array([features]), axis=0)

    unique_targets = np.unique(target)
    print("[INFO] targets found: {}".format(unique_targets))

    dataset = Bunch(data = data,
                    target = target,
                    unique_targets = unique_targets,
                    featureNames = featureNames)

    return dataset


if __name__ == "__main__":
    """
    Test functions in this file
    """
    
    # load all dataset image files
    fileList = loadFiles()
    dataMatrix = initDataMatrix()
    
    for filename in fileList:
        
        print("[INFO] processing image: {}".format(filename))
        