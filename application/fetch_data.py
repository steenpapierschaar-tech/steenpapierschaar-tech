import os
import cv2 as cv
import numpy as np
import pickle
from segment import *
from extract import *
from sklearn.utils import Bunch
from datasetLoader import loadFiles

def initDataMatrix():
    """
    Initialize an empty data matrix with space for both primary and extra features
    """
    featureNames = ['area', 'contourLength', 'ConvexHullLength', 'ConvexityDefects', 'compactness', 'circularity', 'aspectRatio', 'extent'] + [f'HuMoment{i+1}' for i in range(7)]
    
    # Create an empty data matrix with correct number of feature columns
    data = np.empty((0, len(featureNames)), float)  # Empty array with feature length
    
    return data, featureNames

def analyzeImage(image):
    """
    Analyze image and return features and label
    """
    # Get mask from image
    imageMask = prepareImage(image)
    
    # Get the largest contour
    contour = getLargestContour(imageMask)
    
    # Get features from contour
    features = getFeatures(contour)
    
    # Compute moments and Hu Moments
    moments = cv.moments(contour)  # Compute spatial moments
    huMoments = cv.HuMoments(moments).flatten()  # Compute Hu moments and flatten the array
    
    # Normalize Hu moments for numerical stability
    huMoments = -np.sign(huMoments) * np.log10(np.abs(huMoments) + 1e-12)  # Log transformation for numerical stability
    
    # Combine all features
    combinedFeatures = np.hstack((features, huMoments))
    
    return combinedFeatures

def appendToDataset(dataset, features, label):
    """
    Append the extracted features and label to the dataset
    """
    # Combine primary and extra features
    combinedFeatures = np.hstack(features)

    # Append features to the dataset's data matrix
    dataset['data'] = np.vstack([dataset['data'], combinedFeatures])

    # Append the label to the dataset's target list
    dataset['target'].append(label)

    return dataset
def datasetBuilder(file_list, dataset_path='gestures_dataset.pkl'):
    """
    Build the dataset from the list of image files or load if exists.
    If the dataset already exists, it will be loaded from the file system.
    """
    # Check if the dataset already exists
    if os.path.exists(dataset_path):
        print(f"[INFO] Loading existing dataset from {dataset_path}")
        with open(dataset_path, 'rb') as f:
            final_dataset = pickle.load(f)
        return final_dataset

    print("[INFO] Dataset not found, building a new one...")

    # Initialize the data matrix and feature names
    dataMatrix, featureNames = initDataMatrix()

    # Create the dataset structure
    dataset = {
        'data': dataMatrix,
        'target': [],
        'feature_names': featureNames
    }

    # Process each image in the file list
    for filename in file_list:
        print(f"[INFO] Processing image: {filename}")
        
        # Load the image
        image = cv.imread(filename)

        # Analyze the image to extract features
        features = analyzeImage(image)
        
        # Extract the label from the directory (if needed)
        label = filename.split(os.path.sep)[-2]
        
        # Append the features and label to the dataset
        dataset = appendToDataset(dataset, features, label)

    # Get unique target labels
    unique_targets = np.unique(dataset['target'])
    print(f"[INFO] Targets found: {unique_targets}")

    # Create a final dataset using Bunch
    final_dataset = Bunch(data=dataset['data'],
                          target=dataset['target'],
                          unique_targets=unique_targets,
                          feature_names=dataset['feature_names'])

    # Save the final dataset to a file for future use
    with open(dataset_path, 'wb') as f:
        print(f"[INFO] Saving dataset to {dataset_path}")
        pickle.dump(final_dataset, f)
    
    return final_dataset

def loadGestures(file_list, dataset_path='gestures_dataset.pkl'):
    """
    Load the gesture dataset if it exists, otherwise build a new one.
    """
    return datasetBuilder(file_list, dataset_path)

# def datasetBuilder(file_list):
#     """
#     Build the dataset from the list of image files
#     """
#     # Initialize the data matrix and feature names
#     dataMatrix, featureNames = initDataMatrix()
    
#     # Create the dataset structure
#     dataset = {
#         'data': dataMatrix,
#         'target': [],
#         'feature_names': featureNames
#     }

#     # Process each image in the file list
#     for filename in file_list:
#         print(f"[INFO] Processing image: {filename}")
        
#         # Load the image
#         image = cv.imread(filename)

#         # Analyze the image to extract features
#         features = analyzeImage(image)
        
#         # Extract the label from the directory (if needed)
#         label = filename.split(os.path.sep)[-2]
        
#         # Append the features and label to the dataset
#         dataset = appendToDataset(dataset, features, label)


#     # Get unique target labels
#     unique_targets = np.unique(dataset['target'])
#     print(f"[INFO] Targets found: {unique_targets}")

#     # Create a final dataset using Bunch
#     final_dataset = Bunch(data=dataset['data'],
#                           target=dataset['target'],
#                           unique_targets=unique_targets,
#                           feature_names=dataset['feature_names'])
    
#     return final_dataset 

if __name__ == "__main__":
    """
    Main entry point for testing the functions.
    """
    # Load all dataset image files
    fileList = loadFiles()

    # Build the dataset from the image files
    gestures = datasetBuilder(fileList)

    # Output details about the dataset
    print(dir(gestures))
    print(gestures.unique_targets)
