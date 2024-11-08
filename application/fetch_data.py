import os
import cv2 as cv
import numpy as np
import pandas as pd
from segment import *
from extract import *
from sklearn.utils import Bunch
from fileHandler import loadFiles
from sklearn.preprocessing import MinMaxScaler

def initDataMatrix():
    """
    Initialize an empty data matrix with space for both primary and extra features
    """
    featureNames = ['area', 'contourLength', 'ConvexHullLength', 'ConvexityDefects', 'compactness', 
                    'circularity', 'aspectRatio', 'extent'] + [f'HuMoment{i+1}' for i in range(7)]
    
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
    
    return features

def saveDatasetToCSV(dataset, dataset_path):
    """
    Save the dataset to a CSV file.
    """
    df = pd.DataFrame(dataset['data'], columns=dataset['feature_names'])
    df['target'] = dataset['target']
    
    # Scale the features before saving
    features = df.columns[:-1]
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])
    
    df.to_csv(dataset_path, index=False)
    print(f"[INFO] Dataset saved to {dataset_path}")

def loadDatasetFromCSV(dataset_path):
    """
    Load the dataset from a CSV file.
    """
    df = pd.read_csv(dataset_path)
    data = df.drop(columns=['target']).values
    target = df['target'].tolist()
    feature_names = df.columns[:-1].tolist()
    unique_targets = np.unique(target)
    final_dataset = Bunch(data=data, target=target, unique_targets=unique_targets, feature_names=feature_names)
    return final_dataset

def appendToDataset(dataset, features, label):
    """
    Append the extracted features and label to the dataset
    """
    combinedFeatures = np.hstack(features)

    # Append features to the dataset's data matrix
    dataset['data'] = np.vstack([dataset['data'], combinedFeatures])

    # Append the label to the dataset's target list
    dataset['target'].append(label)

    return dataset

def selectTopUniqueFeatures(trainX, feature_names, top_n=5):
    """ Select the top N most unique features based on average correlation """
    # Calculate the correlation matrix
    corr_matrix = np.corrcoef(trainX, rowvar=False)
    
    # Calculate the total (average) correlation for each feature
    total_correlation = np.mean(np.abs(corr_matrix), axis=1)
    
    # Combine feature names with their total correlation for easy sorting
    unique_features = sorted(zip(feature_names, total_correlation), key=lambda x: x[1])
    
    # Extract the top N unique feature names
    top_unique_features = [feature for feature, _ in unique_features[:top_n]]
    
    return top_unique_features

def filterFeatures(data, feature_names, selected_features):
    """ Filter the dataset to include only the selected features """
    selected_indices = [feature_names.index(feature) for feature in selected_features]
    filtered_data = data[:, selected_indices]
    return filtered_data

def datasetBuilder(file_list, output_dir, dataset_filename='gestures_dataset.csv'):
    """
    Build the dataset from the list of image files or load if exists.
    If the dataset already exists, it will be loaded from the file system.
    """
    dataset_path = os.path.join(output_dir, dataset_filename)
    
    # Check if the dataset already exists
    if os.path.exists(dataset_path):
        print(f"[INFO] Loading existing dataset from {dataset_path}")
        return loadDatasetFromCSV(dataset_path)

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

    # Save the final dataset to a CSV file for future use
    saveDatasetToCSV(final_dataset, dataset_path)
    
    return final_dataset

def loadGestures(file_list, output_dir, dataset_filename='gestures_dataset.csv'):
    """
    Load the gesture dataset if it exists, otherwise build a new one.
    """
    return datasetBuilder(file_list, output_dir, dataset_filename)

if __name__ == "__main__":
    """
    Main entry point for testing the functions.
    """
    # Load all dataset image files
    fileList = loadFiles()

    # Build the dataset from the image files
    gestures = datasetBuilder(fileList, 'output_directory')

    # Output details about the dataset
    print(dir(gestures))
    print(gestures.unique_targets)