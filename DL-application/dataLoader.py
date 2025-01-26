import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from collections import Counter
from config import config

def prepareFiles(filelist):
    """
    Prepare files for training
    """
    print(f"[INFO] Prepare files for training: Resize and convert to R without BG")
    for i, filepath in enumerate(filelist):
        image = cv2.imread(filepath)
        height, width = image.shape[:2]
        
        # Resize images using configured dimensions
        resized_image = cv2.resize(image, config.IMAGE_DIMS)
        
        # # Convert to RGB and set blue and green channels to 0 --> negatief resultaat!
        # resized_image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR)
        # resized_image[:, :, 0] = 0  # Set the blue channel to 0
        # resized_image[:, :, 1] = 0  # Set the green channel to 0
        
        # Save the preparet image
        cv2.imwrite(filepath, resized_image)
        
    #     cv2.imshow("Resized image", resized_image)
    
    return filelist

def load_data(filelist, validation_split=config.VALIDATION_SPLIT):
    """
    Load images from filelist and split into training and validation sets
    """
    
    # Initialize lists for images and labels
    images = []
    labels = []
    
    # Create a dictionary to map class names to numeric labels
    classes = sorted(set(os.path.basename(os.path.dirname(f)) for f in filelist))
    class_to_label = {class_name: idx for idx, class_name in enumerate(classes)}
    
    # Create a counter for labels
    label_counter = Counter()
    
    print("\n[INFO] Found classes:", ", ".join(classes))
    
    print("\n[INFO] Loading and preprocessing images...")
    for filepath in filelist:
        # Read and preprocess image
        image = cv2.imread(filepath)
        # image = cv2.resize(image, (64, 64))  # Resize to match model input
        image = image.astype("float32") / 255.0  # Normalize pixel values
        
        # Get class name from directory name
        class_name = os.path.basename(os.path.dirname(filepath))
        label = class_to_label[class_name]
        
        images.append(image)
        labels.append(label)
        
        # Count labels
        label_counter[class_name] += 1
    
    # Print label distribution
    print("\n[INFO] Label distribution in dataset:")
    for class_name, count in label_counter.items():
        print(f"    {class_name}: {count} images")
    
    # Convert to numpy arrays
    X = np.array(images)
    y = np.array(labels)
    
    # Convert labels to one-hot encoding
    y = to_categorical(y, len(classes))
    
    # Split the data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=validation_split, random_state=config.RANDOM_STATE
    )
    
    print(f"[INFO] Training set size: {len(X_train)}")
    print(f"[INFO] Validation set size: {len(X_val)}")
    
    return (X_train, y_train), (X_val, y_val)

def main():
    # Define the path to the dataset
    dataset_path = config.DATASET_PATH
    
    # Get a list of all image files in the dataset
    filelist = []
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".jpg"):
                filelist.append(os.path.join(root, file))
    
    # Prepare files for training
    filelist = prepareFiles(filelist)
    
    # Load the data
    (X_train, y_train), (X_val, y_val) = load_data(filelist)
    
    return (X_train, y_train), (X_val, y_val)
