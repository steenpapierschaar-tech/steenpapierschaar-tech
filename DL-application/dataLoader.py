import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from collections import Counter

def load_data(filelist, validation_split=0.2):
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
        X, y, test_size=validation_split, random_state=42
    )
    
    print(f"[INFO] Training set size: {len(X_train)}")
    print(f"[INFO] Validation set size: {len(X_val)}")
    
    return (X_train, y_train), (X_val, y_val)
