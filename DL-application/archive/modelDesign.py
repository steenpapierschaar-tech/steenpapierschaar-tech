import tensorflow as tf
from tensorflow.keras import layers, models, Input
import matplotlib.pyplot as plt
import os

def createModel(num_classes):
    # Create a CNN model optimized for relatively small images
    # The model uses dropout to prevent overfitting
    print(f"[INFO] Creating CNN model with {num_classes} classes")
    
    # Create model with Input layer - flexible input size for different image dimensions
    inputs = Input(shape=(None, None, 3))  # RGB images with variable size
    # First conv block - basic feature detection
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    # Second conv block - more complex features
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    # Third conv block - highest level features
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x)  # Reduces parameters compared to Flatten
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)  # Prevents overfitting
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)

    return model