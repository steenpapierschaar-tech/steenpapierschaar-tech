import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib 

import tensorflow as tf
from tensorflow import keras
from src.config import config

import keras
from src.config import config
from src.create_dataset import create_dataset

def monitor_predictions(model, dataset, name="Dataset"):
    """Monitor prediction distribution."""
    predictions = model.predict(dataset)
    pred_classes = np.argmax(predictions, axis=1)
    
    print(f"\n{name} Prediction Distribution:")
    for i, count in enumerate(np.bincount(pred_classes)):
        print(f"{config.CLASS_NAMES[i]}: {count} ({count/len(pred_classes)*100:.1f}%)")
    
    # Print confidence statistics
    probs = keras.activations.softmax(predictions).numpy()
    for i, class_name in enumerate(config.CLASS_NAMES):
        confidences = probs[:, i]
        print(f"\n{class_name} confidence:")
        print(f"Mean: {np.mean(confidences):.3f}")
        print(f"Std: {np.std(confidences):.3f}")
        

def build_other_model():
    """Build model architecture based on othermodel.py but adapted for our dataset"""
    model = tf.keras.models.Sequential([
        # Input layer with our target size
        keras.layers.InputLayer(input_shape=(config.TARGET_SIZE[0], config.TARGET_SIZE[1], 3)),
        
        # First conv block
        keras.layers.Conv2D(25, (5,5), 
                           activation='relu', 
                           strides=(1,1), 
                           padding='same',
                           kernel_regularizer=keras.regularizers.l2(0.01)),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(2, 2), padding='same'),
        
        # Second conv block
        keras.layers.Conv2D(50, (5,5), 
                           activation='relu', 
                           strides=(2,2), 
                           padding='same',
                           kernel_regularizer=keras.regularizers.l2(0.01)),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(2,2), padding='same'),
        
        # Third conv block
        keras.layers.Conv2D(70, (3,3), 
                           activation='relu', 
                           strides=(2,2), 
                           padding='same',
                           kernel_regularizer=keras.regularizers.l2(0.01)),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(2,2), padding='valid'),
        
        # Classification layers
        keras.layers.Flatten(),
        keras.layers.Dense(512, 
                         activation='relu',
                         kernel_regularizer=keras.regularizers.l2(0.01)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(3, activation='softmax')
    ])
    
    return model

def main():
    train_ds, val_ds = create_dataset()
    model = build_other_model()
   
    model.compile(loss='categorical_crossentropy',               
                optimizer='Adam', 
                metrics=['accuracy',
                keras.metrics.Precision(class_id=0, name='precision_rock'),
                keras.metrics.Precision(class_id=1, name='precision_paper'),
                keras.metrics.Precision(class_id=2, name='precision_scissors')
                ,]
    )
    model.summary()
    model.fit(
            train_ds,
            epochs=config.EPOCHS,
            validation_data=val_ds,
            # callbacks=callbacks,
            # class_weight=class_weights  # Add class weights
    )

    val_predictions = model.predict(val_ds)
    predicted_classes = np.argmax(val_predictions, axis=1)
    logits = model.predict(val_ds)
    probs = keras.activations.softmax(logits).numpy()
    print(probs[:5])

            # Print class distribution of predictions
    print("Prediction Distribution:", np.bincount(predicted_classes))
    print("Prediction Distribution:", np.bincount(predicted_classes))  # Check class imbalance

    monitor_predictions(model, train_ds, "Training")
    monitor_predictions(model, val_ds, "Validation")
    
if __name__ == "__main__":
    main()
