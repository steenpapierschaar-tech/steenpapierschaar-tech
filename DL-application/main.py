from archive.dataAugmentation import augmentData
from archive.fileHandler import loadFiles, createOutputDir
from archive.modelDesign import createModel
from archive.dataLoader import load_data, prepareFiles
from src.config import config
import os
import json
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def DeepLearning():
     # File handling
    filelist = loadFiles()
    print(f"[INFO] Amount of images loaded: {len(filelist)}")
    
    #prepare files for training
    prepareFiles(filelist)
    
    # First split the data
    (train_images, train_labels), (val_images, val_labels) = load_data(filelist)
    
    # Then augment the training data if needed
    if len(train_images) < config.TARGET_AUGMENTATION_SIZE:
        print(f"[INFO] Augmenting training data from {len(train_images)} to {config.TARGET_AUGMENTATION_SIZE} images")
        train_images, train_labels = augmentData(train_images, train_labels, config.TARGET_AUGMENTATION_SIZE)
    
    print(f"[INFO] Final training set size: {len(train_images)}")
    print(f"[INFO] Validation set size: {len(val_images)}")
    
    # Get number of classes from the final labels shape
    num_classes = train_labels.shape[1]
    
    # Design and compile CNN model      
    #model = createModel(num_classes)                      
    #model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])        
    model = Sequential([
        Conv2D(16, (3, 3), activation='relu', input_shape=(24, 32, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(3, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    
    # Use config paths directly
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=config.LOGS_PATH, 
        histogram_freq=config.TENSORBOARD_HISTOGRAM_FREQ
    )    
    
    # Train model with basic config
    history = model.fit(
        train_images, train_labels,
        validation_data=(val_images, val_labels),
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        callbacks=[tensorboard_callback]
    )
    # Save the trained model
    model.save(config.TRAINED_MODEL_PATH)
    
    # Save training history
    with open(config.TRAINING_HISTORY_PATH, "w") as f:
        json.dump(history.history, f)
    
    # TODO: Parameter tuning using grid search or random search

    # TODO: Performance analysis
    
    # TODO: Add layer visualization
    

def TranferLearning():
     # TODO: Transfer learning using pre-trained models --> p.375 Hands-on machinel learning with scikit-learn, keras & tensorflow
     #Transfer learning does not work well with small dense networks, presumably because small networks learn a few patterns, and dense networks learn very specific patterns, 
     # wich are unlikely to be usefull in other tasks. Transferlearning works best with deep convolutional neural networks, which tend to learn feature detectors that are much general. 
     
    TFmodelDir = "TransferLearnModel/"
    model_tf = tf.keras.models.load_model(os.path.join(TFmodelDir, "rock_paper_scissors_model.h5"))
    model_tf.summary()

if __name__ == "__main__":    
    DeepLearning()   
    #TranferLearning()