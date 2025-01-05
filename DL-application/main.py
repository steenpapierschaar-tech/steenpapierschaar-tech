from dataAugmentation import augmentData
from fileHandler import loadFiles, createOutputDir, createTimestampDir, createSubDir
from modelDesign import createModel
from dataLoader import load_data
import os
import json
import tensorflow as tf

if __name__ == "__main__":
    
    # File handling
    filelist = loadFiles("photoDataset")
    print(f"[INFO] Amount of images loaded: {len(filelist)}")
    
    # First split the data
    (train_images, train_labels), (val_images, val_labels) = load_data(filelist)
    
    # Then augment the training data if needed
    target_size = 500
    if len(train_images) < target_size:
        print(f"[INFO] Augmenting training data from {len(train_images)} to {target_size} images")
        train_images, train_labels = augmentData(train_images, train_labels, target_size)
    
    print(f"[INFO] Final training set size: {len(train_images)}")
    print(f"[INFO] Validation set size: {len(val_images)}")
    
    # Get number of classes from the final labels shape
    num_classes = train_labels.shape[1]
    
    # Design and compile CNN model
    model = createModel(num_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Create output directories
    outputDir = createOutputDir()
    timestampDir = createTimestampDir(outputDir)
    modelDir = createSubDir(timestampDir, "model")
    historyDir = createSubDir(timestampDir, "history")
    logDir = createSubDir(timestampDir,"logs")
    
    # Add %tensorboard
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logDir, histogram_freq=1)
    
    
    # Train model with basic config
    history = model.fit(
        train_images, train_labels,
        validation_data=(val_images, val_labels),
        epochs=20,
        batch_size=32,
        callbacks=[tensorboard_callback]
    )

    # Save the trained model
    model.save(os.path.join(modelDir, "trained_model.keras"))
    
    # Save training history
    with open(os.path.join(historyDir, "training_history.json"), "w") as f:
        json.dump(history.history, f)
    
    # TODO: Parameter tuning using grid search or random search

    # TODO: Performance analysis
    
    # TODO: Add layer visualization

    # TODO: Transfer learning using pre-trained models