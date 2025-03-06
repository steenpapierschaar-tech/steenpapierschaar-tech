"""
Rock Paper Scissors Image Classifier using Deep Learning

This application implements a Convolutional Neural Network (CNN) to classify images
into three categories: rock, paper, or scissors. The model uses multiple
convolutional layers for feature extraction followed by dense layers for classification.

Key Components:
- CNN Architecture: Extracts visual features from images
- Data Augmentation: Improves model generalization by creating variations of training images
- Regularization: Prevents overfitting using techniques like dropout and L2 regularization
- Performance Monitoring: Various plots and metrics to analyze model behavior
"""

import keras
from src.config import config
from src.create_dataset import create_dataset
from src.tensorboard import TensorboardLauncher
from src.training_callbacks import ringring_callbackplease
import numpy as np


def show_classes(train_ds):
    """
    Display the class names from the training dataset.
    """
    # Assuming train_ds is a TensorFlow Dataset object with (image, label) pairs
    class_names = []
    for _, labels in train_ds:
        class_names.extend(np.argmax(labels, axis=1))  # Convert one-hot labels to class indices
    class_names = np.unique(class_names)  # Get unique class indices
    print("Class Names:", class_names)


def compute_class_weights(dataset):
    """Compute balanced class weights."""
    labels = []
    for _, y in dataset:
        labels.extend(np.argmax(y, axis=1))
    
    from sklearn.utils.class_weight import compute_class_weight
    weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels
    )
    return dict(enumerate(weights))


def build_model():
    inputs = keras.layers.Input(shape=(config.TARGET_SIZE[0], config.TARGET_SIZE[1], 3))
    
    # Reduce initial filter size and add regularization
    x = keras.layers.Conv2D(
        64,  # Reduced from 224
        (3, 3),  # Smaller kernel
        activation="relu",
        padding="same",  # Changed to same padding
        kernel_regularizer=keras.regularizers.l2(0.01)
    )(inputs)
    
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D()(x)
    
    # Add more balanced layers
    for filters in [128, 256]:
        x = keras.layers.Conv2D(
            filters, 
            (3, 3),
            activation="relu",
            padding="same",
            kernel_regularizer=keras.regularizers.l2(0.1)
        )(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPooling2D()(x)
    
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dropout(0.6)(x)  # Increased dropout
    outputs = keras.layers.Dense(3, activation="softmax")(x)  # Re-add softmax
    
    return keras.Model(inputs=inputs, outputs=outputs)

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

def main():
    """
    Main training pipeline for the rock-paper-scissors classifier.

    Steps:
    1. Load and prepare training/validation datasets
    2. Configure model training settings
    3. Train the model with early stopping (if SKIP_TRAINING is False)
    4. Generate performance visualization plots
    """
    # Load and prepare datasets with augmentation
    train_ds, val_ds = create_dataset()
    
    show_classes(train_ds)
    tensorboard = TensorboardLauncher()
    tensorboard.start_tensorboard()

    # If model exists and SKIP_TRAINING is True, load it
    if config.SKIP_TRAINING_MANUAL_CNN:
        print("Loading existing model...")
        model = keras.models.load_model(config.PATH_MANUAL_CNN_MODEL)
    else:
        print("Building and training new model...")
        model = build_model()
    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=0.001,
            weight_decay=0.0001
        ),
        loss=keras.losses.CategoricalCrossentropy(
            label_smoothing=0.1,
            from_logits=False  # Since we're using softmax
        ),
        metrics=[
            "accuracy",
            keras.metrics.Precision(class_id=0, name='precision_rock'),
            keras.metrics.Precision(class_id=1, name='precision_paper'),
            keras.metrics.Precision(class_id=2, name='precision_scissors'),
        #     keras.metrics.Recall(class_id=0, name='recall_rock'),
        #     keras.metrics.Recall(class_id=1, name='recall_paper'),
        #     keras.metrics.Recall(class_id=2, name='recall_scissors'),
        ]
    )

    # Display model architecture summary
    model.summary()

    # Early Stopping Callback
    # Stops training if the model achieves high accuracy and low loss
    # This prevents overfitting and saves training time
    if not config.SKIP_TRAINING_MANUAL_CNN:
        callbacks = ringring_callbackplease(
            logs_dir=tensorboard.log_dir,
            csv_log_path=config.PATH_MANUAL_CNN_LOG,
            use_model_checkpoint=True,
            use_early_stopping=False,
            use_csv_logger=True,
            use_timeout=False,
            use_custom_callback=False,
            use_tensorboard=True
        )
        y_train_labels = np.concatenate([y for x, y in train_ds], axis=0)  # Get training labels
        y_train_labels = np.argmax(y_train_labels, axis=1)  # Convert one-hot to class indices

        # Compute class weights
        class_weights = compute_class_weights(train_ds)
        print("Class weights:", class_weights)
        
        model.fit(
            train_ds,
            epochs=config.EPOCHS,
            validation_data=val_ds,
            callbacks=callbacks,
            class_weight=class_weights  # Add class weights
        )



# Get validation predictions
    val_predictions = model.predict(val_ds)
    predicted_classes = np.argmax(val_predictions, axis=1)
    logits = model.predict(val_ds)
    # probs = keras.activations.softmax(logits).numpy()
    print(probs[:5])

    # Print class distribution of predictions
    print("Prediction Distribution:", np.bincount(predicted_classes))
    print("Prediction Distribution:", np.bincount(predicted_classes))  # Check class imbalance

    monitor_predictions(model, train_ds, "Training")
    monitor_predictions(model, val_ds, "Validation")

if __name__ == "__main__":
    main()
