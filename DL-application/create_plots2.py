import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import keras
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from src.config import config


def dataset_from_dir(directory):
    print("\n=== Dataset Loading Debug Info ===")
    print(f"Loading from directory: {directory}")
    print(f"Config class names: {config.CLASS_NAMES}")
    
    # Check actual directory contents
    class_dirs = sorted([d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))])
    print(f"Actual directory classes (alphabetical): {class_dirs}")
    
    dataset = keras.utils.image_dataset_from_directory(
        directory=directory,
        labels="inferred",
        label_mode="categorical",
        color_mode="rgb",
        batch_size=config.BATCH_SIZE,
        image_size=config.TARGET_SIZE,
        shuffle=False,
        seed=config.RANDOM_STATE,
        verbose=1,
    )
    
    print(f"Keras inferred class names: {dataset.class_names}")
    if dataset.class_names != config.CLASS_NAMES:
        print("WARNING: Mismatch between Keras inferred classes and config.CLASS_NAMES!")
        print("This could cause incorrect label mapping!")
    
    # Show sample images with labels
    for images, labels in dataset.take(1):
        plt.figure(figsize=(10, 6))
        for i in range(min(9, len(images))):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            pred_class = np.argmax(labels[i])
            plt.title(f"{dataset.class_names[pred_class]}\n({config.CLASS_NAMES[pred_class]})")
            plt.axis("off")
        plt.tight_layout()
        debug_dir = os.path.join(os.path.dirname(directory), "debug")
        os.makedirs(debug_dir, exist_ok=True)
        plt.savefig(os.path.join(debug_dir, "sample_images_with_labels.png"))
        plt.close()
        break

    return dataset


def get_predictions(model, dataset):
    """Get model predictions and true labels with detailed debugging"""
    print("\n=== Prediction Debug Info ===")
    print(f"Dataset class names: {dataset.class_names}")
    print(f"Config class names: {config.CLASS_NAMES}")
    
    # Get and analyze predictions
    y_pred = model.predict(dataset, verbose=0)
    print(f"\nPredictions:")
    print(f"Shape: {y_pred.shape}")
    print(f"Sample raw prediction:\n{y_pred[0]}")
    pred_classes = np.argmax(y_pred, axis=1)
    
    # Get and analyze true labels
    true_labels = np.concatenate([y for x, y in dataset])
    print(f"\nTrue Labels:")
    print(f"Shape: {true_labels.shape}")
    print(f"Sample true label:\n{true_labels[0]}")
    true_classes = np.argmax(true_labels, axis=1)
    
    # Detailed comparison
    print("\n=== Detailed Label Analysis ===")
    for i in range(min(5, len(pred_classes))):
        print(f"\nSample {i}:")
        print(f"Raw prediction probabilities: {y_pred[i]}")
        print(f"Predicted class index: {pred_classes[i]} -> {dataset.class_names[pred_classes[i]]}")
        print(f"True class index: {true_classes[i]} -> {dataset.class_names[true_classes[i]]}")
        
    print("\n=== Dataset Batch Info ===")
    total_samples = 0
    for batch, (images, labels) in enumerate(dataset):
        total_samples += len(images)
        if batch == 0:
            print(f"First batch shapes - Images: {images.shape}, Labels: {labels.shape}")
    print(f"Total samples: {total_samples}")
    print("============================\n")
    
    return true_classes, pred_classes


def plot_dataset_distribution(plots_dir, use_external):
    """Simple bar plot showing image count per class"""
    directory = config.DATASET_EXTERNAL_DIR if use_external else config.DATASET_ROOT_DIR
    counts = []
    for cls in config.CLASS_NAMES:
        class_dir = os.path.join(directory, cls)
        count = len(os.listdir(class_dir))
        counts.append(count)

    plt.figure(figsize=(10, 6))
    plt.bar(config.CLASS_NAMES, counts)
    plt.title("Dataset Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Number of Images")
    plt.savefig(os.path.join(plots_dir, "class_distribution.png"))
    plt.close()


def plot_model_performance(csv_path, plots_dir):
    """Combined plot showing accuracy and loss"""
    try:
        history = pd.read_csv(csv_path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        print(f"Warning: Could not read history file: {csv_path}")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Accuracy plot
    ax1.plot(history["accuracy"], label="Training")
    ax1.plot(history["val_accuracy"], label="Validation")
    ax1.set_title("Model Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True)

    # Loss plot
    ax2.plot(history["loss"], label="Training")
    ax2.plot(history["val_loss"], label="Validation")
    ax2.set_title("Model Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "model_performance.png"))
    plt.close()


def plot_confusion_matrix(model, plots_dir, dataset):
    """Basic confusion matrix heatmap"""
    y_true, y_pred = get_predictions(model, dataset)
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=config.CLASS_NAMES,
        yticklabels=config.CLASS_NAMES,
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(os.path.join(plots_dir, "confusion_matrix.png"))
    plt.close()


def plot_precision_recall(model, plots_dir, dataset):
    """Basic precision/recall bar plot"""
    y_true, y_pred = get_predictions(model, dataset)

    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)

    x = np.arange(len(config.CLASS_NAMES))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - width / 2, precision, width, label="Precision")
    plt.bar(x + width / 2, recall, width, label="Recall")

    plt.xlabel("Class")
    plt.ylabel("Score")
    plt.title("Precision and Recall by Class")
    plt.xticks(x, config.CLASS_NAMES)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "precision_recall.png"))
    plt.close()


def main():
    """Generate plots for both datasets"""
    import keras

    dataset_internal = dataset_from_dir(config.DATASET_ROOT_DIR)
    dataset_external = dataset_from_dir(config.DATASET_EXTERNAL_DIR)

    # Manual CNN
    if os.path.exists(config.PATH_MANUAL_CNN_MODEL):
        print("Generating plots for Manual CNN...")
        print("\nLoading Manual CNN model (Keras 3)...")
        model = keras.models.load_model(config.PATH_MANUAL_CNN_MODEL)

        # Generate plots with internal dataset
        plots_dir = os.path.join(config.DIR_MANUAL_CNN_PLOTS, "internal")

        os.makedirs(plots_dir, exist_ok=True)
        plot_dataset_distribution(plots_dir, use_external=False)
        plot_model_performance(config.PATH_MANUAL_CNN_HISTORY, plots_dir)
        plot_confusion_matrix(model, plots_dir, dataset_internal)
        plot_precision_recall(model, plots_dir, dataset_internal)

        # Generate plots with external dataset
        plots_dir = os.path.join(config.DIR_MANUAL_CNN_PLOTS, "external")
        os.makedirs(plots_dir, exist_ok=True)
        plot_dataset_distribution(plots_dir, use_external=True)
        plot_model_performance(config.PATH_MANUAL_CNN_HISTORY, plots_dir)
        plot_confusion_matrix(model, plots_dir, dataset_external)
        plot_precision_recall(model, plots_dir, dataset_external)

    # AutoKeras
    if os.path.exists(config.PATH_AUTO_KERAS_BEST):
        print("\nGenerating plots for AutoKeras...")
        print("Loading AutoKeras model (using tf.keras for compatibility)...")
        try:
            # Try loading with tf.keras for backward compatibility
            import tensorflow as tf
            model = tf.keras.models.load_model(config.PATH_AUTO_KERAS_BEST, compile=False)
            print("Successfully loaded AutoKeras model with tf.keras")
        except Exception as e:
            print(f"Failed to load with tf.keras: {e}")
            print("Falling back to standard keras.models.load_model...")
            model = keras.models.load_model(config.PATH_AUTO_KERAS_BEST)

        # Generate plots with internal dataset
        plots_dir = os.path.join(config.DIR_AUTO_KERAS_PLOTS, "internal")
        os.makedirs(plots_dir, exist_ok=True)
        plot_dataset_distribution(plots_dir, use_external=False)
        plot_model_performance(config.PATH_AUTO_KERAS_HISTORY, plots_dir)
        plot_confusion_matrix(model, plots_dir, dataset_internal)
        plot_precision_recall(model, plots_dir, dataset_internal)

        # Generate plots with external dataset
        plots_dir = os.path.join(config.DIR_AUTO_KERAS_PLOTS, "external")
        os.makedirs(plots_dir, exist_ok=True)
        plot_dataset_distribution(plots_dir, use_external=True)
        plot_model_performance(config.PATH_AUTO_KERAS_HISTORY, plots_dir)
        plot_confusion_matrix(model, plots_dir, dataset_external)
        plot_precision_recall(model, plots_dir, dataset_external)

    # HP Tuner
    if os.path.exists(config.PATH_HP_TUNER_BEST):
        print("\nGenerating plots for HP Tuner...")
        print("Loading HP Tuner model (Keras 3)...")
        model = keras.models.load_model(config.PATH_HP_TUNER_BEST)

        # Generate plots with internal dataset
        plots_dir = os.path.join(config.DIR_HP_TUNER_PLOTS, "internal")
        os.makedirs(plots_dir, exist_ok=True)
        plot_dataset_distribution(plots_dir, use_external=False)
        plot_model_performance(config.PATH_HP_TUNER_HISTORY, plots_dir)
        plot_confusion_matrix(model, plots_dir, dataset_internal)
        plot_precision_recall(model, plots_dir, dataset_internal)

        # Generate plots with external dataset
        plots_dir = os.path.join(config.DIR_HP_TUNER_PLOTS, "external")
        os.makedirs(plots_dir, exist_ok=True)
        plot_dataset_distribution(plots_dir, use_external=True)
        plot_model_performance(config.PATH_HP_TUNER_HISTORY, plots_dir)
        plot_confusion_matrix(model, plots_dir, dataset_external)
        plot_precision_recall(model, plots_dir, dataset_external)


if __name__ == "__main__":
    main()
