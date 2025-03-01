import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from src.config import config
# Add to Python path to import from src
import sys
sys.path.append('.')

def create_test_dataset(use_external=False):
    """Create a test dataset from either the root or external dataset directory.
    
    Args:
        use_external (bool): If True, use DATASET_EXTERNAL_DIR, otherwise use DATASET_ROOT_DIR
        
    Returns:
        tf.data.Dataset: Test dataset with pixel values in [0, 1] range
    """
    from src.create_dataset import dir_to_dataset, rescale_dataset
    from pathlib import Path
    
    dataset_dir = config.DATASET_EXTERNAL_DIR if use_external else config.DATASET_ROOT_DIR
    dataset = dir_to_dataset(Path(dataset_dir))
    return rescale_dataset(dataset)

def get_predictions(model, dataset):
    """Helper function to get predictions and true labels from dataset"""
    # Get predictions for entire dataset at once
    y_pred = model.predict(dataset, verbose=0)
    y_pred = np.argmax(y_pred, axis=1)
    
    # Extract true labels from the dataset
    y_true = np.concatenate([y for x, y in dataset])
    y_true = np.argmax(y_true, axis=1)
    
    return y_true, y_pred

def plot_dataset_distribution(plots_dir, use_external=False):
    """Generate a bar plot showing the number of images per class
    
    Args:
        plots_dir: Directory to save the plot
        use_external (bool): If True, use DATASET_EXTERNAL_DIR, otherwise use DATASET_ROOT_DIR
    """
    dataset_dir = config.DATASET_EXTERNAL_DIR if use_external else config.DATASET_ROOT_DIR
    counts = []
    for cls in config.CLASS_NAMES:
        class_dir = os.path.join(dataset_dir, cls)
        count = len(os.listdir(class_dir))
        counts.append(count)
    
    plt.figure(figsize=(10, 6))
    plt.bar(config.CLASS_NAMES, counts)
    plt.title('Dataset Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Number of Images')
    plt.savefig(os.path.join(plots_dir, 'class_distribution.png'))
    plt.close()

def plot_training_history(csv_path, plots_dir):
    """Plot training metrics from CSV log"""
    # Read training history
    history_df = pd.read_csv(csv_path)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot training and validation loss
    ax1.plot(history_df['loss'], label='Training Loss')
    ax1.plot(history_df['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot training and validation accuracy
    ax2.plot(history_df['accuracy'], label='Training Accuracy')
    ax2.plot(history_df['val_accuracy'], label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'training_history.png'))
    plt.close()

def plot_confusion_matrix(model, plots_dir, dataset=None):
    """Generate confusion matrix from model predictions using external test dataset by default"""
    if dataset is None:
        dataset = create_test_dataset()
    y_true, y_pred = get_predictions(model, dataset)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=config.CLASS_NAMES,
                yticklabels=config.CLASS_NAMES)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    # Save plot
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'confusion_matrix.png'))
    plt.close()

def plot_metrics_comparison(model, plots_dir, dataset=None):
    """Plot precision/recall metrics with per-class and macro averages using external test dataset by default"""
    if dataset is None:
        dataset = create_test_dataset()
    y_true, y_pred = get_predictions(model, dataset)
    
    # Calculate per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    
    # Calculate macro averages
    macro_precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot per-class metrics
    x = np.arange(len(config.CLASS_NAMES))
    width = 0.35
    
    ax1.bar(x - width/2, precision_per_class, width, label='Precision')
    ax1.bar(x + width/2, recall_per_class, width, label='Recall')
    ax1.set_xticks(x)
    ax1.set_xticklabels(config.CLASS_NAMES)
    ax1.set_ylabel('Score')
    ax1.set_title('Per-class Metrics')
    ax1.legend()
    ax1.grid(True)
    
    # Plot macro averages
    metrics = ['Precision', 'Recall']
    scores = [macro_precision, macro_recall]
    
    ax2.bar(metrics, scores)
    ax2.set_ylabel('Score')
    ax2.set_title('Macro Averages')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'metrics_comparison.png'))
    plt.close()

def plot_bias_variance(csv_path, plots_dir):
    """Plot combined error analysis with moving averages"""
    history = pd.read_csv(csv_path)
    
    # Calculate moving averages
    window_size = config.PLOT_WINDOW_SIZE
    history['smooth_train_loss'] = history['loss'].rolling(window=window_size, center=True).mean()
    history['smooth_val_loss'] = history['val_loss'].rolling(window=window_size, center=True).mean()
    
    # Calculate error components
    history['bias'] = history['smooth_train_loss']
    history['variance'] = history['smooth_val_loss'] - history['smooth_train_loss']
    
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(history) + 1)
    
    # Plot error components as stacked area
    plt.fill_between(epochs, history['bias'], color='lightblue', alpha=0.5, label='Bias')
    plt.fill_between(epochs, history['smooth_train_loss'], history['smooth_val_loss'], 
                    color='salmon', alpha=0.5, label='Variance')
    
    # Plot smoothed curves
    plt.plot(epochs, history['smooth_train_loss'], 'b-', label='Training Loss (MA)')
    plt.plot(epochs, history['smooth_val_loss'], 'r-', label='Validation Loss (MA)')
    
    plt.title('Bias-Variance Analysis')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'bias_variance.png'))
    plt.close()

def generate_all_plots(model, csv_path, plots_dir, use_external=False):
    """Generate all evaluation plots for a model and its training history in one call
    
    Args:
        model: Trained Keras model
        csv_path: Path to training history CSV
        plots_dir: Directory to save plots
        use_external (bool): If True, use external dataset as test set
    """
    # Create output directory if it doesn't exist
    os.makedirs(plots_dir, exist_ok=True)
    
    # Create test dataset
    test_dataset = create_test_dataset(use_external)
    
    # Generate all plots
    plot_dataset_distribution(plots_dir, use_external)
    plot_training_history(csv_path, plots_dir)
    plot_confusion_matrix(model, plots_dir, test_dataset)
    plot_metrics_comparison(model, plots_dir, test_dataset)
    plot_bias_variance(csv_path, plots_dir)
    plot_metric_gap_analysis(csv_path, plots_dir)

def plot_metric_gap_analysis(csv_path, plots_dir):
    """Analyze overfitting through metric divergence"""
    history = pd.read_csv(csv_path)
    
    # Calculate gaps between training and validation metrics
    history['accuracy_gap'] = history['accuracy'] - history['val_accuracy']
    history['loss_gap'] = history['val_loss'] - history['loss']
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    epochs = range(1, len(history) + 1)
    
    # Plot accuracy gap
    ax1.plot(epochs, history['accuracy_gap'], 'b-')
    ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax1.fill_between(epochs, history['accuracy_gap'], 0, 
                    where=history['accuracy_gap'] >= 0,
                    color='salmon', alpha=0.3, label='Potential Overfitting')
    ax1.set_title('Accuracy Gap Analysis')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training - Validation Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss gap
    ax2.plot(epochs, history['loss_gap'], 'b-')
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax2.fill_between(epochs, history['loss_gap'], 0,
                    where=history['loss_gap'] >= 0,
                    color='salmon', alpha=0.3, label='Potential Overfitting')
    ax2.set_title('Loss Gap Analysis')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation - Training Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'metric_gap_analysis.png'))
    plt.close()

def main():
    """
    Generate plots for all three training strategies twice:
    1. Using photoDataset as test set
    2. Using photoDataset_external as test set
    """
    import keras

    # First run - Using photoDataset as test set
    print("\nGenerating plots using photoDataset as test set...")
    
    # Manual CNN with internal dataset
    if os.path.exists(config.PATH_MANUAL_CNN_MODEL):
        print("Generating plots for Manual CNN...")
        model = keras.models.load_model(config.PATH_MANUAL_CNN_MODEL)
        generate_all_plots(model, config.PATH_MANUAL_CNN_LOG, 
                         os.path.join(config.DIR_MANUAL_CNN_PLOTS, 'internal_dataset'),
                         use_external=False)

    # AutoKeras with internal dataset
    if os.path.exists(config.PATH_AUTO_KERAS_BEST):
        print("Generating plots for AutoKeras...")
        model = keras.models.load_model(config.PATH_AUTO_KERAS_BEST)
        generate_all_plots(model, config.PATH_AUTO_KERAS_LOG,
                         os.path.join(config.DIR_AUTO_KERAS_PLOTS, 'internal_dataset'),
                         use_external=False)

    # HP Tuner with internal dataset
    if os.path.exists(config.PATH_HP_TUNER_BEST):
        print("Generating plots for HP Tuner...")
        model = keras.models.load_model(config.PATH_HP_TUNER_BEST)
        generate_all_plots(model, config.PATH_HP_TUNER_LOG,
                         os.path.join(config.DIR_HP_TUNER_PLOTS, 'internal_dataset'),
                         use_external=False)

    # Second run - Using photoDataset_external as test set
    print("\nGenerating plots using photoDataset_external as test set...")
    
    # Manual CNN with external dataset
    if os.path.exists(config.PATH_MANUAL_CNN_MODEL):
        print("Generating plots for Manual CNN...")
        model = keras.models.load_model(config.PATH_MANUAL_CNN_MODEL)
        generate_all_plots(model, config.PATH_MANUAL_CNN_LOG,
                         os.path.join(config.DIR_MANUAL_CNN_PLOTS, 'external_dataset'),
                         use_external=True)

    # AutoKeras with external dataset
    if os.path.exists(config.PATH_AUTO_KERAS_BEST):
        print("Generating plots for AutoKeras...")
        model = keras.models.load_model(config.PATH_AUTO_KERAS_BEST)
        generate_all_plots(model, config.PATH_AUTO_KERAS_LOG,
                         os.path.join(config.DIR_AUTO_KERAS_PLOTS, 'external_dataset'),
                         use_external=True)

    # HP Tuner with external dataset
    if os.path.exists(config.PATH_HP_TUNER_BEST):
        print("Generating plots for HP Tuner...")
        model = keras.models.load_model(config.PATH_HP_TUNER_BEST)
        generate_all_plots(model, config.PATH_HP_TUNER_LOG,
                         os.path.join(config.DIR_HP_TUNER_PLOTS, 'external_dataset'),
                         use_external=True)

if __name__ == "__main__":
    main()
