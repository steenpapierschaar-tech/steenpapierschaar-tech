import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from src.config import config

def create_test_dataset():
    """Create a test dataset from the external dataset directory"""
    test_ds = tf.keras.utils.image_dataset_from_directory(
        config.DATASET_EXTERNAL_DIR,
        labels='inferred',
        label_mode='int',
        class_names=config.CLASS_NAMES,
        color_mode='rgb',
        batch_size=config.BATCH_SIZE,
        image_size=config.TARGET_SIZE,
        shuffle=True,
        seed=config.RANDOM_STATE
    )
    
    # Configure the dataset for performance
    AUTOTUNE = tf.data.AUTOTUNE
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    return test_ds

def get_predictions(model, dataset):
    """Helper function to get predictions and true labels from dataset"""
    y_pred = []
    y_true = []
    
    for x, y in dataset:
        # Get predictions for current batch
        predictions = model.predict(x, verbose=0)  # Reduce verbosity
        pred_labels = np.argmax(predictions, axis=1)
        y_pred.extend(pred_labels.tolist())
        
        # Handle true labels - for test dataset, labels are already integers
        y_true.extend(y.numpy().tolist())
    
    # Convert lists to numpy arrays
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    
    return y_true, y_pred

def plot_dataset_distribution():
    """Generate a bar plot showing the number of images per class"""
    counts = []
    for cls in config.CLASS_NAMES:
        class_dir = os.path.join(config.DATASET_ROOT_DIR, cls)
        count = len(os.listdir(class_dir))
        counts.append(count)
    
    plt.figure(figsize=(10, 6))
    plt.bar(config.CLASS_NAMES, counts)
    plt.title('Dataset Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Number of Images')
    plt.savefig(os.path.join(config.PLOTS_DIR, 'class_distribution.png'))
    plt.close()

def plot_training_history(csv_path):
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
    plt.savefig(os.path.join(config.PLOTS_DIR, 'training_history.png'))
    plt.close()

def plot_confusion_matrix(model, dataset=None):
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
    plt.savefig(os.path.join(config.PLOTS_DIR, 'confusion_matrix.png'))
    plt.close()

def plot_metrics_comparison(model, dataset=None):
    """Plot precision/recall metrics with per-class and macro averages using external test dataset by default"""
    if dataset is None:
        dataset = create_test_dataset()
    y_true, y_pred = get_predictions(model, dataset)
    
    # Calculate per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None)
    recall_per_class = recall_score(y_true, y_pred, average=None)
    
    # Calculate macro averages
    macro_precision = precision_score(y_true, y_pred, average='macro')
    macro_recall = recall_score(y_true, y_pred, average='macro')
    
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
    plt.savefig(os.path.join(config.PLOTS_DIR, 'metrics_comparison.png'))
    plt.close()

def plot_bias_variance(csv_path):
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
    plt.savefig(os.path.join(config.PLOTS_DIR, 'bias_variance.png'))
    plt.close()

def generate_all_plots(model, csv_path):
    """Generate all evaluation plots for a model and its training history in one call"""
    # Create test dataset
    test_dataset = create_test_dataset()
    
    # Generate all plots
    plot_dataset_distribution()
    plot_training_history(csv_path)
    plot_confusion_matrix(model, test_dataset)
    plot_metrics_comparison(model, test_dataset)
    plot_bias_variance(csv_path)
    plot_metric_gap_analysis(csv_path)

def plot_metric_gap_analysis(csv_path):
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
    plt.savefig(os.path.join(config.PLOTS_DIR, 'metric_gap_analysis.png'))
    plt.close()
