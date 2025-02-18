import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Model
from src.config import config
from tensorboard.backend.event_processing import event_accumulator

# Create plots subdirectory
PLOTS_DIR = os.path.join(config.OUTPUT_DIR, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

class DLVisualizer:
    def __init__(self, class_names):
        self.class_names = class_names
        plt.style.use('seaborn')
        sns.set_palette("husl")
        
    def plot_confusion_matrix(self, y_true, y_pred, normalize=False, title='Confusion Matrix'):
        """
        Plot a confusion matrix with optional normalization
        """
        cm = confusion_matrix(y_true, y_pred)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                    cmap='Blues', xticklabels=self.class_names,
                    yticklabels=self.class_names)
        plt.title(title)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        return plt
        
    def plot_training_history(self, history, metrics=['accuracy', 'loss']):
        """
        Plot training and validation metrics over epochs
        """
        plt.figure(figsize=(12, 6))
        for i, metric in enumerate(metrics):
            plt.subplot(1, len(metrics), i+1)
            plt.plot(history.history[metric], label=f'Training {metric}')
            plt.plot(history.history[f'val_{metric}'], label=f'Validation {metric}')
            plt.title(f'{metric.capitalize()} over epochs')
            plt.xlabel('Epoch')
            plt.ylabel(metric.capitalize())
            plt.legend()
        plt.tight_layout()
        return plt
        
    def plot_feature_maps(self, model, layer_name, input_image, n_cols=8):
        """
        Visualize feature maps from a specific CNN layer
        """
        layer_output = model.get_layer(layer_name).output
        intermediate_model = Model(inputs=model.input, outputs=layer_output)
        feature_maps = intermediate_model.predict(input_image[np.newaxis, ...])
        
        n_features = feature_maps.shape[-1]
        n_rows = int(np.ceil(n_features / n_cols))
        
        plt.figure(figsize=(n_cols * 2, n_rows * 2))
        for i in range(n_features):
            plt.subplot(n_rows, n_cols, i+1)
            plt.imshow(feature_maps[0, :, :, i], cmap='viridis')
            plt.axis('off')
        plt.tight_layout()
        return plt
        
    def plot_augmentation_examples(self, original, augmented, n_examples=3):
        """
        Show before-and-after examples of data augmentation
        """
        plt.figure(figsize=(n_examples * 4, 8))
        for i in range(n_examples):
            plt.subplot(2, n_examples, i+1)
            plt.imshow(original[i])
            plt.title('Original')
            plt.axis('off')
            
            plt.subplot(2, n_examples, n_examples+i+1)
            plt.imshow(augmented[i])
            plt.title('Augmented')
            plt.axis('off')
        plt.tight_layout()
        return plt

    def save_plot(self, plt, filename, dpi=300):
        """
        Save plot to file with high quality
        """
        plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=dpi, bbox_inches='tight')
        plt.close()

def load_tensorboard_data(log_dir):
    """Load training data from TensorBoard logs"""
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()
    
    data = {}
    for tag in ea.Tags()['scalars']:
        data[tag] = [(s.step, s.value) for s in ea.Scalars(tag)]
    
    return data

def main():
    """Demonstrate plotting capabilities"""
    print("DL Visualization Module")
    print(f"Plots will be saved to: {PLOTS_DIR}")
    
    # Load TensorBoard data
    log_dir = os.path.join(config.LOGS_DIR, '0002/execution0/train')
    if os.path.exists(log_dir):
        print(f"Loading training data from: {log_dir}")
        training_data = load_tensorboard_data(log_dir)
        
        # Example usage
        visualizer = DLVisualizer(['Class 0', 'Class 1', 'Class 2'])
        
        # Plot training metrics
        if 'epoch_accuracy' in training_data:
            steps, acc = zip(*training_data['epoch_accuracy'])
            plt.figure()
            plt.plot(steps, acc, label='Training Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            visualizer.save_plot(plt, 'training_accuracy.png')
            
        print("Training plots generated successfully!")
    else:
        print("No training logs found. Running example plots...")
        # Example usage
        visualizer = DLVisualizer(['Class 0', 'Class 1', 'Class 2'])
        
        # Example data
        y_true = np.random.randint(0, 3, 100)
        y_pred = np.random.randint(0, 3, 100)
        
        # Create and save example plots
        plt = visualizer.plot_confusion_matrix(y_true, y_pred)
        visualizer.save_plot(plt, 'example_confusion_matrix.png')
        
        print("Example plots generated successfully!")

if __name__ == "__main__":
    main()
