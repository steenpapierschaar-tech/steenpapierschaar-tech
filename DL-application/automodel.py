import autokeras as ak
import keras
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from src.config import config
from src.tensorboard import TensorboardLauncher
from src.training_callbacks import ringring_callbackplease
from src.create_plots import (
    plot_dataset_distribution,
    plot_training_history,
    plot_confusion_matrix_automodel,
    plot_metrics_comparison_automodel,
    plot_bias_variance,
    plot_metric_gap_analysis
)

# Enable mixed precision training for better performance on compatible GPUs
# This allows some operations to use float16 instead of float32
keras.mixed_precision.set_global_policy("mixed_float16")

# Set global random seed for reproducibility
# This ensures consistent results across runs
keras.utils.set_random_seed(config.RANDOM_STATE)

# Clear any existing Keras session to free up memory
# This helps prevent memory leaks between training runs
keras.utils.clear_session(free_memory=True)

# Initialize and start TensorBoard for visualization
# This allows monitoring of training metrics in real-time
tensorboard = TensorboardLauncher(config.LOGS_DIR)
tensorboard.start_tensorboard()

train_data = ak.image_dataset_from_directory(
    directory=config.DATASET_ROOT_DIR,
    batch_size=config.BATCH_SIZE,
    seed=config.RANDOM_STATE,
    shuffle=True,
    validation_split=config.VALIDATION_SPLIT,
    subset="training",
    image_size=config.TARGET_SIZE,
)

validation_data = ak.image_dataset_from_directory(
    directory=config.DATASET_ROOT_DIR,
    batch_size=config.BATCH_SIZE,
    seed=config.RANDOM_STATE,
    shuffle=True,
    validation_split=config.VALIDATION_SPLIT,
    subset="validation",
    image_size=config.TARGET_SIZE,
)



# Calculate steps before modifying datasets
train_steps = len(train_data)
validation_steps = len(validation_data)

# Autotune the dataset for performance and add repeat
AUTOTUNE = tf.data.AUTOTUNE
train_data = train_data.prefetch(buffer_size=AUTOTUNE)
validation_data = validation_data.prefetch(buffer_size=AUTOTUNE)
input = ak.ImageInput()
output = ak.ImageBlock(
    block_type="vanilla",
)(input)

classification = ak.ClassificationHead()(output)

# Define the AutoKeras image classifier
clf = ak.AutoModel(
    inputs=input,
    outputs=classification,
    directory=config.MODEL_DIR,
    max_trials=config.MAX_TRIALS,
    seed=config.RANDOM_STATE,
    project_name="AutoKeras",
)

# Train the classifier
clf.fit(
    x=train_data,
    validation_data=validation_data,
    callbacks=ringring_callbackplease(),
    epochs=config.EPOCHS,
    steps_per_epoch=train_steps,
    validation_steps=validation_steps
)

print("Model training complete.")

# Evaluate the classifier

test_data = ak.image_dataset_from_directory(
    directory=config.DATASET_EXTERNAL_DIR,
    batch_size=config.BATCH_SIZE,
    seed=config.RANDOM_STATE,
    shuffle=True,
    image_size=config.TARGET_SIZE,
)

print("Evaluating the model...")
results = clf.evaluate(test_data)
print("Model evaluation complete.")
print(f"Test accuracy: {results}")

# Export and save the model
exported_model = clf.export_model()
exported_model.save(config.MODEL_BEST_PATH)

# Generate plots showing dataset and model performance
plot_dataset_distribution()  # Dataset balance visualization
plot_training_history(config.CSV_LOG_PATH)  # Training progress
plot_confusion_matrix_automodel(exported_model, test_data, config.CLASS_NAMES)  # Classification performance
plot_metrics_comparison_automodel(exported_model, test_data, config.CLASS_NAMES)  # Precision/Recall analysis
plot_bias_variance(config.CSV_LOG_PATH)  # Bias-Variance tradeoff
plot_metric_gap_analysis(config.CSV_LOG_PATH)  # Overfitting analysis
