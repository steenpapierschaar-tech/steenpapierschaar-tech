import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import KFold
from src.config import config
from src.create_dataset import create_dataset

# Load your dataset
train_ds, val_ds = create_dataset()

# Convert the dataset to NumPy arrays
X_train = np.concatenate([x for x, y in train_ds], axis=0)
y_train = np.concatenate([y for x, y in train_ds], axis=0)

# Load the saved Keras model
model_path = config.MODEL_BEST_PATH
original_model = keras.models.load_model(model_path)

# Set up K-fold cross-validation
k = 5  # Number of folds
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# Store evaluation metrics
scores = []

for train_index, test_index in kf.split(X_train):
    X_train_fold, X_test_fold = X_train[train_index], X_train[test_index]
    y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]

    # Reload model for each fold (to reset weights)
    model = keras.models.clone_model(original_model)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # Train the model
    model.fit(X_train_fold, y_train_fold, epochs=10, batch_size=32, verbose=0)

    # Evaluate on the test set
    score = model.evaluate(X_test_fold, y_test_fold, verbose=0)
    scores.append(score[1])  # Accuracy

# Print cross-validation results
#explean this better in the print statement
print(f"Cross-validation accuracy: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
