import os
import numpy as np
import datetime
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
from fetch_data import loadFiles, datasetBuilder

def score_features(gestures):
    """ Train model and score feature importance """
    # Encode the categorical labels
    le = LabelEncoder()
    coded_labels = le.fit_transform(gestures.target)

    # Partition the data into training and testing splits using 75% of
    # the data for training and the remaining 25% for testing
    trainX, testX, trainY, testY = train_test_split(
        gestures.data, coded_labels, test_size=0.25, stratify=gestures.target)

    # Data preparation (scaling)
    scaler = StandardScaler()
    trainX = scaler.fit_transform(trainX)
    testX = scaler.transform(testX)

    # Train a Random Forest Classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(trainX, trainY)

    # Feature importance from the Random Forest model
    feature_importances = clf.feature_importances_

    # Permutation importance
    perm_importance = permutation_importance(clf, testX, testY, n_repeats=10, random_state=42)
    
    return feature_importances, perm_importance

def plot_feature_importance(importances, feature_names, output_subdir, plot_title, file_name):
    """ Plot and save feature importance """
    sorted_idx = np.argsort(importances)[::-1]
    sorted_importances = importances[sorted_idx]
    sorted_features = np.array(feature_names)[sorted_idx]

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x=sorted_importances, y=sorted_features, palette="Blues_d")
    plt.title(plot_title)
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig(os.path.join(output_subdir, file_name))
    plt.show()

if __name__ == "__main__":
    # Create output directory with a timestamped subfolder
    output_dir = 'output'
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_subdir = os.path.join(output_dir, timestamp)    
    if not os.path.exists(output_subdir):
        os.makedirs(output_subdir, exist_ok=True)

    # Load the dataset
    fileList = loadFiles()
    gestures = datasetBuilder(fileList)

    # Score the features
    feature_importances, perm_importance = score_features(gestures)

    # Plot Random Forest Feature Importance
    plot_feature_importance(feature_importances, gestures.feature_names, output_subdir, 
                            'Random Forest Feature Importance', 'rf_feature_importance.png')

    # Plot Permutation Importance
    plot_feature_importance(perm_importance.importances_mean, gestures.feature_names, output_subdir, 
                            'Permutation Feature Importance', 'permutation_importance.png')

    # Save importance scores to CSV
    importance_data = pd.DataFrame({
        'Feature': gestures.feature_names,
        'RandomForest Importance': feature_importances,
        'Permutation Importance': perm_importance.importances_mean
    }).sort_values(by='RandomForest Importance', ascending=False)

    importance_data.to_csv(os.path.join(output_subdir, 'feature_importance_scores.csv'), index=False)

    print(f"[INFO] Feature scoring completed and saved to {output_subdir}")
