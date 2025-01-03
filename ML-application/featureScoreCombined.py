import os
import numpy as np
import datetime
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
from fetch_data import datasetBuilder
from fileHandler import loadFiles, createOutputDir, createSubDir, createTimestampDir

def generate_combined_features(trainX, degree=2):
    """ Generate polynomial features up to a given degree """
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    combined_features = poly.fit_transform(trainX)
    feature_names = poly.get_feature_names_out(gestures.feature_names)
    
    return combined_features, feature_names

def score_combined_features(gestures):
    """ Train model and score feature importance on combined features """
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

    # Generate combined polynomial features
    combined_trainX, combined_feature_names = generate_combined_features(trainX)
    combined_testX, _ = generate_combined_features(testX)

    # Train a Random Forest Classifier on combined features
    clf = RandomForestClassifier(n_estimators=500, random_state=42)
    clf.fit(combined_trainX, trainY)

    # Feature importance from the Random Forest model
    feature_importances = clf.feature_importances_

    # Permutation importance
    perm_importance = permutation_importance(clf, combined_testX, testY, n_repeats=10, random_state=42)
    
    return feature_importances, perm_importance, combined_feature_names

def plot_feature_importance(importances, feature_names, output_subdir, plot_title, file_name, top_n=5):
    """ Plot and save feature importance for the top N most important features """
    sorted_idx = np.argsort(importances)[::-1]  # Sort features by importance
    sorted_importances = importances[sorted_idx][:top_n]  # Get top N importances
    sorted_features = np.array(feature_names)[sorted_idx][:top_n]  # Get top N features

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
    # Create output directory
    output_dir = createOutputDir()
    
    # Create timestamped subdirectory
    timestamped_dir = createTimestampDir(output_dir)

    # Load the dataset
    fileList = loadFiles()
    gestures = datasetBuilder(fileList, timestamped_dir)

    # Score combined features
    feature_importances, perm_importance, combined_feature_names = score_combined_features(gestures)

    # Plot only the top 10 Random Forest Feature Importance
    plot_feature_importance(feature_importances, combined_feature_names, timestamped_dir, 
                            'Random Forest Feature Importance (Top 10 Features)', 'rf_combined_feature_importance_top5.png', top_n=5)

    # Plot only the top 10 Permutation Importance
    plot_feature_importance(perm_importance.importances_mean, combined_feature_names, timestamped_dir, 
                            'Permutation Feature Importance (Top 10 Features)', 'permutation_combined_importance_top5.png', top_n=5)

    # Save combined feature importance scores to CSV
    importance_data = pd.DataFrame({
        'Feature': combined_feature_names,
        'RandomForest Importance': feature_importances,
        'Permutation Importance': perm_importance.importances_mean
    }).sort_values(by='RandomForest Importance', ascending=False)

    importance_data.to_csv(os.path.join(timestamped_dir, 'combined_feature_importance_scores.csv'), index=False)

    print(f"[INFO] Combined feature scoring completed and saved to {timestamped_dir}")