import matplotlib.pyplot as plt
import pandas as pd
import os
import datetime
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from fileHandler import loadFiles, createOutputDir, createSubDir, createTimestampDir
from fetch_data import datasetBuilder
from featureScore import score_features, plot_feature_importance

def save_classification_report(report, outputDirectory, model_name):
    """ Save classification report to a CSV file """
    
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(outputDirectory, f'{model_name}_classification_report.csv'), index=True, index_label='Label')
    
    print(f"[INFO] Classification report saved to {model_name}_classification_report.csv')")

def ML_DecisionTree(gestures, coded_labels, label_names, outputDirectory):
    print("[INFO] Starting Decision Tree Classifier")
    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(gestures.data, coded_labels, test_size=0.2, random_state=42)
    
    # Best hyperparameters found
    best_params = {
        'criterion': 'entropy',
        'max_depth': None,
        'max_features': 'sqrt',
        'min_samples_leaf': 1,
        'min_samples_split': 2,
        'splitter': 'best'
    }

    # Create and train the Decision Tree classifier
    clf = DecisionTreeClassifier(**best_params)
    clf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = clf.predict(X_test)

    # Evaluate model performance
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=label_names, output_dict=True)
        
    # Perform cross-validation (default 5-fold)
    scores = cross_val_score(clf, X_train, y_train, cv=5)  # `cv=5` for 5-fold cross-validation

    # Save classification report to CSV
    save_classification_report(report, outputDirectory, 'DecisionTree')
    
    # Generate and plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels= label_names)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    disp.plot(cmap="Blues", ax=ax, values_format='d')
    ax.set_title('Confusion Matrix for Decision Tree Model')
    plt.savefig(os.path.join(outputDirectory, 'confusion_matrix_DecisionTree.png'))
    plt.close(fig)
    
    

def ML_KNN(gestures, coded_labels, label_names, outputDirectory, n_neighbors = 5):
    """ Train and evaluate the K-Nearest Neighbors classifier """
    
    print(f"[INFO] Starting K-Nearest Neighbors Classifier with {n_neighbors} neighbors")
    X_train, X_test, y_train, y_test = train_test_split(gestures.data, coded_labels, test_size=0.2, random_state=42)

    knn = KNeighborsClassifier(n_neighbors)
    knn.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = knn.predict(X_test)

    # Evaluate model performance
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=label_names, output_dict=True)
    
    # Save classification report to CSV
    save_classification_report(report, outputDirectory, 'KNN')
    
    predictions = knn.predict(X_test)
    
    # Generate and plot confusion matrix
    cm = confusion_matrix(y_test, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels= label_names)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    disp.plot(cmap="Blues", ax=ax, values_format='d')
    ax.set_title('Confusion Matrix for knn')
    plt.savefig(os.path.join(outputDirectory, 'confusion_matrix_knn.png'))
    plt.close(fig)
    
def ML_SVM(gestures, coded_labels, label_names, outputDirectory, kernel='rbf'):
    print(f"[INFO] Starting Support Vector Machine Classifier with {kernel} kernel")

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(gestures.data, coded_labels, test_size=0.2, random_state=42)

    # Initialize the SVM model with the specified kernel
    svm = SVC(kernel=kernel)
    
    # Train the SVM model
    svm.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = svm.predict(X_test)

    # Evaluate model performance
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=label_names, output_dict=True)
    
    # Save classification report to CSV
    save_classification_report(report, outputDirectory, 'SVM')
    
    predictions = svm.predict(X_test)
    
    # Generate and plot confusion matrix
    cm = confusion_matrix(y_test, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels= label_names)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    disp.plot(cmap="Blues", ax=ax, values_format='d')
    ax.set_title('Confusion Matrix for svm')
    plt.savefig(os.path.join(outputDirectory, 'confusion_matrix_svm.png'))
    plt.close(fig)
    return svm

if __name__ == "__main__":
    # Create output directory
    outputDir = createOutputDir()
    
    # Create timestamped subdirectory
    timestampDir = createTimestampDir(outputDir)

    # Load the dataset
    fileList = loadFiles()
    gestures = datasetBuilder(fileList, timestampDir)

    # Score the features
    feature_importances, perm_importance = score_features(gestures)

    # Plot Random Forest Feature Importance
    plot_feature_importance(feature_importances, gestures.feature_names, timestampDir, 
                            'Random Forest Feature Importance', 'rf_feature_importance.png')

    # Plot Permutation Importance
    plot_feature_importance(perm_importance.importances_mean, gestures.feature_names, timestampDir, 
                            'Permutation Feature Importance', 'permutation_importance.png')

    # Save importance scores to CSV
    importance_data = pd.DataFrame({
        'Feature': gestures.feature_names,
        'RandomForest Importance': feature_importances,
        'Permutation Importance': perm_importance.importances_mean
    }).sort_values(by='RandomForest Importance', ascending=False)

    importance_data.to_csv(os.path.join(timestampDir, 'feature_importance_scores.csv'), index=False)

    print(f"[INFO] Feature scoring completed and saved to {timestampDir}")