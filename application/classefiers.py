from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import os
import datetime
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

    
def ML_DecisionTree(gestures, coded_labels, label_names):
    output_dir = 'output'
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_subdir = os.path.join(output_dir, timestamp)    
    if not os.path.exists(output_subdir):
        os.makedirs(output_subdir, exist_ok=True)
    print(f"Starting DecisionTree")
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
    report = classification_report(y_test, y_pred, target_names=label_names)
        
    # Perform cross-validation (default 5-fold)
    scores = cross_val_score(clf, X_train, y_train, cv=5)  # `cv=5` for 5-fold cross-validation

    # Print the results
    print("Cross-validation scores:", scores)
    print("Mean cross-validation score:", scores.mean())
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:\n", report)
    
    # Generate and plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels= label_names)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    disp.plot(cmap="Blues", ax=ax, values_format='d')
    ax.set_title('Confusion Matrix for Decision Tree Model')
    plt.savefig(os.path.join(output_subdir, 'confusion_matrix_DecisionTree.png'))
    plt.close(fig)

    
def ML_KNN(gestures, coded_labels, label_names, n_neighbors = 5):  
    output_dir = 'output'
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_subdir = os.path.join(output_dir, timestamp)    
    if not os.path.exists(output_subdir):
        os.makedirs(output_subdir, exist_ok=True)
    print(f"Starting K-Nearest Neighbors with {n_neighbors} Neighbours")
    X_train, X_test, y_train, y_test = train_test_split(gestures.data, coded_labels, test_size=0.2, random_state=42)

    knn = KNeighborsClassifier(n_neighbors)
    knn.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = knn.predict(X_test)

    # Evaluate model performance
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=label_names)

    # Display results
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:\n", report)
    
    predictions = knn.predict(X_test)
    
    # Generate and plot confusion matrix
    cm = confusion_matrix(y_test, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels= label_names)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    disp.plot(cmap="Blues", ax=ax, values_format='d')
    ax.set_title('Confusion Matrix for knn')
    plt.savefig(os.path.join(output_subdir, 'confusion_matrix_knn.png'))
    plt.close(fig)
    
        
def ML_SVM(gestures, coded_labels, label_names, kernel='rbf'):
    output_dir = 'output'
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_subdir = os.path.join(output_dir, timestamp)    
    if not os.path.exists(output_subdir):
        os.makedirs(output_subdir, exist_ok=True)
    print(f"Starting Support Vector Machine with {kernel} kernel")

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
    report = classification_report(y_test, y_pred, target_names=label_names)

    # Display results
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:\n", report)
    predictions = svm.predict(X_test)
    
    # Generate and plot confusion matrix
    cm = confusion_matrix(y_test, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels= label_names)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    disp.plot(cmap="Blues", ax=ax, values_format='d')
    ax.set_title('Confusion Matrix for svm')
    plt.savefig(os.path.join(output_subdir, 'confusion_matrix_svm.png'))
    plt.close(fig)
    return svm