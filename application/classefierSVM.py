from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

def ML_DecisionTree(gestures, coded_labels, label_names):
    print(f"Starting DecisionTree")
    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(gestures.data, coded_labels, test_size=0.2, random_state=42)

    # Create and train the Decision Tree classifier
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = clf.predict(X_test)

    # Evaluate model performance
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=label_names)

    # Display results
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:\n", report)
    
def ML_KNN(gestures, coded_labels, label_names, n_neighbors = 5):  
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
    
def ML_SVM(gestures, coded_labels, label_names, kernel='rbf'):
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