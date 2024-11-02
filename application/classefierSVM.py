from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

def ML_DecisionTree(gestures, coded_labels, label_names):
    # Encode labels as integers
    label_encoder = LabelEncoder()
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