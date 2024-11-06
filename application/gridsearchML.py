from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

def GS_DecisionTree(gestures, coded_labels, label_names):
    print(f"Starting DecisionTree with Hyperparameter Tuning")

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(gestures.data, coded_labels, test_size=0.2, random_state=42)

    # Define the model (Decision Tree Classifier)
    dtree = DecisionTreeClassifier(random_state=42)

    # Define the hyperparameter grid for tuning
    param_grid = {
        'criterion': ['gini', 'entropy'],  # Criterion for split quality
        'max_depth': [None, 5, 10, 15, 20],  # Maximum depth of the tree
        'min_samples_split': [2, 5, 10],  # Minimum samples required to split a node
        'min_samples_leaf': [1, 2, 5],  # Minimum samples at leaf nodes
        'max_features': [None, 'auto', 'sqrt', 'log2'],  # Features to consider for splitting
        'splitter': ['best', 'random']  # Splitter strategy for choosing the best split
    }

    # Set up GridSearchCV with 5-fold cross-validation
    grid_search = GridSearchCV(estimator=dtree, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)

    # Fit the GridSearchCV to the training data
    grid_search.fit(X_train, y_train)

    # Print the best hyperparameters found by GridSearchCV
    print("\nBest hyperparameters found:", grid_search.best_params_)

    # Get the best model from GridSearchCV
    best_dtree = grid_search.best_estimator_

    # Make predictions with the best model on the test set
    y_pred = best_dtree.predict(X_test)

    # Evaluate model performance
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=label_names)

    # Display the results
    print(f"\nTest set accuracy: {accuracy:.2f}")
    print("Classification Report:\n", report)

    # Return the best model and the best hyperparameters
    return best_dtree, grid_search.best_params_