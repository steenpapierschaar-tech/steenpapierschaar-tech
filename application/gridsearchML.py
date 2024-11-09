from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import os
import datetime
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from fileHandler import loadFiles, createOutputDir, createSubDir, createTimestampDir



def GS_DecisionTree(gestures, coded_labels, label_names, outputDir, verbose=2):

    print(f"[INFO] Starting DecisionTree with Hyperparameter Tuning")

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(gestures.data, coded_labels, test_size=0.2, random_state=42)

    # Define the model (Decision Tree Classifier)
    dtree = DecisionTreeClassifier(random_state=42)

    # Define the hyperparameter grid for tuning
    param_grid = {
        'criterion': ['gini', 'entropy'],  # Criterion for split quality
        'max_depth': [None, 1, 2, 5, 10, 15, 20],  # Maximum depth of the tree
        'min_samples_split': [2, 5, 10],  # Minimum samples required to split a node
        'min_samples_leaf': [1, 2, 5],  # Minimum samples at leaf nodes
        'max_features': [None, 'sqrt', 'log2'],  # Features to consider for splitting
        'splitter': ['best', 'random']  # Splitter strategy for choosing the best split
    }

    # Set up GridSearchCV with 5-fold cross-validation
    grid_search = GridSearchCV(estimator=dtree, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=verbose)

    # Fit the GridSearchCV to the training data
    grid_search.fit(X_train, y_train)

    # Print the best hyperparameters found by GridSearchCV
    #print("\nBest hyperparameters found:", grid_search.best_params_)

    # Get the best model from GridSearchCV
    best_dtree = grid_search.best_estimator_

    # Make predictions with the best model on the test set
    y_pred = best_dtree.predict(X_test)

    # Evaluate model performance
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=label_names)

    # Display the results
    #print(f"\nTest set accuracy: {accuracy:.2f}")
    #print("Classification Report:\n", report)
    
    predictions = grid_search.predict(X_test)
    
    # Generate and plot confusion matrix
    cm = confusion_matrix(y_test, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels= label_names)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    disp.plot(cmap="Blues", ax=ax, values_format='d')
    ax.set_title('Confusion Matrix for grid Decision Tree Model')
    plt.savefig(os.path.join(outputDir, 'confusion_matrix_grid_DecisionTree.png'))
    plt.close(fig)

    # Return the best model and the best hyperparameters
    return best_dtree, grid_search.best_params_ , grid_search