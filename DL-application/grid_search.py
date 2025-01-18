from sklearn.model_selection import GridSearchCV

def perform_grid_search(estimator, param_grid, X_train, y_train):
    """
    Perform grid search to find the best hyperparameters for the given estimator.

    Parameters:
    - estimator: The machine learning estimator (e.g., SVM, RandomForest).
    - param_grid: Dictionary with parameters names (str) as keys and lists of parameter settings to try as values.
    - X_train: Training data features.
    - y_train: Training data labels.

    Returns:
    - best_estimator: Estimator with the best found parameters.
    - best_params: Dictionary containing the best parameters.
    """
    grid_search = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_estimator = grid_search.best_estimator_
    best_params = grid_search.best_params_
    return best_estimator, best_params
