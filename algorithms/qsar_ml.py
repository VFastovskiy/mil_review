from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import balanced_accuracy_score

class QSARModel:
    def __init__(self, X_train, y_train, X_test, y_test):
        print("Initializing QSARModel...")
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        print("Initialization complete.")

    def run_xgboost_with_cpu_optimization(self):
        """
        Train an XGBoost classifier with hyperparameter optimization using GridSearchCV and CPU support.
        This version uses 15 CPU cores.
        """
        print("Setting up XGBoost model with CPU optimization (using 15 cores)...")
        # Define the XGBoost model with CPU support, remove `use_label_encoder`
        xgb_model = XGBClassifier(
            eval_metric='logloss',
            tree_method='hist',  # Use CPU with histogram-based algorithm
            n_jobs=15  # Set the number of CPU cores to use (15 out of 16)
        )

        # Define hyperparameters to tune
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0]
        }

        print("Starting GridSearchCV for hyperparameter tuning...")
        # Grid search for optimal parameters
        grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid,
                                   scoring='balanced_accuracy', cv=3, n_jobs=15, verbose=1)

        # Fit the model on the training data
        print("Fitting the model to the training data...")
        grid_search.fit(self.X_train, self.y_train)
        print("GridSearchCV complete.")

        # Best model from grid search
        best_model = grid_search.best_estimator_
        print("Best model selected with optimal hyperparameters.")

        # Predict on the test data
        print("Predicting on the test data...")
        y_pred = best_model.predict(self.X_test)

        # Calculate balanced accuracy
        print("Calculating balanced accuracy...")
        balanced_acc = balanced_accuracy_score(self.y_test, y_pred)

        print(f"Best Parameters: {grid_search.best_params_}")
        print(f"Balanced Accuracy: {balanced_acc}")

        return balanced_acc, grid_search.best_params_
