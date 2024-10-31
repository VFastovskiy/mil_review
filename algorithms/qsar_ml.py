from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score

class QSARModel:
    def __init__(self, X_train, y_train, X_test, y_test):
        print("Initializing QSARModel...")
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        print("Initialization complete.")

    def run_random_forest(self):
        """
        Train a RandomForestClassifier with 500 trees and default settings
        """
        print("Setting up RandomForestClassifier with 500 trees...")
        rf_model = RandomForestClassifier(n_estimators=500, n_jobs=-1)

        print("Fitting the model to the training data...")
        rf_model.fit(self.X_train, self.y_train)
        print("Model training complete.")

        print("Predicting on the test data...")
        y_pred = rf_model.predict(self.X_test)

        print("Calculating balanced accuracy...")
        balanced_acc = balanced_accuracy_score(self.y_test, y_pred)

        print(f"Balanced Accuracy: {balanced_acc}")

        return balanced_acc
