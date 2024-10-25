import pandas as pd
import os
from fingerprints.scikit_fingerprints import MolecularFingerprints
from algorithms.qsar_ml import QSARModel


class FingerprintPipeline:
    def __init__(self, train_sdf, test_sdf, train_labels, test_labels, output_dir='output', log_file='log.txt'):
        print("Initializing pipeline...")
        self.train_sdf = train_sdf
        self.test_sdf = test_sdf
        self.train_labels = train_labels
        self.test_labels = test_labels
        self.output_dir = output_dir
        self.log_file = log_file

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(log_file, 'w') as log:
            log.write("Fingerprint calculation log:\n")

        print(f"Loading train molecules from {train_sdf}...")
        self.train_molecules = MolecularFingerprints(train_sdf)
        print(f"Loading test molecules from {test_sdf}...")
        self.test_molecules = MolecularFingerprints(test_sdf)
        print("Initialization complete.")

    def calculate_and_save_fingerprint(self, fingerprint_name):
        """
        Calculate and save fingerprints for train and test datasets as CSV files.
        """
        try:
            print(f"Calculating {fingerprint_name} fingerprint for train dataset...")
            X_train = self.train_molecules.calculate_fingerprint(fingerprint_name)
            train_df = pd.DataFrame(X_train)
            train_filename = os.path.join(self.output_dir, f"{fingerprint_name}_train.csv")
            train_df.to_csv(train_filename, index=False)
            print(f"Saved {fingerprint_name} fingerprint for train dataset to {train_filename}")

            print(f"Calculating {fingerprint_name} fingerprint for test dataset...")
            X_test = self.test_molecules.calculate_fingerprint(fingerprint_name)
            test_df = pd.DataFrame(X_test)
            test_filename = os.path.join(self.output_dir, f"{fingerprint_name}_test.csv")
            test_df.to_csv(test_filename, index=False)
            print(f"Saved {fingerprint_name} fingerprint for test dataset to {test_filename}")

        except Exception as e:
            print(f"Error calculating fingerprint {fingerprint_name}: {e}")
            # Log the error to log.txt
            with open(self.log_file, 'a') as log:
                log.write(f"Failed to calculate {fingerprint_name}: {str(e)}\n")
            return False

        return True

    def load_fingerprint_data(self, fingerprint_name):
        """
        Load fingerprints for train and test datasets from CSV files.
        """
        print(f"Loading {fingerprint_name} fingerprint data from CSV files...")
        train_filename = os.path.join(self.output_dir, f"{fingerprint_name}_train.csv")
        test_filename = os.path.join(self.output_dir, f"{fingerprint_name}_test.csv")

        X_train = pd.read_csv(train_filename).values
        X_test = pd.read_csv(test_filename).values

        print(f"Loaded {fingerprint_name} fingerprint data for train and test.")
        return X_train, X_test

    def run_optimization(self, fingerprint_name):
        """
        Load saved fingerprints and run optimization for the selected fingerprint.
        """
        print(f"Running optimization for {fingerprint_name} fingerprint...")

        X_train, X_test = self.load_fingerprint_data(fingerprint_name)

        # Train and test using XGBoost with CPU optimization (15 cores)
        model = QSARModel(X_train, self.train_labels, X_test, self.test_labels)
        accuracy, best_params = model.run_xgboost_with_cpu_optimization()

        print(f"Balanced Accuracy for {fingerprint_name}: {accuracy}")
        print(f"Best Hyperparameters for {fingerprint_name}: {best_params}")

        return accuracy

    def calculate_all_fingerprints(self, fingerprints_to_run):
        """
        Calculate and save all fingerprints provided in the list.
        """
        print(f"Starting fingerprint calculation for {len(fingerprints_to_run)} fingerprints...")
        for fingerprint_name in fingerprints_to_run:
            print(f"\n--- Calculating and saving {fingerprint_name} ---")
            if not self.calculate_and_save_fingerprint(fingerprint_name):
                print(f"Skipping {fingerprint_name} due to error.")
        print("Fingerprint calculation complete.")

    def run_all_optimizations(self, fingerprints_to_run):
        """
        Run model optimization for all fingerprints provided in the list.
        Collect results in a CSV file with two columns: fingerprints and balanced accuracy.
        """
        print(f"Starting model optimization for {len(fingerprints_to_run)} fingerprints...")
        results = []

        for fingerprint_name in fingerprints_to_run:
            try:
                print(f"\n--- Running optimization for {fingerprint_name} ---")
                accuracy = self.run_optimization(fingerprint_name)
                results.append((fingerprint_name, accuracy))
            except Exception as e:
                print(f"Error optimizing fingerprint {fingerprint_name}: {e}")
                # Log the error to log.txt
                with open(self.log_file, 'a') as log:
                    log.write(f"Failed to optimize {fingerprint_name}: {str(e)}\n")

        results_df = pd.DataFrame(results, columns=['fingerprint', 'balanced_accuracy'])
        results_filename = os.path.join(self.output_dir, "optimization_results.csv")
        results_df.to_csv(results_filename, index=False)
        print(f"Saved optimization results to {results_filename}")

        print("Model optimization complete.")
