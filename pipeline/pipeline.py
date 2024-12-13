import pandas as pd
import os

from fingerprints.scikit_fingerprints import MolecularFingerprints
from algorithms.qsar_ml import QSARModel


class FingerprintPipeline:
    def __init__(self, train_sdf, test_sdf, train_labels, test_labels, output_dir='output', log_file='log_1.txt'):
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
        print(f"Calculating {fingerprint_name} fingerprint for train dataset...")

        fingerprint_calculator = self.train_molecules.calculate_fingerprint
        x_train = fingerprint_calculator(fingerprint_name)

        train_df = pd.DataFrame(x_train)
        train_filename = os.path.join(self.output_dir, f"{fingerprint_name}_train.csv")
        train_df.to_csv(train_filename, index=False)
        print(f"Saved {fingerprint_name} fingerprint for train dataset to {train_filename}")

        print(f"Calculating {fingerprint_name} fingerprint for test dataset...")
        x_test = self.test_molecules.calculate_fingerprint(fingerprint_name)
        test_df = pd.DataFrame(x_test)
        test_filename = os.path.join(self.output_dir, f"{fingerprint_name}_test.csv")
        test_df.to_csv(test_filename, index=False)
        print(f"Saved {fingerprint_name} fingerprint for test dataset to {test_filename}")

        return True

    def load_fingerprint_data(self, fingerprint_name):
        """
        Load fingerprints for train and test datasets from CSV files.
        """
        print(f"Loading {fingerprint_name} fingerprint data from CSV files...")
        train_filename = os.path.join(self.output_dir, f"{fingerprint_name}_train.csv")
        test_filename = os.path.join(self.output_dir, f"{fingerprint_name}_test.csv")

        x_train = pd.read_csv(train_filename, header=0).values
        x_test = pd.read_csv(test_filename, header=0).values

        print(f"Loaded {fingerprint_name} fingerprint data for train and test.")
        return x_train, x_test

    def run_evaluation(self, fingerprint_name):
        """
        Load saved fingerprints and evaluate the model for the selected fingerprint.
        """
        print(f"Running evaluation for {fingerprint_name} fingerprint...")

        x_train, x_test = self.load_fingerprint_data(fingerprint_name)
        model = QSARModel(x_train, self.train_labels, x_test, self.test_labels)
        # model.create_tsne_visualization(output_path=os.path.join(self.output_dir, f'{fingerprint_name}_tsne.png'))
        results = model.run_models(use_test_set=False, fp_name=fingerprint_name, output_dir=self.output_dir)

        # results = None
        # print(f"Evaluation results for {fingerprint_name}: {results}")
        return results

    def calculate_all_fingerprints(self, fingerprints_to_run):
        """
        Calculate and save all fingerprints
        """
        print(f"Starting fingerprint calculation for {len(fingerprints_to_run)} fingerprints...")
        for fingerprint_name in fingerprints_to_run:
            print(f"\n--- Calculating and saving {fingerprint_name} ---")
            if not self.calculate_and_save_fingerprint(fingerprint_name):
                print(f"Skipping {fingerprint_name} due to error.")
        print("Fingerprint calculation complete.")

    def run_all_evaluations(self, fingerprints_to_run):
        """
        Run model evaluations for all fingerprints provided in the fingerprints_to_run.
        Collect results in a CSV file with columns: fingerprint, metrics.
        """
        print(f"Starting model evaluation for {len(fingerprints_to_run)} fingerprints...")
        results = []

        for fingerprint_name in fingerprints_to_run:
            try:
                print(f"\n--- Running evaluation for {fingerprint_name} ---")
                evaluation_results = self.run_evaluation(fingerprint_name)
                if evaluation_results is None:
                    continue

                # Collect results in structured format
                for model_name, metrics in evaluation_results.items():
                    results.append({
                        "Fingerprint": fingerprint_name,
                        "Model": model_name,
                        **metrics
                    })
            except Exception as e:
                print(f"Error evaluating fingerprint {fingerprint_name}: {e}")
                with open(self.log_file, 'a') as log:
                    log.write(f"Failed to evaluate {fingerprint_name}: {str(e)}\n")

        # Save results to a CSV file
        results_df = pd.DataFrame(results)
        results_filename = os.path.join(self.output_dir, "Report_exp_3_custom2.csv")
        results_df.to_csv(results_filename, index=False)
        print(f"Saved evaluation results to {results_filename}")

        print("Model evaluation complete.")
