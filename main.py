import pandas as pd
from pipeline.pipeline import FingerprintPipeline
from utils.fingerprint_definitions import fingerprint_dimensions
from rdkit.Chem import AllChem
from rdkit import Chem

if __name__ == "__main__":

    Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

    train_sdf = "data/dataset/3d_qsar_experiment/train_set_598_points.sdf"
    test_sdf = "data/dataset/3d_qsar_experiment/test_set_200_points.sdf"

    train_labels = pd.read_csv("data/dataset/3d_qsar_experiment/train_set_598_points.csv")['label'].values
    test_labels = pd.read_csv("data/dataset/3d_qsar_experiment/test_set_200_points.csv")['label'].values

    # initialize and run the pipeline
    pipeline = FingerprintPipeline(train_sdf, test_sdf, train_labels, test_labels, output_dir="output_fingerprints", log_file="log.txt")

    fingerprints_to_run = list(fingerprint_dimensions.keys())
    pipeline.calculate_all_fingerprints(fingerprints_to_run)
    pipeline.run_all_evaluations(fingerprints_to_run)