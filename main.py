import pandas as pd
from pipeline.pipeline import FingerprintPipeline
from utils.fingerprint_definitions import fingerprint_dimensions
from rdkit.Chem import AllChem
from rdkit import Chem
# from sqlalchemy import create_engine, Column, Integer, String
# from sqlalchemy.orm import sessionmaker, declarative_base
# from utils.db import Base

if __name__ == "__main__":

    Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

    # engine = create_engine('sqlite:///fingerprint_pipeline.db')
    # Base.metadata.create_all(engine)  # Creates tables if they don't exist
    # Session = sessionmaker(bind=engine)
    # session = Session()



    train_sdf = "data/dataset/3d_qsar_exp_2/train_set_594_points.sdf"
    test_sdf = "data/dataset/3d_qsar_exp_2/test_set_199_points.sdf"

    train_labels = pd.read_csv("data/dataset/3d_qsar_exp_2/train_set_594_points.csv")['label'].values
    test_labels = pd.read_csv("data/dataset/3d_qsar_exp_2/test_set_199_points.csv")['label'].values

    # initialize and run the pipeline
    pipeline = FingerprintPipeline(train_sdf, test_sdf, train_labels, test_labels, output_dir="fingerprints_exp_2", log_file="log_2.txt")

    fingerprints_to_run = list(fingerprint_dimensions.keys())
    pipeline.calculate_all_fingerprints(fingerprints_to_run)
    pipeline.run_all_evaluations(fingerprints_to_run)