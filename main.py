import pandas as pd
from pipeline.pipeline import FingerprintPipeline
from utils.fingerprint_definitions import fingerprint_dimensions
from rdkit.Chem import AllChem
from rdkit import Chem
from algorithms.qsar_ml import QSARModel
import os
# from sqlalchemy import create_engine, Column, Integer, String
# from sqlalchemy.orm import sessionmaker, declarative_base
# from utils.db import Base



def calc_pmapper_descriptors():
    basedir = '/home/vfastovskii/Desktop/mil_rev_last_november_2/data/dataset_base/3d_qsar_exp_2/exp2_last_try_rm_005'

    x_train = pd.read_csv(os.path.join(basedir, '3DphFP_train.csv'), header=0)
    y_train = pd.read_csv(os.path.join(basedir, '3DphFP_train_with_labels.csv'), usecols=["label"], header=0)['label'].values
    x_test = pd.read_csv(os.path.join(basedir, '3DphFP_test.csv'))
    y_test = pd.read_csv(os.path.join(basedir, '3DphFP_test_with_labels.csv'), usecols=["label"], header=0)['label'].values

    qsar_model = QSARModel(x_train, y_train, x_test, y_test)
    balanced_accuracy = qsar_model.run_random_forest()
    with open(os.path.join(basedir, 'exp2_pmapper_results.txt'), "w") as file:
        file.write(f"Balanced Accuracy: {balanced_accuracy}\n")






if __name__ == "__main__":

    Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

    # engine = create_engine('sqlite:///fingerprint_pipeline.db')
    # Base.metadata.create_all(engine)  # Creates tables if they don't exist
    # Session = sessionmaker(bind=engine)
    # session = Session()


    # train_sdf = "data/dataset_base/3d_qsar_exp_2/train_set_594_points.sdf"
    # test_sdf = "data/dataset_base/3d_qsar_exp_2/test_set_199_points.sdf"
    #
    # train_labels = pd.read_csv("data/dataset_base/3d_qsar_exp_2/train_set_594_points.csv", header=0)['label'].values
    # test_labels = pd.read_csv("data/dataset_base/3d_qsar_exp_2/test_set_199_points.csv", header=0)['label'].values
    #
    # # initialize and run the pipeline
    # pipeline = FingerprintPipeline(train_sdf, test_sdf, train_labels, test_labels, output_dir="fingerprints_exp_2", log_file="log_2.txt")
    #
    # fingerprints_to_run = list(fingerprint_dimensions.keys())
    # pipeline.calculate_all_fingerprints(fingerprints_to_run)

    # add an external descriptors
    # fingerprints_to_run.append('MOE3D')
    # pipeline.run_all_evaluations(fingerprints_to_run)

    calc_pmapper_descriptors()