import pandas as pd

fp_test_2d = pd.read_csv('ECFPFingerprint_test.csv', header=0)
fp_train_2d = pd.read_csv('ECFPFingerprint_train.csv', header=0)

fp_test_3d = pd.read_csv('RDFFingerprint_test.csv', header=0)
fp_train_3d = pd.read_csv('RDFFingerprint_train.csv', header=0)

concated_test = pd.concat([fp_test_2d, fp_test_3d], axis=1)
concated_test.to_csv('ECFPRDF_test.csv', index=False)
concated_train = pd.concat([fp_train_2d, fp_train_3d], axis=1)
concated_train.to_csv('ECFPRDF_train.csv', index=False)