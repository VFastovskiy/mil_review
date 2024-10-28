import  pandas as pd
import os


def txt2csv(file_list, output_dir):
    for file_name in file_list:
        df = pd.read_csv(file_name, delimiter=',')
        columns_to_remove = ['mol.smiles', 'energy', 'full_id', 'mol_id', 'act_mol', 'act_conf']
        df = df.drop(columns=columns_to_remove, errors='ignore')
        df.dropna(axis='columns', how='any', inplace=True)
        base_name = os.path.basename(file_name).replace('.txt', '.csv')
        output_file = os.path.join(output_dir, base_name)
        df.to_csv(output_file, index=False)


if __name__ == '__main__':
    output_dir = '../fingerprints_exp_1'
    file_list = ['../fingerprints_exp_1/moe_3d_200p_test.txt', '../fingerprints_exp_1/moe_3d_598p_train.txt']
    txt2csv(file_list, output_dir)