import pandas as pd
from rdkit import Chem
from rdkit.Chem import SDWriter

def parse_sdf(file_path):
    suppl = Chem.SDMolSupplier(file_path)
    mol_dict = {}

    for mol in suppl:
        if mol is None:
            continue

        full_id = mol.GetProp('full_id')
        mol_dict[full_id] = mol

    return mol_dict

def write_sdf(mol_dict, output_file):
    with SDWriter(output_file) as writer:
        for mol in mol_dict.values():
            writer.write(mol)

def split_train_test(combined_dict, test_ids):
    train_dict = {}
    test_dict = {}

    for full_id, mol in combined_dict.items():
        if full_id in test_ids:
            test_dict[full_id] = mol
        else:
            train_dict[full_id] = mol

    return train_dict, test_dict

def main():
    test_csv = 'test_ids.csv'
    test_data = pd.read_csv(test_csv)

    # Create full_id by combining Molecule and Conformer_Idx columns with '_'
    test_data['full_id'] = test_data['Molecule'] + '_' + test_data['Conformer_Idx'].astype(str)
    csv_test_ids = set(test_data['full_id'].unique())

    combined_input_sdf = '3d_qsar_experiment/combined798_3d_qsar_experiment.sdf'
    combined_dict = parse_sdf(combined_input_sdf)

    test_ids = csv_test_ids.intersection(combined_dict.keys())
    print("Test set contains {} molecules".format(len(test_ids)))
    train_dict, test_dict = split_train_test(combined_dict, test_ids)

    train_output_sdf = '3d_qsar_experiment/train_set.sdf'
    test_output_sdf = '3d_qsar_experiment/test_set.sdf'
    write_sdf(train_dict, train_output_sdf)
    write_sdf(test_dict, test_output_sdf)

    print(f"Train set written to {train_output_sdf}")
    print(f"Test set written to {test_output_sdf}")
    print(f"Number of full_ids in test set: {len(test_dict)}")
    print(f"Number of full_ids in train set: {len(train_dict)}")

if __name__ == "__main__":
    main()
