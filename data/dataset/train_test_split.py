import pandas as pd
from rdkit import Chem
from rdkit.Chem import SDWriter


def parse_sdf(file_path):
    """
    Parse the SDF file and return a dictionary with full_id as the key and molecule as the value.
    """
    suppl = Chem.SDMolSupplier(file_path)
    mol_dict = {}

    for mol in suppl:
        if mol is None:
            continue

        full_id = mol.GetProp('full_id')
        mol_id = mol.GetProp('mol_id')
        act_mol = mol.GetProp('act_mol')  # get act_mol (label)

        mol_dict[full_id] = {'mol': mol, 'mol_id': mol_id, 'act_mol': act_mol}

    return mol_dict


def write_sdf(mol_dict, output_file):
    with SDWriter(output_file) as writer:
        for mol_data in mol_dict.values():
            writer.write(mol_data['mol'])


def create_csv(mol_dict, output_csv):
    data = [{'Molecule': mol_data['mol_id'], 'full_id': full_id, 'label': mol_data['act_mol']}
            for full_id, mol_data in mol_dict.items()]

    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"CSV written to {output_csv}")


def split_train_test(combined_dict, test_ids):
    train_dict = {}
    test_dict = {}

    for full_id, mol_data in combined_dict.items():
        if full_id in test_ids:
            test_dict[full_id] = mol_data
        else:
            train_dict[full_id] = mol_data

    return train_dict, test_dict


def main():
    test_csv = 'test_ids.csv'
    test_data = pd.read_csv(test_csv)

    # Create full_id by combining Molecule and Conformer_Idx columns with '_'
    test_data['full_id'] = test_data['Molecule'] + '_' + test_data['Conformer_Idx'].astype(str)
    csv_test_ids = set(test_data['full_id'].unique())
    combined_input_sdf = '3d_qsar_experiment/combined798_3d_qsar_experiment.sdf'
    combined_dict = parse_sdf(combined_input_sdf)

    # Filter only full_ids present in the combined dataset
    test_ids = csv_test_ids.intersection(combined_dict.keys())
    print(f"Test set contains {len(test_ids)} molecules")

    # Split into train and test sets
    train_dict, test_dict = split_train_test(combined_dict, test_ids)
    train_output_sdf = '3d_qsar_experiment/train_set_598_points.sdf'
    test_output_sdf = '3d_qsar_experiment/test_set_200_points.sdf'
    write_sdf(train_dict, train_output_sdf)
    write_sdf(test_dict, test_output_sdf)

    train_output_csv = '3d_qsar_experiment/train_set_598_points.csv'
    test_output_csv = '3d_qsar_experiment/test_set_200_points.csv'
    create_csv(train_dict, train_output_csv)
    create_csv(test_dict, test_output_csv)

    print(f"Train set SDF written to {train_output_sdf}")
    print(f"Test set SDF written to {test_output_sdf}")
    print(f"Train CSV written to {train_output_csv}")
    print(f"Test CSV written to {test_output_csv}")
    print(f"Number of full_ids in test set: {len(test_dict)}")
    print(f"Number of full_ids in train set: {len(train_dict)}")


if __name__ == "__main__":
    main()

