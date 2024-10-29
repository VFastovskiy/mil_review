import os
import csv
from rdkit import Chem
from pmapper.pharmacophore import Pharmacophore as P


def load_mols(sdf_file):
    print("Loading molecules...")
    suppl = Chem.SDMolSupplier(sdf_file)
    return [mol for mol in suppl if mol is not None]


def generate_ph_fp(molecules):
    fingerprints = []
    print("Calculating pharmacophore fingerprints...")
    for mol in molecules:
        p = P()
        p.load_from_mol(mol)
        fp = list(p.get_fp(min_features=4, max_features=4))
        fingerprints.append(fp)
    print(f"Calculated FP: {len(fingerprints)}")
    return fingerprints


def save_fingerprints_to_csv(fingerprints, output_file):
    print("Saving fingerprints...")
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        for fp in fingerprints:
            writer.writerow(fp)
    print(f"Fingerprints saved to {output_file}")


def pipeline(train_file, test_file, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train_molecules = load_mols(train_file)
    test_molecules = load_mols(test_file)

    save_fingerprints_to_csv(generate_ph_fp(train_molecules), os.path.join(output_dir, "train_fingerprints.csv"))
    save_fingerprints_to_csv(generate_ph_fp(test_molecules), os.path.join(output_dir, "test_fingerprints.csv"))


if __name__ == '__main__':
    train_file = "../data/dataset/3d_qsar_exp_1/train_set_598_points.sdf"
    test_file = "../data/dataset/3d_qsar_exp_1/test_set_200_points.sdf"
    output_dir = "pharmacophore_fingerprints"

    pipeline(train_file, test_file, output_dir)
