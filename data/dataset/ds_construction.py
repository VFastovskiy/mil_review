from rdkit import Chem
from rdkit.Chem import SDWriter

def parse_sdf(file_path, active_only=True):
    suppl = Chem.SDMolSupplier(file_path)
    mol_dict = {}  # dictionary to store the lowest energy conformer for each mol_id

    for mol in suppl:
        if mol is None:
            continue

        mol_id = mol.GetProp('mol_id')  # Get mol_id (same for all conformers of a molecule)
        full_id = mol.GetProp('full_id')  # Get full_id (unique for each conformer)
        energy = float(mol.GetProp('energy'))
        act_conf = mol.GetProp('act_conf')  # Get act_conf (1 - active, 0 - inactive)

        # For actives, only consider active conformers (act_conf == 1)
        if active_only and act_conf != '1':
            continue

        if mol_id not in mol_dict or energy < mol_dict[mol_id]['energy']:
            mol_dict[mol_id] = {'mol': mol, 'energy': energy, 'full_id': full_id, 'act_conf': act_conf}

    return mol_dict


def write_lowest_energy_sdf(mol_dict, output_file):
    with SDWriter(output_file) as writer:
        for mol_data in mol_dict.values():
            writer.write(mol_data['mol'])


def main():
    decoy_flag = False
    input_sdf = 'decoys399_all_confs.sdf' if decoy_flag else 'actives399_all_confs.sdf'
    output_sdf = '3d_qsar_experiment/decoys399_lowest_energy_3d_qsar_experiment.sdf' if decoy_flag else '3d_qsar_experiment/actives399_lowest_energy_3d_qsar_experiment.sdf'

    mol_dict = parse_sdf(input_sdf, active_only=not decoy_flag)

    unique_mol_count = len(mol_dict)
    print(f"Number of unique {'decoys' if decoy_flag else 'actives'}: {unique_mol_count}")

    write_lowest_energy_sdf(mol_dict, output_sdf)

if __name__ == "__main__":
    main()

