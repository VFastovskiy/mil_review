from rdkit import Chem
from skfp.preprocessing import MolFromSmilesTransformer, MolStandardizer
from utils.fingerprint_definitions import fingerprint_dimensions
from sklearn.pipeline import make_pipeline


class MolecularFingerprints:
    def __init__(self, sdf_file):
        print(f"Initializing MolecularFingerprints with SDF file: {sdf_file}")
        self.sdf_file = sdf_file
        self.molecules = self._load_molecules()
        print(f"Loaded {len(self.molecules)} molecules from {sdf_file}")

    def _load_molecules(self):
        """
        Load molecules from an SDF file using RDKit.
        """
        print("Loading molecules from SDF file...")
        supplier = Chem.SDMolSupplier(self.sdf_file)
        return [mol for mol in supplier if mol is not None]

    def _convert_to_smiles(self):
        """
        Convert 3D molecules to standardized SMILES using RDKit.
        """
        print("Converting molecules to SMILES...")
        smiles = []
        for mol in self.molecules:
            if mol is not None:
                smiles.append(Chem.MolToSmiles(mol))
        print(f"Converted {len(smiles)} molecules to SMILES.")
        return smiles

    def calculate_fingerprint(self, fingerprint_name):
        """
        Dynamically calculate fingerprints based on the type (2D/3D).
        - 2D fingerprints: Convert molecules to SMILES.
        - 3D fingerprints: Use molecules directly from SDF.
        """
        print(f"Calculating {fingerprint_name} fingerprint...")

        # retrieve fingerprint dimension and class
        fingerprint_data = fingerprint_dimensions.get(fingerprint_name)

        if fingerprint_data is None:
            raise ValueError(f"Fingerprint class '{fingerprint_name}' not found.")

        dimension, fingerprint_class = fingerprint_data

        if dimension == "2D":
            print(f"Fingerprint {fingerprint_name} is 2D. Converting to SMILES...")
            # for 2D fingerprints, mol converted into SMILES and standardized
            smiles = self._convert_to_smiles()
            pipeline = make_pipeline(MolFromSmilesTransformer(), MolStandardizer(), fingerprint_class(n_jobs=-1))
            fingerprints = pipeline.fit_transform(smiles)
            print(f"Calculated {fingerprint_name} fingerprints for {len(smiles)} molecules.")
            return fingerprints

        elif dimension == "3D":
            for mol in self.molecules:
                if mol is not None:
                    full_id = mol.GetProp('full_id')
                    try:
                        conf_id = int(full_id.split('_')[-1])
                        for conf in mol.GetConformers():
                            conf.SetId(conf_id)
                    except ValueError:
                        print(f"Warning: Could not convert conf_id from full_id '{full_id}' for molecule.")
            print(f"Fingerprint {fingerprint_name} is 3D. Using Mol objects directly...")
            # For 3D fingerprints, directly use the Mol objects from SDF
            fingerprints = fingerprint_class(n_jobs=-1).fit_transform(self.molecules)
            print(f"Calculated {fingerprint_name} fingerprints for {len(self.molecules)} molecules.")
            return fingerprints

        else:
            raise ValueError(f"Fingerprint '{fingerprint_name}' has an unknown dimension.")

    def available_fingerprints(self):
        """
        Return the list of available fingerprints from the scikit-fingerprints package.
        """
        print("Returning available fingerprints...")
        return list(fingerprint_dimensions.keys())


