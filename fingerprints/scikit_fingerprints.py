import importlib
from rdkit import Chem
from skfp.preprocessing import MolFromSmilesTransformer, MolStandardizer
from sklearn.pipeline import make_pipeline

# fingerprints implemented in skfp
fingerprint_dimension = {
    "AtomPairFingerprint": "2D",
    "AutocorrFingerprint": "2D",
    "AvalonFingerprint": "2D",
    "E3FPFingerprint": "3D",
    "ECFPFingerprint": "2D",
    "ElectroShapeFingerprint": "3D",
    "ERGFingerprint": "2D",
    "EStateFingerprint": "2D",
    "GETAWAYFingerprint": "3D",
    "GhoseCrippenFingerprint": "2D",
    "KlekotaRothFingerprint": "2D",
    "LaggnerFingerprint": "2D",
    "LayeredFingerprint": "2D",
    "LingoFingerprint": "2D",
    "MACCSFingerprint": "2D",
    "MAPFingerprint": "2D",
    "MHFPFingerprint": "2D",
    "MordredFingerprint": "2D",  # 3D=false
    "MORSEFingerprint": "3D",
    "MQNsFingerprint": "2D",
    "PatternFingerprint": "2D",
    "PharmacophoreFingerprint": "3D",
    "PhysiochemicalPropertiesFingerprint": "2D",
    "PubChemFingerprint": "2D",
    "RDFFingerprint": "3D",
    "RDKitFingerprint": "2D",
    "SECFPFingerprint": "2D",
    "TopologicalTorsionFingerprint": "2D",
    "USRFingerprint": "3D",
    "USRCATFingerprint": "3D",
    "VSAFingerprint": "3D",
    "WHIMFingerprint": "3D"
}


class MolecularFingerprint:
    def __init__(self, sdf_file):
        self.sdf_file = sdf_file
        self.molecules = self._load_molecules()

    def _load_molecules(self):
        """
        Load molecules from an SDF.
        """
        supplier = Chem.SDMolSupplier(self.sdf_file)
        return [mol for mol in supplier if mol is not None]

    def _convert_to_smiles(self):
        """
        Convert 3D molecules to standardized SMILES.
        """
        smiles = []
        for mol in self.molecules:
            if mol is not None:
                smiles.append(Chem.MolToSmiles(mol))
        return smiles

    def calculate_fingerprint(self, fingerprint_name):
        """
        Dynamically calculate fingerprints based on the type (2D/3D).
        - 2D fingerprints: Convert molecules to SMILES.
        - 3D fingerprints: Use molecules directly from SDF.
        """
        module = importlib.import_module('skfp.fingerprints')
        fingerprint_class = getattr(module, fingerprint_name, None)

        if fingerprint_class is None:
            raise ValueError(f"Fingerprint class '{fingerprint_name}' not found.")

        fingerprint_type = fingerprint_dimension.get(fingerprint_name)

        if fingerprint_type == "2D":
            # For 2D fingerprints, convert to SMILES and standardize molecules
            smiles = self._convert_to_smiles()
            pipeline = make_pipeline(MolFromSmilesTransformer(), MolStandardizer(), fingerprint_class(n_jobs=-1))
            return pipeline.fit_transform(smiles)

        elif fingerprint_type == "3D":
            # For 3D fingerprints, directly use the Mol objects from SDF
            return fingerprint_class(n_jobs=-1).fit_transform(self.molecules)

        else:
            raise ValueError(f"Fingerprint '{fingerprint_name}' has an unknown type.")

    def available_fingerprints(self):
        """
        Return the list of available fingerprints from the scikit-fingerprints package.
        """
        return list(fingerprint_dimension.keys())
