from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from pyzernike import ZernikeMoments
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Functions as defined above
def read_sdf(file_path):
    suppl = Chem.SDMolSupplier(file_path, removeHs=False)
    molecules = [mol for mol in suppl if mol is not None]
    return molecules

def prepare_molecule(mol):
    mol = Chem.AddHs(mol)
    if mol.GetNumConformers() == 0:
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        AllChem.UFFOptimizeMolecule(mol)
    return mol

def compute_zernike_descriptors(mol, l_max=8, n_max=8, voxel_grid_size=32):
    mol = prepare_molecule(mol)
    conf = mol.GetConformer()
    coords = conf.GetPositions()
    atomic_numbers = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()])
    coords -= np.mean(coords, axis=0)
    max_dist = np.max(np.linalg.norm(coords, axis=1))
    coords /= max_dist
    coords *= 0.9
    grid = np.zeros((voxel_grid_size, voxel_grid_size, voxel_grid_size), dtype=np.float32)
    indices = ((coords + 1) * (voxel_grid_size - 1) / 2).astype(int)
    grid[indices[:,0], indices[:,1], indices[:,2]] = 1.0
    zm = ZernikeMoments(radius=1.0, lmax=l_max, nmax=n_max)
    moments = zm.compute(grid)
    descriptors = np.absolute(moments)
    return descriptors

def compute_descriptors_for_dataset(molecules):
    descriptors_list = []
    for mol in molecules:
        descriptors = compute_zernike_descriptors(mol)
        descriptors_list.append(descriptors)
    return np.array(descriptors_list)

def extract_labels(molecules, label_property='Label'):
    labels = []
    for mol in molecules:
        label = mol.GetProp(label_property)
        labels.append(int(label))
    return np.array(labels)

# Main script
if __name__ == '__main__':
    # Read molecules
    train_molecules = read_sdf('train.sdf')
    test_molecules = read_sdf('test.sdf')

    # Compute descriptors
    X_train = compute_descriptors_for_dataset(train_molecules)
    X_test = compute_descriptors_for_dataset(test_molecules)

    # Extract labels
    y_train = extract_labels(train_molecules)
    y_test = extract_labels(test_molecules)

    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train Random Forest classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = rf_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')
    print('Classification Report:')
    print(classification_report(y_test, y_pred))
    print('Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred))
