import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict

def str_to_vec(dsc_str, dsc_num):
    tmp = defaultdict(int)
    for item in dsc_str.split(' '):
        key, value = item.split(':')
        tmp[int(key)] = int(value)

    # Generate the sorted vector with missing keys filled as 0
    vec = [tmp[i] for i in range(dsc_num)]
    return vec

def process_molecule(args):
    mol_idx, dsc_tmp, idx_tmp, dsc_num = args
    bag, idx_ = [], []
    for dsc_str, idx in zip(dsc_tmp, idx_tmp):
        if idx == mol_idx:
            bag.append(str_to_vec(dsc_str, dsc_num))
            idx_.append(idx)
    return np.array(bag).astype('uint8'), idx_[0]

def load_svm_data(fname):
    # Read descriptor data
    with open(fname) as f:
        dsc_tmp = [line.strip() for line in f]

    # Read molecule names
    with open(fname.replace('txt', 'rownames')) as f:
        mol_names = [line.strip() for line in f]

    idx_tmp = [name.split(':')[0] for name in mol_names]
    dsc_num = max(max(int(item.split(':')[0]) for item in line.split(' ')) for line in dsc_tmp)

    unique_mol_idxs = np.unique(idx_tmp)

    # Prepare arguments for parallel processing
    args = [(mol_idx, dsc_tmp, idx_tmp, dsc_num) for mol_idx in unique_mol_idxs]

    # Process each molecule in parallel
    with ProcessPoolExecutor() as executor:
        results = executor.map(process_molecule, args)

    # Collect results
    bags, idx = zip(*results)
    return np.array(bags), np.array(idx)




if __name__ == '__main__':
    base_dir = '../data/dataset_base/all_confs_pmapper_ds'
    fname = 'combined798_all_confs_pmapper_ds.txt'
    labels_fname = '../data/dataset_base/combined798_all_confs_labels.csv'
    csv_fname_labeled = os.path.join(base_dir, 'exp_3_descriptors_with_labels.csv')
    csv_fname = os.path.join(base_dir, 'exp_3_descriptors_without_labels.csv')

    dsc_fname = os.path.join(base_dir, fname)
    bags_fname = os.path.join(base_dir, 'bags_idx.pkl')

    # print("scv2df")
    # bags, idx = load_svm_data(dsc_fname)
    # print(f'There are {len(bags)} molecules encoded with {bags[0].shape[1]} descriptors')
    #
    # with open(bags_fname, 'wb') as f:
    #     pickle.dump((bags, idx), f)

    # Load the data from the pickle file
    with open(bags_fname, 'rb') as f:
        bags, idx = pickle.load(f)

    # Display the first 5 entries of idx and bags
    print("First 5 entries in idx:", idx[:5])
    print("First 5 entries in bags:")
    for bag in bags[:5]:
        for i in range(1, len(bag)):
            print(bag[i])

    # data = []
    #
    # for molecule_id, bag in zip(idx, bags):
    #     for instance_idx, instance in enumerate(bag):
    #         row = {'Molecule': f"{molecule_id}_{instance_idx + 1}",
    #                **{f'd_{i}': val for i, val in enumerate(instance)}}
    #         data.append(row)
    #
    # descriptors_df = pd.DataFrame(data)
    # descriptors_df.to_csv(csv_fname, index=False)
    #
    # # mapping Molecule to a label
    # labeled_df = pd.read_csv(labels_fname)
    # label_dict = dict(zip(labeled_df['Molecule'], labeled_df['label']))
    # descriptors_df['label'] = descriptors_df['Molecule'].map(label_dict)
    #
    # # manual test/train split
    # test_molecules_df = pd.read_csv('../data/dataset_base/test_ids.csv')
    # test_molecules = test_molecules_df['Molecule'].tolist()
    #
    # test_df = descriptors_df[descriptors_df['Molecule'].str.split('_').str[0].isin(test_molecules)]
    # test_df.to_csv(os.path.join(base_dir, '3DphFP_test_with_labels.csv'), index=False)
    # test_df_descriptors_only = test_df.drop(columns=['Molecule', 'label'])
    # test_df_descriptors_only.to_csv(os.path.join(base_dir, '3DphFP_test.csv'), index=False)
    #
    # train_df = descriptors_df[~descriptors_df['Molecule'].str.split('_').str[0].isin(test_molecules)]
    # train_df.to_csv(os.path.join(base_dir, '3DphFP_train_with_labels.csv'), index=False)
    # train_df_descriptors_only = train_df.drop(columns=['Molecule', 'label'])
    # train_df_descriptors_only.to_csv(os.path.join(base_dir, '3DphFP_train.csv'), index=False)
    #
    # descriptors_df.to_csv(csv_fname_labeled, index=False)
    # print(f"CSV saved to {csv_fname_labeled}")

