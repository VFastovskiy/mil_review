import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_svm_data(fname):
    def str_to_vec(dsc_str, dsc_num):

        tmp = {}
        for i in dsc_str.split(' '):
            tmp[int(i.split(':')[0])] = int(i.split(':')[1])
        #
        tmp_sorted = {}
        for i in range(dsc_num):
            tmp_sorted[i] = tmp.get(i, 0)
        vec = list(tmp_sorted.values())

        return vec

    with open(fname) as f:
        dsc_tmp = [i.strip() for i in f.readlines()]

    with open(fname.replace('txt', 'rownames')) as f:
        mol_names = [i.strip() for i in f.readlines()]

    # labels_tmp = [float(i.split(':')[1]) for i in mol_names]
    idx_tmp = [i.split(':')[0] for i in mol_names]
    dsc_num = max([max([int(j.split(':')[0]) for j in i.strip().split(' ')]) for i in dsc_tmp])

    bags, idx = [], []
    for mol_idx in list(np.unique(idx_tmp)):
        bag, idx_ = [], []
        for dsc_str, i in zip(dsc_tmp, idx_tmp):
            if i == mol_idx:
                bag.append(str_to_vec(dsc_str, dsc_num))
                idx_.append(i)

        bags.append(np.array(bag).astype('uint8'))
        idx.append(idx_[0])

    return np.array(bags), np.array(idx)


if __name__ == '__main__':
    base_dir = '../data/dataset_base/3d_qsar_exp_1/exp1_last_try_rm_005'
    fname = 'exp_1_last_try_rm_005.txt'
    labels_fname = '../data/dataset_base/3d_qsar_exp_1/combined798_3d_qsar_experiment.csv'
    csv_fname_labeled = os.path.join(base_dir, 'exp_1_descriptors_with_labels.csv')
    csv_fname = os.path.join(base_dir, 'exp_1_descriptors_without_labels.csv')

    dsc_fname = os.path.join(base_dir, fname)

    bags, idx = load_svm_data(dsc_fname)
    print(f'There are {len(bags)} molecules encoded with {bags[0].shape[1]} descriptors')

    descriptors = pd.DataFrame([bag[0] for bag in bags if bag.shape[0] > 0])
    descriptors.to_csv(csv_fname, index=False)
    descriptors['Molecule'] = idx

    # mapping Molecule to a label
    labeled_df = pd.read_csv(labels_fname)
    label_dict = dict(zip(labeled_df['Molecule'], labeled_df['label']))
    descriptors['label'] = descriptors['Molecule'].map(label_dict)

    # manual test/train split
    test_molecules_df = pd.read_csv('../data/dataset_base/test_ids.csv')
    test_molecules = set(test_molecules_df['Molecule'])

    test_df = descriptors[descriptors['Molecule'].isin(test_molecules)]
    test_df = test_df.drop(columns=['Molecule', 'label'])
    test_df.to_csv(os.path.join(base_dir, '3DphFP_test.csv'), index=False)

    train_df = descriptors[~descriptors['Molecule'].isin(test_molecules)]
    train_df = train_df.drop(columns=['Molecule', 'label'])
    train_df.to_csv(os.path.join(base_dir, '3DphFP_train.csv'), index=False)

    descriptors.to_csv(csv_fname_labeled, index=False)
    print(f"CSV saved to {csv_fname_labeled}")

