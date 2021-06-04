import os
import pickle
from configparser import ConfigParser

import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import numpy as np
from tqdm import tqdm

# from dataset.ChexpertDataset import CheXpertDataSet
from dataset.ChexpertDataset import ChexDataset
from environment_setup import PROJECT_ROOT_DIR, read_config, threshold_dict, lambda_dict

# combine all column values into a list

label_columns = [
    'No Finding',
    'Enlarged Cardiomediastinum',
    'Cardiomegaly',
    'Lung Opacity',
    'Lung Lesion',
    'Edema',
    'Consolidation',
    'Pneumonia',
    'Atelectasis',
    'Pneumothorax',
    'Pleural Effusion',
    'Pleural Other',
    'Fracture',
    'Support Devices'
]



def build_cooccurence(train_data_items, LABELS, uncertainty_labels):
    complete_label_list = []
    threshold = threshold_dict[uncertainty_labels]
    labda = lambda_dict[uncertainty_labels]
    # Iterate over all the labels present in the form of a list of tuple
    for _, label in tqdm(train_data_items):
        label_arr_np = np.array(label)
        # how to handle uncertain labels? for ex: u-positive or u-negative
        if uncertainty_labels == 'negative':
            label_arr_np[label_arr_np == -1] = 0
        elif uncertainty_labels == 'positive':
            label_arr_np[label_arr_np == -1] = 1
        elif uncertainty_labels == 'multiclass':
            label_arr_np[label_arr_np == -1] = 2
        # Now we use these labels as our processed one
        complete_label_list.append(label_arr_np)
    labels_mat = np.stack(complete_label_list)
    co_occur_mat = np.zeros((len(LABELS), len(LABELS)))

    for i in range(len(LABELS)):
        for j in range(len(LABELS)):

            if i == j:
                # along the diagonal, save its frequency
                co_occur_mat[i, j] = np.sum(labels_mat[:, i])
            else:
                # temp only has the columns for 'i' and 'j' labels
                a = np.zeros_like(labels_mat)
                a[:, i] = labels_mat[:, i]
                a[:, j] = labels_mat[:, j]

                # is np.sum(temp, -1) = 2, i and j co-occur
                b = np.sum(a, -1)
                freq_ij = np.sum(b == 2)
                # Because of symmetry of the matrix
                co_occur_mat[i, j] = freq_ij
    # normalise based on causality
    co_occur_directional = (co_occur_mat/np.diag(co_occur_mat).reshape(-1,1))
    # First threshold
    co_occur_directional[co_occur_directional < threshold] = 0
    co_occur_directional[co_occur_directional >= threshold] = 1
    # Smoothing effects as suggested
    co_occur_directional = co_occur_directional * labda / (co_occur_directional.sum(0, keepdims=True) + 1e-6)
    for i in range(14):
        co_occur_directional[i, i] += 1
    # Now we need to normalize
    # col_sum = np.sum(co_occur_directional, axis=0)
    # co_occur_directional = labda * co_occur_directional / (col_sum.reshape(-1, 1) + 1e-6)

    return co_occur_mat, co_occur_directional


def data_loader_dict(uncertainty_labels='positive', batch_size=64, num_workers=4, build_grph=True):
    pd.set_option('mode.chained_assignment', None)
    # load train list
    all_data = pd.read_csv(os.path.join(PROJECT_ROOT_DIR, 'dataset', 'CheXpert-v1.0-small', 'train.csv'))
    # Drop the non required columns
    relevant_data = all_data.drop(['Sex', 'Age', 'Frontal/Lateral', 'AP/PA'], axis=1)
    # Fill all missing labels with 0.
    relevant_data = relevant_data.fillna(0)
    print('[*] total train patients: ', len(relevant_data))
    #  All rows with patient having uncertainty label
    # https://www.statology.org/pandas-find-value-any-column/
    uncertain_index = relevant_data.isin([-1]).any(axis=1)
    uncertain_data = relevant_data[uncertain_index]
    certain_data = relevant_data[uncertain_index.apply(lambda x: not x)]
    # Let us combine all labels into a single column.
    # It seems that applying the next 4 steps can be avoided if we apply it to the `relevant_data` itself.
    # However, doing that would cause an error while computing `uncertain_index`. The error is
    # SystemError: <built-in method view of numpy.ndarray object at 0x7f4e8a1de260> returned a result with an error set
    # Hence do the extra work as a work around.
    uncertain_data['Labels'] = uncertain_data[label_columns].values.tolist()
    certain_data['Labels'] = certain_data[label_columns].values.tolist()
    # Drop the remaining columns from the dataframe
    uncertain_data = uncertain_data.drop(label_columns, axis=1)
    certain_data = certain_data.drop(label_columns, axis=1)

    # separate uncertain and certain patients. split 'certain' into train + test + val. Add 'uncertain' to train only.
    data_dict_uncertain = {row['Path']: row['Labels'] for idx, row in uncertain_data.iterrows()}
    data_dict_certain = {row['Path']: row['Labels'] for idx, row in certain_data.iterrows()}

    # split 'certain' into train + test + val. Add 'uncertain' to train only.

    tr_keys = list(data_dict_certain.keys())
    np.random.RandomState(42).shuffle(tr_keys)

    val_size = int(0.1 * len(relevant_data))
    test_size = int(0.2 * len(relevant_data))

    te_keys, va_keys, tr_keys = np.split(tr_keys, [test_size, test_size + val_size])

    data_items_tr = [(key,data_dict_certain[key]) for key in tr_keys]

    data_items_va = [(key,data_dict_certain[key]) for key in va_keys]

    data_items_te = [(key,data_dict_certain[key]) for key in te_keys]

    # now add 'uncertain' to train only.
    tr_keys = list(data_dict_uncertain.keys())
    temp = [(key, data_dict_uncertain[key]) for key in tr_keys]
    data_items_tr += temp

    data_items = {}

    data_items['train'] = data_items_tr
    data_items['valid'] = data_items_va
    data_items['test'] = data_items_te
    data_items['train_val'] = data_items_te[:len(data_items_va)]

    # [!] For overfitting!
    # data_items['train'] = data_items_tr[::100]
    # data_items['valid'] = data_items_va[::10]
    # data_items['test']  = data_items_te[::10]

    def get_data_loader(data_items, split, batch_size, augment, preprocess, shuffle=True):
        data_items = data_items[split]

        ds = ChexDataset(data_items, augment=augment, preprocess=preprocess, uncertainty_labels=uncertainty_labels)

        print('Loaded {} dataset with {} IDs.'.format(split, ds.__len__()))

        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=max(0, num_workers)
        )

    data_iter = {
        'train': get_data_loader(data_items=data_items, split='train', batch_size=batch_size, augment=True,
                                 preprocess=True, shuffle=True),
        'valid': get_data_loader(data_items=data_items, split='valid', batch_size=batch_size, augment=False,
                                 preprocess=True, shuffle=False),
        'test': get_data_loader(data_items=data_items, split='test', batch_size=batch_size, augment=False,
                                preprocess=True, shuffle=False),
        'train_val': get_data_loader(data_items=data_items, split='train_val', batch_size=batch_size, augment=False,
                                preprocess=True, shuffle=False),
    }

    # The training set needs to be used for creating the graph in our baseline method. Hence, we save it here.
    if build_grph:
        co_occur_mat, co_occur_directional = build_cooccurence(train_data_items=data_items['train'],
                                                               LABELS=label_columns,
                                                               uncertainty_labels=uncertainty_labels)
        np.save(os.path.join(PROJECT_ROOT_DIR, "models", "graph_base", "co_occur_mat.npy"), co_occur_mat)
        np.save(os.path.join(PROJECT_ROOT_DIR, "models", "graph_base", "co_occur_directional.npy"), co_occur_directional)

    print('[*] Finished everything concerned with data loading!')
    return data_iter


if __name__ == '__main__':
    parser = read_config()
    uncertainty_labels = parser['data'].get('uncertainty_labels')
    data = data_loader_dict(uncertainty_labels=uncertainty_labels, batch_size=64, num_workers=4)
    train = data['train']
    for img, label in train:
        print(img.shape)
        print(label.shape)
        print(label.dtype)
        img = img[0].squeeze()
        plt.imshow(img)
        plt.show()
        break