import os
import pickle

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import numpy as np

from dataset.mimic.MimicDataset import MimicDataset
from dataset.mimic.mimic_utils import extra_train_labels
from environment_setup import PROJECT_ROOT_DIR

data_dir = os.path.join(PROJECT_ROOT_DIR, "dataset", "mimic", "physionet.org", "files", "mimic-cxr-jpg", "2.0.0")

# Makes use of different label columns in this case compared to baseline. It may cause some issues and we should take note
LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "No Finding",
    "Pleural Effusion",
    "Pleural Other",
    "Pneumonia",
    "Pneumothorax",
    "Support Devices"
]

LABELS.extend(extra_train_labels)


def build_mimic_graph(split):
    complete_label_list = []
    extra_labels_dict = pickle.load(open(os.path.join(PROJECT_ROOT_DIR, "dataset", "mimic", "extra_label.pkl"), "rb"))
    ds = MimicDataset(split=split, augment=False, preprocess=False)
    if os.path.exists(os.path.join(PROJECT_ROOT_DIR, "models", "graph_extra_labels", "label_mat.npy")):
        labels_mat = np.load(os.path.join(PROJECT_ROOT_DIR, "models", "graph_extra_labels", "label_mat.npy"))
    else:
        for idx in tqdm(range(len(ds))):
            extra_labels_key = ds[idx][3]
            extra_train_labels = np.asarray(extra_labels_dict[extra_labels_key])
            # Add these extra labels at the end of the original labels
            combined_label = np.append(ds[idx][2].numpy(), extra_train_labels)
            complete_label_list.append(combined_label)
        labels_mat = np.stack(complete_label_list)
    co_occur_mat = np.zeros((len(LABELS), len(LABELS)))
    num_samples = labels_mat.shape[0]
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            if i == j:
                # along the diagonal save their count
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

    # We need to convert the co-occurrence matrix into prob matrix
    prob_mat = co_occur_mat / num_samples  # eqn 5
    prob_i = np.diag(co_occur_mat).reshape(1, -1) / num_samples  # row vector, eqn 6
    prob_j = np.diag(co_occur_mat).reshape(-1, 1) / num_samples  # column vector, eqn 6
    pmi = np.log(prob_mat / (prob_i * prob_j))
    # Remove the edges that occur less than expected value
    pmi[pmi < 0] = 0
    # Next we enter diagonal values
    for i in range((len(LABELS))):
        pmi[i, i] = 1  # eqn 3

    return co_occur_mat, pmi, labels_mat


def get_mimic_dataloader_dict(num_workers, batch_size):
    # Custom collate function since we want minimal change to the original framework
    def collate_fn(data):
        # It is returning a list of tuples
        list_data, labels, extra_label_key = zip(*data)
        labels = torch.stack(labels)
        img1, img2 = zip(*list_data)
        images_list = [torch.stack(img1), torch.stack(img2)]
        return images_list, labels

    # An inner function to make our lives easier
    def get_data_loader(split, batch_size, augment, preprocess, shuffle=True):
        ds = MimicDataset(split=split, augment=augment, preprocess=preprocess)

        print('Loaded {} dataset with {} IDs.'.format(split, ds.__len__()))

        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=max(0, num_workers),
            collate_fn=collate_fn
        )

    data_iter = {
        'train': get_data_loader(split='train', batch_size=batch_size, augment=True,
                                 preprocess=True, shuffle=True),
        'valid': get_data_loader(split='valid', batch_size=batch_size, augment=False,
                                 preprocess=True, shuffle=False),
        'test': get_data_loader(split='test', batch_size=batch_size, augment=False,
                                preprocess=True, shuffle=False),
        'train_val': get_data_loader(split='train_val', batch_size=batch_size, augment=False,
                                     preprocess=True, shuffle=False),
    }

    # TODO: Remove this
    # co_occur_mat, pmi, label_mat = build_mimic_graph(split='train')
    # The training set needs to be used for creating the graph in our baseline method. Hence, we save it here.
    if not os.path.exists(os.path.join(PROJECT_ROOT_DIR, "models", "graph_extra_labels", "pmi_mat.npy")):
        co_occur_mat, pmi, label_mat = build_mimic_graph(split='train')
        np.save(os.path.join(PROJECT_ROOT_DIR, "models", "graph_extra_labels", "co_occur.npy"), co_occur_mat)
        np.save(os.path.join(PROJECT_ROOT_DIR, "models", "graph_extra_labels", "pmi_mat.npy"), pmi)
        np.save(os.path.join(PROJECT_ROOT_DIR, "models", "graph_extra_labels", "label_mat.npy"), label_mat)

    print('[*] Finished everything concerned with data loading!')
    return data_iter


if __name__ == '__main__':
    print(len(LABELS))
    d_dict = get_mimic_dataloader_dict(num_workers=4, batch_size=16)
    for imgs, label in d_dict['train']:
        img_l, img_f = imgs
        print(img_l.shape)
        print(img_f.shape)
        print(label.shape)
        break
