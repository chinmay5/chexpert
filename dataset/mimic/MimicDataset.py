import pickle

import os
from collections import Counter

import PIL
import matplotlib.pyplot as plt
import pandas as pd

from PIL import Image

import numpy as np

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from environment_setup import PROJECT_ROOT_DIR, read_config

img_base_path = os.path.join(PROJECT_ROOT_DIR, "dataset", "mimic", "physionet.org", "files", "mimic-cxr-jpg", "2.0.0", "files")

img_root_dir = os.path.join(PROJECT_ROOT_DIR, "dataset", "mimic", "images")
os.makedirs(img_root_dir, exist_ok=True)


view_file = pd.read_csv(os.path.join(PROJECT_ROOT_DIR, "dataset", "mimic", "physionet.org", "files", "mimic-cxr-jpg", "2.0.0", "mimic-cxr-2.0.0-metadata.csv"))



class MimicDataset(Dataset):

    def __init__(
            self,
            split,
            augment=False,
            preprocess=False,
    ):

        """
        uncertainity_
        mode: 'positive', 'negative', 'ignore'. Denote U-ignore or U-positive or U-negative
        """
        self.split = split
        self.preprocess = preprocess
        self.augment = augment
        self.parse_dictionary()
        self.img_size = read_config()['data'].getint('img_shape')


    def parse_dictionary(self):
        self.data = []
        data_dict = pickle.load(open(os.path.join(PROJECT_ROOT_DIR, "dataset", "temp", f"{self.split}.pkl"), "rb"))
        # patient_id {
        #           study_id{
        #                   [img1, img2, target_labels]
        #                   }
        #             }
        for patient_id, study_dict in tqdm(data_dict.items()):
            for study_id, data_items in study_dict.copy().items():
                # We can iterate over the specific subset with values p13,p15...
                patient_root_id = f"p{str(patient_id)[:2]}"
                self.data.append(
                    [self.create_image_path(patient_id=patient_id, study_id=study_id, filename=data_items[0]),
                     self.create_image_path(patient_id=patient_id, study_id=study_id, filename=data_items[1]),
                     data_items[2]])


    def create_image_path(self, patient_id, study_id, filename):
        return os.path.join(f"p{str(patient_id)[:2]}", f"p{str(patient_id)}", f"s{str(study_id)}", filename)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        # Add augmentations later on
        img1, img2, img_label = self.data[item][0], self.data[item][1], self.data[item][2]
        labels = np.array(img_label)
        # We use frontal image as the first one and lateral as the second else we might have consistency issues
        # while training the network.
        if 'FRONTAL' in img1:
            img_f, img_l = img1, img2
        else:
            img_f, img_l = img2, img1

        save_path = os.path.join(img_root_dir, img_f[:img_f.rfind("/")])
        image_f = np.load(os.path.join(save_path, img_f[img_f.rfind("/") + 1:img_f.rfind(".jpg")] + ".npy"))
        image_l = np.load(os.path.join(save_path, img_l[img_l.rfind("/") + 1:img_l.rfind(".jpg")] + ".npy"))
        # We have stored the extra labels in the format , KEY -> 'p10000032/s50414267', value -> [1,0,1,....]
        extra_label_key = img_f[4:img_f.rfind("/")]

        # read data
        # image_f = np.asarray(PIL.Image.open(os.path.join(img_base_path, img_f)))
        # image_l = np.asarray(PIL.Image.open(os.path.join(img_base_path, img_l)))
        # -- process image

        # Pad appropriately to make it of square shape
        # image_f = make_square(image_f, shape=(512, 512))
        # image_l = make_square(image_l, shape=(512, 512))
        # TODO: Remove this storage logic. This should be a one time thing
        # Store these
        # save_path = os.path.join(img_root_dir, img_f[:img_f.rfind("/")])
        # os.makedirs(save_path, exist_ok=True)
        # np.save(os.path.join(save_path, img_f[img_f.rfind("/") + 1:img_f.rfind(".jpg")]), image_f)
        # np.save(os.path.join(save_path, img_l[img_l.rfind("/") + 1:img_l.rfind(".jpg")]), image_l)

        # apply augmentations
        if self.augment:
            image_f = augment_fn(image=image_f)
            image_l = augment_fn(image=image_l)

        # apply preprocessinging
        if self.preprocess:
            image_f = preprocess_fn(x=image_f)
            image_l = preprocess_fn(x=image_l)

        # Convert them to tensors and add an extra channel for the image
        image_f = torch.as_tensor(image_f, dtype=torch.float32).unsqueeze(0)
        image_l = torch.as_tensor(image_l, dtype=torch.float32).unsqueeze(0)
        labels = torch.as_tensor(labels)
        return [image_f, image_l], labels, extra_label_key


if __name__ == '__main__':
    # def iterate_dataset(dataset):
    #     labels = []
    #     for x in tqdm(range(len(dataset))):
    #         labels.extend(dataset[x][2].numpy().tolist())
    #     counter = Counter(labels)
    #     print(counter)
    # print("train img resize")
    # dataset = MimicDataset(split="train")
    # iterate_dataset(dataset=dataset)
    # print("val img resize")
    # dataset = MimicDataset(split="valid")
    # iterate_dataset(dataset=dataset)
    # print("test img resize")
    # dataset = MimicDataset(split="test")
    # iterate_dataset(dataset=dataset)

    # TODO: Later
    train_dict = pickle.load(open(os.path.join(PROJECT_ROOT_DIR, "dataset", "temp", "train.pkl"), "rb"))
    val_dict = pickle.load(open(os.path.join(PROJECT_ROOT_DIR, "dataset", "temp", "valid.pkl"), "rb"))
    train_val_dict = {k: train_dict[k] for k in list(train_dict)[:len(val_dict)]}
    pickle.dump(train_val_dict, open(os.path.join(PROJECT_ROOT_DIR, "dataset", "temp", "train_val.pkl"), "wb"))
    # dataset = MimicDataset(split="valid", augment=True)
    dataset = MimicDataset(split="test", augment=True)
    f, axarr = plt.subplots(1, 2)
    for idx in range(len(dataset)):
        print(dataset[idx][0][0].shape)
        print(dataset[idx][0][1].shape)
        print(dataset[idx][1].shape)
        axarr[0].imshow(dataset[idx][0][0][0])
        axarr[1].imshow(dataset[idx][0][1][0])
        plt.show()
        break
