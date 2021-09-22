import os
import random

import PIL
import torch
from PIL import Image
from torch.utils.data import Dataset

import numpy as np
from skimage import transform

from torchvision.transforms import transforms

from dataset.data_utils import augment_fn
from environment_setup import PROJECT_ROOT_DIR, read_config

img_base_path = os.path.join(PROJECT_ROOT_DIR, 'dataset')
save_base_path = os.path.join(PROJECT_ROOT_DIR, 'dataset', 'chexpert', 'images')

class ChexDataset(Dataset):

    def __init__(
            self,
            data_items,
            augment=False,
            preprocess=False,
            uncertainty_labels='positive'
    ):

        """
        uncertainity_
        mode: 'positive', 'negative', 'ignore'. Denote U-ignore or U-positive or U-negative
        """
        self.data_items = data_items
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[129.09964561039797], std=[73.81427725645409]
            )
        ])
        self.preprocess = preprocess
        self.augment = augment
        self.uncertainty_labels = uncertainty_labels
        self.img_size = read_config()['data'].getint('img_shape')
        print('[*] uMode:', self.uncertainty_labels)

    def __getitem__(self, i):

        data_item = self.data_items[i]
        img_path = data_item[0]
        img_label = data_item[1]
        labels = np.array(img_label)
        # NOTE: Use this section when changing the image size. For the sake of speed, I preprocessed the input if it is
        # of a fixed size and simply load the numpy arrays.
        # read data
        image = PIL.Image.open(os.path.join(img_base_path, img_path))
        labels = np.array(img_label)
        # Resize the image
        image = transforms.Resize((self.img_size, self.img_size))(image)
        # apply augmentations
        if self.augment:
            image = augment_fn(image=image)

        # apply preprocessinging
        if self.preprocess:
            image = self.transforms(image)

        # how to handle uncertain labels? for ex: u-positive or u-negative
        if self.uncertainty_labels == 'negative':
            labels[labels == -1] = 0
        elif self.uncertainty_labels == 'positive':
            labels[labels == -1] = 1
        elif self.uncertainty_labels == 'multiclass':
            labels[labels == -1] = 2
        # Convert them to tensors and add an extra channel for the image
        image = torch.as_tensor(image, dtype=torch.float32).unsqueeze(0)
        labels = torch.as_tensor(labels)

        return image, labels

    def __len__(self):
        return len(self.data_items)
