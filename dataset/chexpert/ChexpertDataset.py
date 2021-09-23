import os
import random

import PIL
import torch
from PIL import Image
from torch.utils.data import Dataset

import numpy as np

from torchvision.transforms import transforms

from dataset.data_utils import augment_fn
from environment_setup import PROJECT_ROOT_DIR, read_config

img_base_path = os.path.join(PROJECT_ROOT_DIR, 'dataset')
save_base_path = os.path.join(PROJECT_ROOT_DIR, 'dataset', 'chexpert', 'images')

class ChexDataset(Dataset):

    def __init__(
            self,
            data_items,
            mode='train',
            uncertainty_labels='positive'
    ):

        """
        uncertainity_
        mode: 'positive', 'negative', 'ignore'. Denote U-ignore or U-positive or U-negative
        """
        self.data_items = data_items
        self.mode = mode
        self.uncertainty_labels = uncertainty_labels
        img_size = read_config()['data'].getint('img_shape')
        crop_size = read_config()['data'].getint('crop_size')
        # Create the augmentations
        normalize = transforms.Normalize(
            mean=[129.09964561039797], std=[73.81427725645409]
        )
        train_transforms, val_transforms, test_transforms = self.get_transforms(crop_size, img_size, normalize)
        if mode == 'train':
            self.transform = train_transforms
        elif mode == 'valid' or mode == 'train_val':
            self.transform = val_transforms
        elif mode == 'test':
            self.transform = test_transforms
        else:
            raise AttributeError(f"Invalid mode selected. Selection was {mode}")
        print('[*] uMode:', self.uncertainty_labels)

    def get_transforms(self, crop_size, img_size, normalize):
        # The train transforms
        train_transforms = []
        train_transforms.append(transforms.Resize(img_size))
        train_transforms.append(transforms.RandomResizedCrop(crop_size))
        train_transforms.append(transforms.RandomHorizontalFlip())
        train_transforms.append(transforms.ToTensor())
        train_transforms.append(normalize)
        train_transforms = transforms.Compose(train_transforms)
        # The val transforms
        val_transforms = []
        val_transforms.append(transforms.Resize(crop_size))  # crop determines the size
        val_transforms.append(transforms.ToTensor())
        val_transforms.append(normalize)
        val_transforms = transforms.Compose(val_transforms)
        # The test transforms
        test_transforms = []
        test_transforms.append(transforms.Resize(img_size))
        test_transforms.append(transforms.TenCrop(crop_size))
        test_transforms.append(
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
        test_transforms.append(transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])))
        test_transforms = transforms.Compose(test_transforms)
        return train_transforms, val_transforms, test_transforms

    def __getitem__(self, i):

        data_item = self.data_items[i]
        img_path = data_item[0]
        img_label = data_item[1]
        image = PIL.Image.open(os.path.join(img_base_path, img_path))
        labels = np.array(img_label)
        # apply task specific transformations
        image = self.transform(image)

        # how to handle uncertain labels? for ex: u-positive or u-negative
        if self.uncertainty_labels == 'negative':
            labels[labels == -1] = 0
        elif self.uncertainty_labels == 'positive':
            labels[labels == -1] = 1
        elif self.uncertainty_labels == 'multiclass':
            labels[labels == -1] = 2
        # Convert them to tensors and add an extra channel for the image
        image = torch.as_tensor(image, dtype=torch.float32)
        labels = torch.as_tensor(labels)

        return image, labels

    def __len__(self):
        return len(self.data_items)
