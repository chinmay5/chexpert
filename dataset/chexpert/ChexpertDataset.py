import csv

import os
import random

import PIL
import math
import torch
from PIL import Image
from torch.utils.data import Dataset

import numpy as np
from skimage import transform

from environment_setup import PROJECT_ROOT_DIR, read_config

img_base_path = os.path.join(PROJECT_ROOT_DIR, 'dataset')

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

        self.preprocess = preprocess
        self.augment = augment
        self.uncertainty_labels = uncertainty_labels
        self.img_size = read_config()['data'].getint('img_shape')
        print('[*] uMode:', self.uncertainty_labels)

    def augment_fn(self, image):

        if random.random() > 0.25:
            tform = self.generate_perturbation_matrix_2D(max_t=0, max_s=1.4, min_s=0.7, max_r=5)
            image = transform.warp(image, tform.inverse, order=3)

        return image

    def preprocess_fn(self, x):
        # stats computed in prepare_chexpert.ipynb
        MEAN_chexpert = 129.09964561039797
        STD_chexpert = 73.81427725645409

        x = x - MEAN_chexpert
        x = x / STD_chexpert

        return x

    def __getitem__(self, i):

        data_item = self.data_items[i]
        img_path = data_item[0]
        img_label = data_item[1]
        # read data
        image = np.asarray(PIL.Image.open(os.path.join(img_base_path, img_path)))
        labels = np.array(img_label)
        # -- process image

        # Pad appropriately to make it of square shape
        image = self.make_square(image, shape=(self.img_size, self.img_size))

        # apply augmentations
        if self.augment:
            image = self.augment_fn(image=image)

        # apply preprocessinging
        if self.preprocess:
            image = self.preprocess_fn(x=image)

        # -- process labels

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

    def generate_perturbation_matrix_2D(self, max_t=5, max_s=1.4, min_s=0.7, max_r=5):

        # translation
        tx = np.random.uniform(-max_t, max_t)
        ty = np.random.uniform(-max_t, max_t)

        # scaling
        sx = np.random.uniform(min_s, max_s)
        sy = np.random.uniform(min_s, max_s)

        # rotation
        r = (math.pi / 180) * np.random.uniform(-max_r, max_r)  # x-roll (w.r.t x-axis)

        # Generate perturbation matrix

        tform = transform.AffineTransform(scale=(sx, sy),
                                          rotation=r,
                                          translation=(tx, ty))

        return tform

    def make_square(self, image, shape):
        h, w = image.shape
        if h < w:
            diff = w - h
            if diff % 2 == 0:
                pad = (diff // 2, diff // 2)
            else:
                pad = (diff // 2 + 1, diff // 2)
            image = np.pad(image, pad_width=(pad, (0, 0)))
        elif h > w:
            diff = h - w
            if diff % 2 == 0:
                pad = (diff // 2, diff // 2)
            else:
                pad = (diff // 2 + 1, diff // 2)
            image = np.pad(image, pad_width=((0, 0), pad))

        image = transform.resize(image, shape, order=1, preserve_range=True, anti_aliasing=True)

        return image
