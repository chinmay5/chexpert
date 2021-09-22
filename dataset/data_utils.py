import random

import math

import numpy as np
from skimage import transform


def preprocess_fn(x):
    # stats computed in prepare_chexpert.ipynb
    MEAN_chexpert = 129.09964561039797
    STD_chexpert = 73.81427725645409

    x = x - MEAN_chexpert
    x = x / STD_chexpert

    return x


def generate_perturbation_matrix_2D(max_t=5, max_s=1.4, min_s=0.7, max_r=5):
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


def make_square(image, shape):
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


def augment_fn(image):
    if random.random() > 0.25:
        tform = generate_perturbation_matrix_2D(max_t=0, max_s=1.4, min_s=0.7, max_r=5)
        image = transform.warp(image, tform.inverse, order=3)

    return image
