import random

import math

import numpy as np
from torchvision.transforms import transforms


def augment_fn(image):
    if random.random() > 0.25:
        tform = transforms.RandomAffine(degrees=5, translate=(1, 1), scale=(0.7, 1.4))
        image = tform(image)

    return image
