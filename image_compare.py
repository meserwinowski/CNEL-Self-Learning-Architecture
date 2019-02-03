# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 14:42:28 2019

@author: meser

image_compare.py - Implement a similarity metric to compare two images. Current
method is two import two images and do an element wise L2 norm.

"""

import fes_gamma as fg
from skimage import io, color, transform
import scipy.io
import scipy.signal
from PIL import Image
import time
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})


if __name__ == '__main__':
    image = fg.imageObject('./AIM/eyetrackingdata/original_images/',
                           '120', '.jpg', RGB=True)
#    salience_map = fg.imageObject('./', 'smap_120', '.jpg', RGB=False)
    ground_truth = fg.imageObject('./AIM/eyetrackingdata/ground_truth/',
                                  'd120', '.jpg', RGB=False)
    # If image is RGB...
    if (image.rgb):
        image.original = io.imread(image.path +
                                   image.name +
                                   image.ex)
        image.img = color.rgb2lab(image.original)
        image.original = transform.resize(image.original,
                                          np.array([128, 171]),
                                          mode='constant',
                                          anti_aliasing=False)
        image.img = transform.resize(image.img,
                                     np.array([128, 171]),
                                     mode='constant',
                                     anti_aliasing=False)
#    salience_map = io.imread(salience_map.path +
#                             salience_map.name +
#                             salience_map.ex)
    ground_truth = Image.open(ground_truth.path +
                             ground_truth.name +
                             ground_truth.ex)

    plt.figure()
    plt.imshow(ground_truth)
