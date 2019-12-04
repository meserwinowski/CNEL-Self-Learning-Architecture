# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 13:55:38 2019

@author: meser
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt

# Standard Imports
import sys
import time

# Local Imports
import foa_image as foai
import foa_convolution as foac
import foa_saliency as foas
from generate_svhn import SVHNImage


def generate_svhn_ground_truth(input=SVHNImage):
    assert (input.image is not None)
    img = np.array(input.image)

    bbox = input.gt_coord
    print(img.shape)
    ground_truth = np.zeros((img.shape[0], img.shape[1]), dtype=float)
    print(ground_truth.shape)
    for num in bbox:
        print(num)
        top_left = num['top_left']
        bottom_right = num['bottom_right']

        for i in range(len(ground_truth)):
            for j in range(len(ground_truth[0])):
                if (i > top_left[0] and i < bottom_right[0] and
                    j > top_left[1] and j < bottom_right[1]):
                    ground_truth[i][j] = 1.0

    return ground_truth


def run_foa_svhn(input=SVHNImage, tf=int, gt=None, k=None, mu=None):
    assert (input.image is not None)
    img = np.array(input.image)
    x_dim = img.shape[0]
    y_dim = img.shape[1]
    print(np.array(img.shape))

    # Load Image Object
    img = foai.ImageObject(img)
    img.ground_truth = gt

    # Generate Gaussian Blur Prior - Time ~0.0020006
    foveation_prior = foac.matlab_style_gauss2D(img.modified.shape, 300)

    # Generate Gamma Kernel
    # k = np.array([1, 26, 1, 25, 1, 19], dtype=float)
    # mu = np.array([2, 2, 1, 1, 0.5, 0.5], dtype=float)
    k = np.array([1, 5, 1, 9, 1, 13], dtype=float)
    mu = np.array([0.8, 0.7, 0.3, 0.5, 0.1, 0.3], dtype=float)
    kernel = foac.gamma_kernel(img, mask_size=(32, 32), k=k, mu=mu)

    # Generate Saliency Map
    start = time.time()
    foac.convolution(img, kernel, foveation_prior)
    stop = time.time()
    print(f"Salience Map Generation: {stop - start} seconds")

    # Bound and Rank the most Salient Regions of Saliency Map
    foas.salience_scan(img, rank_count=4, bbox_size=(y_dim // 4, y_dim // 4))
    bbt = 2
    if (x_dim < 100 and y_dim < 100):
        bbt = 1

    img.draw_image_patches(bbt=bbt)

    # Threshold
    img.salience_map = np.where(img.salience_map > tf, img.salience_map, 0)

    img.plot_main()
    img.plot_patched_map()


if __name__ == "__main__":
    plt.close()

    # SVHN - Format 1
    count = 0
    filename = "./SVHN/format1/train.pkl"
    with open(filename, 'rb') as infile:
        try:
            while count < 10:
                svhn_data = pickle.load(infile)
                print(svhn_data)
                run_foa_svhn(svhn_data, 0.2, generate_svhn_ground_truth(svhn_data))
                count += 1
        except EOFError:
            print("End of File")
            pass
