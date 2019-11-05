# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 11:55:52 2018

@author: meser

focus_attention_test.py - Initial test of the DPCN front end system gamma
kernel concept in Python. This script applys the gamma kernel saliency model to
single images.

"""

# Standard Library Imports
import os
import sys
import time
import tempfile as tf

# 3P Imports
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Local Imports
import foa_image as foai
import foa_convolution as foac
import foa_saliency as foas

plt.rcParams.update({'font.size': 22})
#image_set = np.array([False])


# Main Routine
if __name__ == '__main__':
    plt.close('all')

    # Load image set data
    if (image_set.any() == False):
        print("Importing data...")
        image_set = np.load("./STATE0.npy")

    # Test Image
    test_image = image_set[0]
    cv2.imwrite("0.png", test_image)
    test_image = foai.ImageObject(cv2.imread("0.png"))
    os.remove("0.png")

# %% Generate Gaussian and Gamma Kernel

    # Generate Gaussian Blur Prior - Time ~0.0020006
    foveation_prior = foac.matlab_style_gauss2D(test_image.modified.shape, 300)

    # Generate Gamma Kernel
    kernel = foac.gamma_kernel(test_image)

# %% Process Images
    image_height = image_set.shape[1]
    image_width = image_set.shape[2]
    rankCount = 10  # Number of maps scans to run
    processed_data = np.empty((len(image_set), rankCount, image_height, image_width))

    # Generate Saliency Map
    start = time.time()
    for i, img in enumerate(image_set, 0):
        # TODO: File needs to be resaved and opened with cv2 for some reason...
        file_name = f"{i}.png"
        cv2.imwrite(file_name, img)
        test_image = foai.ImageObject(cv2.imread(file_name))
        os.remove(file_name)

        # Create saliency map
        foac.convolution(test_image, kernel, foveation_prior)

        # Display Progress
        if (i % 100 == 0):
            stop = time.time()
            print(f"Saliency Map Generation {i}: {stop - start} seconds")
            start = time.time()

        # Bound and Rank the most salient regions of saliency map
        foas.salience_scan(test_image, rankCount=rankCount)

        # Stack images
        processed_data[i] = test_image.patched_sequence

    # Save Processed Images
    np.save("./out_file.npy", processed_data)
