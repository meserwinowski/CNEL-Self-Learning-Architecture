# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 11:55:52 2018

@author: meser

foa_test.py - Initial test of the DPCN front end system gamma
kernel concept in Python. This script applys the gamma kernel saliency model to
single images.

"""

# Standard Library Imports
import sys
import time

# 3P Imports
import matplotlib.pyplot as plt
import numpy as np

# Local Imports
import foa_image as foai
import foa_convolution as foac
import foa_saliency as foas

# plt.rcParams.update({'font.size': 22})
np.set_printoptions(threshold=sys.maxsize)


# Main Routine
if __name__ == '__main__':
    plt.close('all')

    # Open test images as 8-bit RGB values - Time ~0.0778813
    file = "./SMW_Test_Image.png"
    mario = foai.ImageObject(file)
    file = "../datasets/AIM/eyetrackingdata/original_images/22.jpg"
    banana = foai.ImageObject(file)
    file = "../datasets/AIM/eyetrackingdata/original_images/120.jpg"
    corner = foai.ImageObject(file)

    # Test Image
    test_image = mario

# %% Generate Saliency Map

    # Generate Gaussian Blur Prior - Time ~0.0020006
    foveation_prior = foac.matlab_style_gauss2D(test_image.modified.shape, 300)

    # Generate Gamma Kernel
    kernel = foac.gamma_kernel(test_image)

    # Generate Saliency Map
    start = time.time()
    foac.convolution(test_image, kernel, foveation_prior)
    stop = time.time()
    print(f"Salience Map Generation: {stop - start} seconds")

    # Bound and Rank the most Salient Regions of Saliency Map
    foas.salience_scan(test_image, rankCount=10)

# %% Display Results

    test_image.plot_original_map()
    # Plot Bounding Box Patches
    test_image.draw_image_patches()
    for i in range(len(test_image.patched_sequence)):
        test_image.display_map(test_image.patched_sequence[i], f"{i}")
    test_image.plot_patched_map()
    # test_image.plot_saliency_map()

#    # Create a figure with 2 subplots
#    fig, (ax1, ax2) = plt.subplots(1, 2)
#    ax1.set_title('Original')
#    ax2.set_title('Saliency Map')
#    if (test_image.rgb):
#        ax1.imshow(test_image.patched)
#    else:
#        test_image.modified.astype(int)
#        ax1.imshow(test_image.modified)
#    ax2.imshow(test_image.salience_map)
#    plt.show()
