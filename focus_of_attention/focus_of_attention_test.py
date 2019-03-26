# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 11:55:52 2018

@author: meser

focus_attention_test.py - Initial test of the DPCN front end system gamma
kernel concept in Python. This script applys the gamma kernel saliency model to
single images.

"""

# Standard Library Imports
import time

# 3P Imports
import matplotlib.pyplot as plt

# Local Imports
import focus_of_attention as foa

plt.rcParams.update({'font.size': 22})


# Main Routine
if __name__ == '__main__':
    plt.close('all')

    # Open test images as 8-bit RGB values - Time ~0.0778813
    mario = foa.imageObject('./',               # Path
                            'SMW_Test_Image',   # Name
                            '.png',             # Extension
                            rgb=True)           # RGB Boolean
    banana = foa.imageObject('./AIM/eyetrackingdata/original_images/',
                             '22',
                             '.jpg',
                             rgb=True)
    corner = foa.imageObject('./AIM/eyetrackingdata/original_images/',
                             '120',
                             '.jpg',
                             rgb=True)

    # Test Image
    testIMG = mario

    # Convert image to CIELAB Color Space
    testIMG.image_convert()

# %% Generate Saliency Map

    # Generate Gaussian Blur Prior - Time ~0.0020006
    foveation_prior = foa.matlab_style_gauss2D(testIMG.modified.shape, 300)

    # Generate Saliency Map with Gamma Filter
    start = time.time()
    foa.front_end_convolution(testIMG, foveation_prior)
    stop = time.time()
    print("Salience Map Generation: ", stop - start, " seconds")

    # Bound and Rank the most Salient Regions of Saliency Map
    foa.salScan2(testIMG, rankCount=4)

# %% Plot Results

    # Plot Bounding Box Patches
    testIMG.draw_image_patches()

    # Create a figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_title('Original')
    ax2.set_title('Saliency Map')
    if (testIMG.rgb):
        ax1.imshow(testIMG.patched)
    else:
        testIMG.modified.astype(int)
        ax1.imshow(testIMG.modified)
    ax2.imshow(testIMG.salience_map)
    plt.show()
