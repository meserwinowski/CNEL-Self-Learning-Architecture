# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 11:55:52 2018

@author: meser

fes_gamma_test.py - Initial test of the DPCN front end system gamma kernel
concept in Python. This script applys the gamma kernel saliency model to single
images.

"""

import fes_gamma as fg
import scipy.io
import time
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})


# Main Routine
if __name__ == '__main__':
    plt.close('all')

    # Open test images as 8-bit RGB values - Time ~0.0778813
#    mario = fg.imageObject('./ICASSP Ryan/', 'mario', '.png', RGB=True)
    mario = fg.imageObject('./', 'SMW_Test_Image', '.png', RGB=True)
    banana = fg.imageObject('./AIM/eyetrackingdata/original_images/',
                            '22', '.jpg', RGB=True)
    corner = fg.imageObject('./AIM/eyetrackingdata/original_images/',
                            '120', '.jpg', RGB=True)

    testIMG = mario

    # Convert image to CIELAB Color Space - Resize Image and create gray scale
    # version if required
    fg.convert(testIMG)

# %% Generate Gamma Kernel and Saliency Map

    # Set Gamma Filter Orders and Shape parameters
    k = np.array([1, 20, 1, 30, 1, 40], dtype=float)
    mu = np.array([2, 2, 2, 2, 2, 2], dtype=float)
    alpha = 5

    # Generate Gaussian Blur Prior - Time ~0.0020006
    prior = fg.matlab_style_gauss2D((testIMG.img.shape[0],
                                     testIMG.img.shape[1]), sigma=300) * 1
#    mat = scipy.io.loadmat('./ICASSP Ryan/prior.mat')
#    prior = mat['p1']

    # Generate Saliency Map with Gamma Filter
    start = time.time()
    fg.FES_Gamma(testIMG, k, mu, alpha, prior)
    stop = time.time()
    print("Salience Map Generation: ", stop - start, " seconds")

    # Bound and Rank the most Salient Regions of Saliency Map
    fg.salScan(testIMG)

# %% Plot Results

    # Plot Saliency Map
    plt.figure()
    plt.imshow(testIMG.salience_map)
    plt.plot()

    # Plot max intensity regions, and plot bounding box on original image
#    fig, axes = plt.subplots(1, 5, sharey=True)
    bT = 1  # Bounding Box Thickness
    for i in range(len(testIMG.MIC)):

        # Grab bounding coordinates
        a = testIMG.MIC[i][0]
        b = testIMG.MIC[i][1]
        c = testIMG.MIC[i][2]
        d = testIMG.MIC[i][3]

        # Update original image
        if (testIMG.rgb):
            testIMG.original[a:b, c:c+bT] = [255, 150, 100]
            testIMG.original[a:b, d:d+bT] = [255, 150, 100]
            testIMG.original[a:a+bT, c:d+bT] = [255, 100, 100]
            testIMG.original[b:b+bT, c:d+bT] = [255, 100, 100]
        else:
            testIMG.img[a:b, c:c+bT] = [255]
            testIMG.img[a:b, d:d+bT] = [255]
            testIMG.img[a:a+bT, c:d+bT] = [255]
            testIMG.img[b:b+bT, c:d+bT] = [255]

        # Generate intense region subplots
#        axes[i].imshow(testIMG.salience_map[a:b, c:d])
#        axes[i].set_title("X: {}, Y: {}".format(testIMG.MIC[i][1] - 25,
#                                                testIMG.MIC[i][3] - 25))
    plt.show()

    # Create a figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.set_title('Original')
    ax2.set_title('Saliency Map')
    if (testIMG.rgb):
        ax1.imshow(testIMG.original)
    else:
        testIMG.img.astype(int)
        ax1.imshow(testIMG.img)
    ax2.imshow(testIMG.salience_map)
    plt.show()
