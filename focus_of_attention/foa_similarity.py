# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 14:42:28 2019

@author: meser

foa_similarity.py - Implement similarity metrics to compare images.

Current methods:
    ROC Curve AUC (Borji)
    ROC Curve AUC (Judd)
    Correlation Coefficient
    Normalized Scanpath Saliency
    Pixel Eucledian Distance

"""

# Standard Library Imports
import time

# 3P Imports
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import scipy.io

# Local Imports
import foa_image as foai
import foa_convolution as foac
import foa_saliency as foas

plt.rcParams.update({'font.size': 22})


def squarePadImage(image):

    # Find differential between the image dimensions
    dim_diff = abs(np.array(image.shape) - max(image.shape))

    # Determine dimension to pad
    index = 0
    if (image.shape[0] > image.shape[1]):
        index = 1
    else:
        index = 0

    # Calculate amount to pad
    nmap = [(0, 0) for x in range(len(image.shape))]
    nmap[index] = (int(dim_diff[0] / 2), int(dim_diff[0] / 2))

    # Pad
    image = np.pad(image, pad_width=nmap, mode='constant')

    return image


def eigDecompImage(image):
    ''' Perform eigen decomposition on an image map. Requires that the
    image be padded to a square. '''
    assert len(image.shape) == 2, "Image map should be two dimensions!"

    # Pad image if necessary
    if (image.shape[0] != image.shape[1]):
        image = squarePadImage(image)

    # Eigendecomposition - returns 2 items: eigenvals and eigenvects
    return np.linalg.eig(image)


def corrCoeff(image_x, image_y):
    ''' Calculate Pearson's Correlation Coefficient between the two saliency
    maps to quantify similarity '''
    assert len(image_x.shape) == 2, "Image X map should be two dimensions!"
    assert len(image_y.shape) == 2, "Image Y map should be two dimensions!"

    return np.corrcoef(image_x.flat, image_y.flat)[0, 1]


def NSS(image, thresh=0.9):
    ''' Normalized Scanpath Saliency '''
    # Get indices greater than threshold
    indices = np.where(image.ground_truth >= thresh)
    nss_v = np.zeros((len(indices[0]), 1))
    image.salience_map = ((image.salience_map - np.mean(image.salience_map)) /
                          np.std(image.salience_map))
    # Use indices to get values from the generated saliency map
    for i in range(len(nss_v)):
        nss_v[i] = image.salience_map[indices[0][i], indices[1][i]]

    return nss_v


# Main Routine
if __name__ == '__main__':
    plt.close('all')

    # Open test images as 8-bit RGB values - Time ~0.0778813
    file = "./SMW_Test_Image.png"
    mario = foai.imageObject(file)
    file = "./AIM/eyetrackingdata/original_images/22.jpg"
    banana = foai.imageObject(file)
    file = "./AIM/eyetrackingdata/original_images/120.jpg"
    corner = foai.imageObject(file)

    # Test Image
    testIMG = mario

# %% Generate Saliency Map

    # Generate Gaussian Blur Prior - Time ~0.0020006
    foveation_prior = foac.matlab_style_gauss2D(testIMG.modified.shape, 300)

    # Generate Gamma Kernel
    kernel = foac.gamma_kernel(testIMG)

    # Generate Saliency Map
    start = time.time()
    foac.foa_convolution(testIMG, kernel, foveation_prior)
    stop = time.time()
    print("Salience Map Generation: ", stop - start, " seconds")

    # Bound and Rank the most Salient Regions of Saliency Map
    foas.salience_scan(testIMG, rankCount=5)

# %% Plot Results

    # Plot Bounding Box Patches
    testIMG.draw_image_patches()
    testIMG.save_image_patches()

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
