# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 14:42:28 2019

@author: meser

image_compare.py - Implement similarity metrics to compare two images.

Current methods:
    ROC Curve AUC (Borji)
    ROC Curve AUC (Judd)
    Similarity Measure
    Correlation Coefficient
    Normalized Scanpath Saliency
    Pixel Eucledian Distance

"""

# Standard Library Imports
import time

# 3P Imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from skimage import io, color, transform
import scipy.io
import scipy.signal
from PIL import Image

# Local Imports
import fes_gamma as fg

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
    image.salience_map = (image.salience_map - 
                          np.mean(image.salience_map)) / \
                          np.std(image.salience_map)
    # Use indices to get values from the generated saliency map
    for i in range(len(nss_v)):
        nss_v[i] = image.salience_map[indices[0][i], indices[1][i]]

    return nss_v


def SM(image):
    
    return


if __name__ == '__main__':
    image = fg.imageObject('./AIM/eyetrackingdata/original_images/',
                           '120', '.jpg', RGB=True)
#    salience_map = fg.imageObject('./', 'smap_120', '.jpg', RGB=False)
    ground_truth = fg.imageObject('./AIM/eyetrackingdata/ground_truth/',
                                  'd120', '.jpg', RGB=False)

    # Modify image for processing
    fg.convert(image)

    # Generate Gaussian Blur Prior - Time ~0.0020006
    prior = fg.matlab_style_gauss2D((image.modified.shape[0],
                                     image.modified.shape[1]), sigma=300) * 1

    # Generate Saliency Map with Gamma Filter
    start = time.time()
    fg.FES_Gamma(image, image.k, image.mu, image.alpha, prior)
    stop = time.time()
    print("Salience Map Generation: ", stop - start, " seconds")

    ground_truth = Image.open(ground_truth.path +
                              ground_truth.name +
                              ground_truth.ex)

    if (ground_truth != np.ndarray):
        print("Ground truth is PIL")

    ground_truth = np.array(ground_truth).astype(float)
    ground_truth /= ground_truth.max()

    image.ground_truth = ground_truth

    if (image.ground_truth.shape != image.salience_map.shape):
        raise RuntimeError("Images do not have the same pixel dimensions!")

#    plt.figure()
#    plt.imshow(image.salience_map, cmap=cm.jet)
#    plt.figure()
#    plt.imshow(image.ground_truth, cmap=cm.jet)

#    val1, vec1 = eigDecompImage(image.salience_map)
#    val2, vec2 = eigDecompImage(image.ground_truth)

#    test = corrCoeff(image.salience_map, image.ground_truth)
    test = NSS(image)
    print(np.mean(test))

#    hist_1, _ = np.histogram(image.salience_map, bins=100)
#    hist_2, _ = np.histogram(image.ground_truth, bins=100)
#    minima = np.minimum(hist_1, hist_2)
#    intersection = np.true_divide(np.sum(minima), np.sum(hist_2))

#    plt.figure()
#    plt.hist(image.salience_map*100, bins=100, range=[0, 1])
#    plt.hist(image.ground_truth, bins=100)
#    plt.show()
