# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 21:34:28 2019

@author: meser

foa_convolution.py - Contains functions for generating 2D kernels and applying
them to a 2D image via convolution.

"""

# Standard Library Imports
import math

# 3P Imports
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Local Imports


def matlab_style_gauss2D(shape=(3, 3), sigma=1, inverse=False):
    """ 2D Gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian', [shape], [sigma])
        Machine Epsilon - Smallest discrete difference between numbers
    where they are numerically the same; determined by data type """

    # Get 2D Gaussian Dimensional Lengths
    m, n = [(ss - 1) / 2 for ss in shape[:2]]

    # Numpy Open Grid - Create an "unfleshed out" meshgrid - single arrays
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    # Gaussian formula using the open grid and sigma parameter as input
    h = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))

    # Set values in 2D Gaussian to zero if they are less than the machine
    # epsilon multipled by the max value in the grid
    h[h < np.finfo(h.dtype).eps * h.max()] = 0

    # # Divide each element by the sum of all the elements (Normalize)
    max_h = h.max()
    if (max_h != 0):
        h /= max_h

    # Return an inverse gaussian instead
    if (inverse):
        h = 1 - h

    return h


def gamma_kernel(image, mask_size=(16, 16), k=None, mu=None, d=2):
    """ Generate a 2D Gamma Kernel
    d - dimensionality
    k - vector contained kernel orders
    mu - vector containing shape parameters """

    # Gamma Filter Order, Shape
    if k is None:
        k = np.array([1, 20, 1, 30, 1, 40], dtype=float)  # Orders
    if mu is None:
        mu = np.array([4, 4, 4, 4, 4, 4], dtype=float)  # Shapes

    # Declare Kernels and Kernel Mask structures
    g = np.zeros((len(mu), 2 * mask_size[0] + 1, 2 * mask_size[1] + 1))
    n1 = np.arange(-mask_size[1], mask_size[1] + 1)
    n2 = np.arange(-mask_size[0], mask_size[0] + 1)
    gk = np.zeros((2 * mask_size[0] + 1, 2 * mask_size[1] + 1))

    # Meshgrid creates 2D coordinates out of vectors
    NX, NY = np.meshgrid(n1, n2)

    # Create a local support grid (zero at center, max in corners)
    supgrid = (NX ** 2 + NY ** 2)

    ''' "Similar to the Itti method and others, Gamma Saliency is based on the
    center surround principle: a region is salient if it is differnt from the
    surroudning neighborhood. In order to compute these local differences, we
    use a 2D gamma kernel that emphasizes a center while constrasting it with a
    local neighborhood through convolution" '''

    ''' For this kernel, n1 and n2 are the local support grid, µ is the shape
    parameter, and k is the kernel order. Using µ and k, we can control the
    shape of the kernel: when k = 1 the kernel peak is centered around zero.
    For larger kernel orders, the peak is centered k/µ away from the center. In
    addition, smaller values of µ will increase the bandwidth of the peak.'''

    # Calculate kernels - Calculate Time ~0.0169537
    for i in range(len(mu)):
        g[i] = ((mu[i] ** (k[i] + 1)) / (2 * np.pi * math.factorial(k[i]))) * \
               (supgrid ** ((k[i] - 1) * 0.5)) * \
               (np.exp(-mu[i] * (supgrid ** 0.5)))
        # plt.figure()
        # plt.imshow(g[i])

    # Normalize Kernels
    for i in range(len(mu)):
        g[i] = g[i] / g[i].sum()

    ''' "For multiscale saliency measure, we simply combine multiple kernels
    of different sizes before the convolution stage. Kernel with larger center
    scale is subtracted by a surround kernel with a larger and further removed
    neighborhood, effectively searching for larger objects by comparing more
    overall area in the image. Kernel summation described in paper. '''

    # Combine Kernels - Subtract the surround from the center
    for i in range(len(mu)):
        kernel = (g[i] * ((-1) ** i))
        # plt.figure()
        # plt.imshow(kernel)
        # plt.title(f"Kernel {i}; Order = {k[i]}; Shape = {mu[i]}")
        gk += kernel

    # plt.figure()
    # plt.imshow(gk)
    # plt.show()

    return gk


def convolution(image, kernel, prior, alpha=None):
    """ Focus of Attention Convolution
    image - Input Image Object - Contains CIELAB Color Space Image
    kernel - Matrix for filtering the image
    alpha - exponent on the saliency
    prior - foveation prior

    Create a 2D gamma kernel and convolve it with the input image to generate
    a saliency map """

    blurSize = 3  # Gaussian Blur - Odd Size Required
    if alpha is None:
        alpha = 4  # Exponentiation Parameter
    gk = kernel

    # Compute Saliency over each scale and apply a Gaussian Blur
    saliency = np.zeros(image.modified.shape)
    sal_map = np.zeros((image.modified.shape[0], image.modified.shape[1]))

    # Colored Images
    if (image.rgb):
        for i in range(3):
            # Convolution Time
            saliency[:, :, i] = cv2.filter2D(image.modified[:, :, i], -1, gk)
            saliency[:, :, i] = abs(saliency[:, :, i])

            # Gaussian Blur Time - ~0.002 seconds per iteration
            blur = matlab_style_gauss2D((26, 26), 0.2 * 26)
            saliency[:, :, i] = cv2.filter2D(saliency[:, :, i], -1, blur)

        # Average Saliency Scales
        sal_map = (saliency[:, :, 0] + saliency[:, :, 1] + saliency[:, :, 2]) / 3

    # Monochromatic Images
    else:
        # Convolution Time
        saliency[:, :] = abs(cv2.filter2D(image.modified[:, :], -1, gk))

        # Gaussian Blur Time - ~0.002 seconds per iteration
        saliency[:, :] = cv2.GaussianBlur(image.modified[:, :],
                                          (blurSize, blurSize), 10)
        sal_map = saliency

    # Foveate the salience map by multipling a gaussian blur across the entire
    # image - helps mimic the visual functionality of the human eye
    sal_map = sal_map * prior

    # Exponentiate by alpha (alpha > 1) to accentuate peaks in the map
    sal_map = sal_map ** alpha

    # Min-max normalize saliency map
    sal_map = (sal_map - sal_map.min()) / (sal_map.max() - sal_map.min())

    # Save saliency map to object passed in
    image.salience_map = sal_map
