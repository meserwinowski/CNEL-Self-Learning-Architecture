# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 11:10:56 2019

@author: meser
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def conv2(v1, v2, m, mode='same'):
    """
    Two-dimensional convolution of matrix m by vectors v1 and v2

    First convolves each column of 'm' with the vector 'v1'
    and then it convolves each row of the result with the vector 'v2'.

    """
    tmp = np.apply_along_axis(np.convolve, 0, m, v1, mode)
    return np.apply_along_axis(np.convolve, 1, tmp, v2, mode)


def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):

    ''' 2D Gaussian mask - should give the same result as MatLAB's
    fspecial('gaussian', [shape], [sigma])
        Machine Epsilon - Smallest discrete difference between numbers
    where they are numerically the same; determined by data type '''

    # Get 2D Gaussian Dimensional Lengths
    m, n = [(ss - 1) / 2 for ss in shape]

    # Numpy Open Grid - Create an "unfleshed out" meshgrid - single arrays
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    # Gaussian formula using the open grid and sigma parameter as input
    h = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))

    # Set values in 2D Gaussian to zero if they are less than the machine
    # epsilon multipled by the max value in the grid
    h[h < np.finfo(h.dtype).eps * h.max()] = 0

    # Divide each element by the sum of all the elements (average?)
    sumh = h.sum()
    if (sumh != 0):
        h /= sumh

    return h


if __name__ == '__main__':
    plt.close()
    prior = matlab_style_gauss2D(shape=(101, 101), sigma=20)
    plt.figure()
    plt.imshow(prior)

#if __name__ == '__main__':
#    A = np.zeros((10, 10))
#    A[2:8, 2:8] = 1
#    x = np.arange(A.shape[0])
#    y = np.arange(A.shape[1])
#    x, y = np.meshgrid(x, y)
#
#    u = [1, 0, -1]
#    v = [1, 2, 1]
#
#    Ch = conv2(u, v, A, 'same')
#    Cv = conv2(v, u, A, 'same')
#
#    plt.figure()
#    ax = plt.gca(projection='3d')
#    ax.plot_surface(x, y, Ch)
#
#    plt.figure()
#    ax = plt.gca(projection='3d')
#    ax.plot_surface(x, y, Cv)
