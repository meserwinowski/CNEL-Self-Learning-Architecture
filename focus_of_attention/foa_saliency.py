# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 21:45:00 2019

@author: meser

foa_saliency.py - Contains functions for generating saliency maps and
searching for regions of interest to be bounded.

"""

# Standard Library Imports
import sys

# 3P Imports
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Local Imports
from foa_image import ImageObject
from foa_convolution import matlab_style_gauss2D


def build_bounding_box(center, smap, bbox_size=(32, 32)):
    """ Generate the coordinates that defined a square region
    around the detected region of interest.
    center - Center coords of the object
    boundLength - Length and width of the square saliency region """

    # Take most recent center coordinate
    R = center[0]
    C = center[1]

    # Dictionary for Clarity
    bbox_size = {'Row': bbox_size[0], 'Column': bbox_size[1]}

    # Derive upper left coordinate of bounding region
    R1 = int(R - (bbox_size['Row'] / 2))
    if (R1 < 0):
        R1 = 0
    C1 = int(C - (bbox_size['Column'] / 2))
    if (C1 < 0):
        C1 = 0

    # Derive lower right coordinate of bounding region
    R2 = int(R + (bbox_size['Row'] / 2))
    if (R2 > smap.shape[1]):
        R2 = smap.shape[1]
    C2 = int(C + (bbox_size['Column'] / 2))
    if (C2 > smap.shape[2]):
        C2 = smap.shape[2]

    return {"top_left": [R1, C1], "bottom_right": [R2, C2], "center": [R, C]}


def salience_scan(image=ImageObject, rank_count=4, bbox_size=(32, 32)):
    """ Saliency Map Scan

    Scan through the saliency map with a square region to find the
    most salient pieces of the image. Done by picking the maximally intense
    picture and bounding the area around it

    image - ImageObject being scanned
    rankCount - Number of objects to acquire before stopping
    boundLength - Length and width of the square saliency region """

    # Copy salience map for processing
    smap = np.copy(image.salience_map)
    image.patched_sequence = np.empty((0, smap.shape[0], smap.shape[1]))

    # Create an inverse Gaussian kernel for removing salient regions
    inverse_gauss = matlab_style_gauss2D(bbox_size, sigma=28, inverse=True)

    # Pick out the top 'rankCount' maximally intense regions
    for i in range(rank_count):

        # # Copy and Reshape saliency map
        temp_smap = np.copy(smap)
        temp_smap = np.reshape(temp_smap, (1, smap.shape[0], smap.shape[1]))

        # Append modified saliency map
        image.patched_sequence = np.vstack((image.patched_sequence, temp_smap))

        # Grab Maximally Intense Pixel Coordinates (Object Center)
        indices = np.where(smap == smap.max())
        try:
            R = indices[0][0]  # Row
            C = indices[1][0]  # Column
        except IndexError:
            if (i == 1):
                print("Image has no variation, might just be black")
            R = bbox_size[0]
            C = bbox_size[1]

        # Get bounding box coordinates for object
        coords = build_bounding_box([R, C], temp_smap, bbox_size)
        # print(f"Coords {i}: {coords}")

        # Add coordinates to member list on the image object
        image.bb_coords.append(coords)

        # "Zero" the maximally intense region to avoid selecting it again
        R1 = coords["top_left"][0]
        C1 = coords["top_left"][1]
        R2 = coords["bottom_right"][0]
        C2 = coords["bottom_right"][1]

        # Sum up and find the average intensity of the region
        pixel_intensity_sum = 0

        # Traverse through identified region
        for j in range(R1, R2):
            for k in range(C1, C2):
                x_dim = image.original.shape[0]
                y_dim = image.original.shape[1]
                if ((j < x_dim) and (k < y_dim)):
                    pixel_intensity_sum += image.salience_map[j][k]
                    # smap[j][k] = 0  # Zero out pixel
                    smap[j][k] *= inverse_gauss[R2 - j - 1][C2 - k - 1]
