# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 21:45:00 2019

@author: meser

foa_saliency.py - Contains functions for generating saliency maps and
searching for regions of interest to be bounded.

"""

# Standard Library Imports

# 3P Imports
import cv2
import numpy as np

# Local Imports
from foa_image import ImageObject


def build_bounding_box(center, boundLength=32):

    ''' Generate the coordinates that defined a square region
    around the detected region of interest.
    center - Center coords of the object
    boundLength - Length and width of the square saliency region '''

    # Take most recent center coordinate
    R = center[0]
    C = center[1]

    # Dictionary for Clarity
    boxSize = {'Row': boundLength, 'Column': boundLength}

    # Derive upper left coordinate of bounding region
    R1 = int(R - (boxSize['Row'] / 2))
    if (R1 < 0):
        R1 = 0
    C1 = int(C - (boxSize['Column'] / 2))
    if (C1 < 0):
        C1 = 0

    # Derive lower right coordinate of bounding region
    R2 = int(R + (boxSize['Row'] / 2))
    if (R2 > 223):
        R2 = 223
    C2 = int(C + (boxSize['Column'] / 2))
    if (C2 > 255):
        C2 = 255

    return [R1, R2, C1, C2]


def salience_scan(image=ImageObject, rankCount=4, boundLength=32):

    ''' Saliency Map Scan

    Scan through the saliency map with a square region to find the
    most salient pieces of the image. Done by picking the maximally intense
    picture and bounding the area around it

    image - ImageObject being scanned
    rankCount - Number of objects to acquire before stopping
    boundLength - Length and width of the square saliency region '''

    # Copy salience map for processing
    smap = np.copy(image.salience_map)

    # Pick out the top 'rankCount' maximally intense regions
    for i in range(rankCount):

        # Grab Maximally Intense Pixel Coordinates (Object Center)
        indices = np.where(smap == smap.max())
        try:
            R = indices[0][0]  # Row
            C = indices[1][0]  # Column
        except IndexError:
            print("Image has no variation, might just be black")
            R = boundLength
            C = boundLength

        # Get bounding box coordinates for object
        coords = build_bounding_box([R, C], boundLength)

        # Add coordinates to member list on the image object
        image.bb_coords.append(coords)

        # "Zero" the maximally intense region to avoid selecting it again
        R1 = coords[0]
        R2 = coords[1]
        C1 = coords[2]
        C2 = coords[3]

        # Sum up and find the average intensity of the region
        pixel_intensity_sum = 0

        # Traverse through identified region
        for j in range(R1, R2):
            for k in range(C1, C2):
                x_dim = image.original.shape[0]
                y_dim = image.original.shape[1]
                if ((j < x_dim) and (k < y_dim)):
                    pixel_intensity_sum += image.salience_map[j][k]
                    smap[j][k] = 0  # Zero out pixel
