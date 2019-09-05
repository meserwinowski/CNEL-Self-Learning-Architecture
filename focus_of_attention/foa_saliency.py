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


class SalObject():

    ''' Specific object detected in an image. The object contains coordinates
    to better track it in a pixel positional context. '''

    center_coord = []
    bb_coords = []
    patch_name_list = []

    def __init__(self, center_coord=center_coord):
        self.center_coord = [center_coord]
        self.bb_coords = []

    def object_compare(self, new_coord, frame):
        for o in frame.patch_list:
            if new_coord == o.center_coord[-1]:
                return False
        return True

    def center_check(self, new_center_coord, frame):
        # Check if center_coord list is empty
        if not self.center_coord:
            self.center_coord = [new_center_coord]
            return False

        # Row Difference
        Rd = new_center_coord[0] - self.center_coord[-1][0]

        # Column Difference
        Cd = new_center_coord[1] - self.center_coord[-1][1]

        # 2D Eucledian Distance
        distance = (np.sqrt(Rd ** 2 + Cd ** 2))

        # If distance is within threshold...
        flag = self.object_compare(new_center_coord, frame)
        if (distance > 2 and distance < 16 and flag):
            # Attach this coordinate to the object
            self.center_coord.append(new_center_coord)
            return True
        return False

    def build_bounding_box(self, boundLength=32):

        # Take most recent center coordinate
        R = self.center_coord[-1][0]
        C = self.center_coord[-1][1]

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

        self.bb_coords.append([R1, R2, C1, C2])


def salience_scan(image, rankCount=4, boundLength=32):

    ''' Saliency Map Scan
    salmap - Generated Saliency Map

    Scan through the saliency map with a square region to find the
    most salient pieces of the image. Done by picking the maximally intense
    picture and bounding the area around it '''

    # Copy salience map for processing
    smap = np.copy(image.salience_map)

    # Create a dictionary of the row and column distances from the center pixel
    boxSize = {'Row': boundLength, 'Column': boundLength}

    # Pick out the top 'rankCount' maximally intense regions
    for i in range(rankCount):

        # Grab Maximally Intense Pixel Coordinates (Object Center)
        indices = np.where(smap == smap.max())
        try:
            R = indices[0][0]  # Row
            C = indices[1][0]  # Column
        except IndexError:
            print("Image is probably black")
            R = boundLength
            C = boundLength

        # If list of objects is empty, add a new object
        flag = False
        for o in image.patch_list:

            # Check for an object with center coordinates near new coords
            if o.center_check([R, C], image):
                # Get updated object
                obj = o
                flag = True
                break

        if not flag:

            # Create a new object if returned false
            image.patch_list.append(SalObject([R, C]))
            obj = image.patch_list[-1]

        obj.build_bounding_box()
        image.bb_coords.append(obj.bb_coords[-1])

        # "Zero" the maximally intense region to avoid grabbing it again
        # Sum up and find the average intensity of the region
        total = 0
        R1 = obj.bb_coords[-1][0]
        R2 = obj.bb_coords[-1][1]
        C1 = obj.bb_coords[-1][2]
        C2 = obj.bb_coords[-1][3]
        for j in range(R1, R2):
            for k in range(C1, C2):
                if ((j < image.original.shape[0]) and
                   (k < image.original.shape[1])):
                    total += image.salience_map[j][k]
                    smap[j][k] = 0
