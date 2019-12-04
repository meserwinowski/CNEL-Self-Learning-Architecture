# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 11:55:52 2018

@author: meser

focus_of_attention.py - Initial front end system gamma kernel implemenation in
Python. Contains the current functional breakdown of the front end system (FES)
gamma kernel. Current implementation is intended to mimic the initial MATLAB
implementation and results created by Ryan Burt.

"""

# Standard Library Imports

# 3P Imports
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import color

# Local Imports

# plt.rcParams.update({'font.size': 22})


class ImageObject(object):

    """ Image Object encapsulates the meta data related to each image
    being processed by the front end system """

    def __init__(self, img, rgb=True, fc=0):
        """ Initialization """

        # Object members
        self.path = ''   # Folder Path
        self.name = ''   # Image Name
        self.ext = '.'   # Image Extension
        self.rgb = rgb  # RGB Boolean
        self.fc = 0      # Frame Count

        # Image Maps
        self.original = np.array([])  # Original Image
        self.modified = np.array([])  # Modified Image - Processed by Gamma Kernel
        self.patched = np.array([])  # Original Image with image patch bounding boxes
        self.patched_sequence = np.array([])  # Salience Maps processing sequence results
        self.ground_truth = np.array([])  # Ground Truth Map
        self.salience_map = np.array([])  # Saliency Map

        # Objects
        self.patch_list = []  # List of objects in the frame

        # Bounding Box Metadata
        self.gt_coords = []  # Coordinates derived from ground truth data
        self.bb_coords = []  # Bounding box coordinates by FoA - Ranked by intensity

        if (isinstance(img, str)):
            # Extract image name, path, and extension
            self.file_parse(img)
            self.original = cv2.imread(img, cv2.IMREAD_COLOR)
        else:
            self.original = img

        # Convert image to CIELAB Color Space
        self.image_convert()

    def file_parse(self, file=str):
        """ Discretize file path, name, and extension """

        # Current Directory
        for i, c in enumerate(file):
            if c == '/':
                file = file[i:]
                break

        # Extension
        s1 = file.split('.')
        self.ext += s1[-1]

        # Name
        s2 = s1[0].split('/')
        self.name = s2[-1]

        # Path
        s2.remove(s2[-1])
        for f in s2:
            self.path += (f + '/')

        if (self.path[0] == '/'):
            self.path = '.' + self.path

    def image_convert(self):
        """ Convert RGB image to CIELAB Color Space """

        if (self.rgb):  # Image is RGB
            try:
                # Read original image into object
                self.original = cv2.cvtColor(self.original, cv2.COLOR_BGR2RGB)
            except TypeError as e:
                print(e)

            # Convert RGB to CIELAB
            self.modified = color.rgb2lab(self.original)
        else:
            self.modified = self.original

    def display_map(self, img, title):
        """ Plot input image """

        plt.figure()
        plt.title(title)
        plt.imshow(img)
        plt.show()

    def plot_original_map(self):
        assert self.original is not None
        self.display_map(self.original, "Original Image")

    def plot_modified_map(self):
        assert self.modified is not None
        self.display_map(self.modified, "Modified Image")

    def plot_saliency_map(self):
        assert self.salience_map is not None
        self.display_map(self.salience_map, "Saliency Map")

    def plot_patched_map(self):
        assert self.patched is not None
        self.display_map(self.patched, "Patched Map")

    def plot_ground_truth(self):
        assert self.ground_truth is not None
        self.display_map(self.ground_truth, "Ground Truth")

    def plot_patches(self):
        assert self.patch_list is not None
        fig = plt.figure()
        rank = str(len(self.patch_list))
        fig.suptitle(f"Top {rank} Patches", y=0.8)
        for i, p in enumerate(self.patch_list):
            ax = fig.add_subplot(int("1" + rank + str(i + 1)))
            ax.imshow(p)
        plt.tight_layout()

    def plot_main(self):
        assert self.original is not None
        assert self.salience_map is not None
        assert self.patched is not None

        fig = plt.figure()
        ax1 = fig.add_subplot(131)
        ax1.imshow(self.original)
        plt.title("Original")
        ax2 = fig.add_subplot(132)
        ax2.imshow(self.salience_map)
        plt.title("Saliency")
        ax3 = fig.add_subplot(133)
        ax3.imshow(self.patched)
        plt.title("Patched")
        plt.show()

    def draw_image_patches(self, bbt=1, salmap=False):
        """ Apply bounding boxes to original image in the patched map.
        bbt - Bounding box line thickness
        salmap - Apply bounding boxes to saliency map """

        # Copy the original image
        self.patched = np.copy(self.original)

        # Draw bounding boxes for ground truth
        for o in self.gt_coords:
            lc = [100, 150, 255]  # Line Color - Blue
            self.__draw(o, lc, bbt, salmap)

        # Draw bounding boxes on max intensity regions
        for o in self.bb_coords:
            lc = [255, 150, 100]  # Line Color - Red
            self.__draw(o, lc, bbt, salmap)

    def __draw(self, obj, lc, bbt=int, salmap=bool):
        """ Helper method for drawing bounding boxes """

        # Get bounding coordinates
        a = int(obj["top_left"][0])  # R1
        b = int(obj["bottom_right"][0])  # R2
        c = int(obj["top_left"][1])  # C1
        d = int(obj["bottom_right"][1])  # C2

        # Allow for bounding boxes at the edge of the scene to be generated
        if (a - bbt < 0):
            a = bbt
        if (b + bbt > self.original.shape[0]):
            b = self.original.shape[0] - bbt - 1
        if (c - bbt < 0):
            c = bbt
        if (d + bbt > self.original.shape[1]):
            d = self.original.shape[1] - bbt - 1

        # Draw bounding boxes on patched map - +bbt and -bbt for for the bounding box
        if (self.rgb):  # RGB
            self.patched[a-bbt:b, c-bbt:c] = lc  # Left
            self.patched[a:b+bbt, d:d+bbt] = lc  # Right
            self.patched[a-bbt:a, c:d+bbt] = lc  # Top
            self.patched[b:b+bbt, c-bbt:d] = lc  # Bottom
            assert (self.patched[b:b+bbt, c-bbt:d] == lc).all()
            self.patch_list.append(self.original[a:b, c:d])
        else:
            self.patched[a-bbt:b, c-bbt:c] = self.patched.max()   # Left
            self.patched[a:b+bbt, d:d+bbt] = self.patched.max()   # Right
            self.patched[a-bbt:a, c:d+bbt] = self.patched.max()   # Top
            self.patched[b:b+bbt, c-bbt:d] = self.patched.max()   # Bottom
            self.patch_list.append(self.original[a:b, c:d])

        # Salience Map
        if (salmap):
            self.salience_map[a-bbt:b, c-bbt:c] = 1.0  # Left
            self.salience_map[a:b+bbt, d:d+bbt] = 1.0  # Right
            self.salience_map[a-bbt:a, c:d+bbt] = 1.0  # Top
            self.salience_map[b:b+bbt, c-bbt:d] = 1.0  # Bottom
            self.patch_list.append(self.salience_map[b:b+bbt, c-bbt:d])
