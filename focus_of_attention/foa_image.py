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
import errno
import os

# 3P Imports
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import color
from PIL import Image

# Local Imports

plt.rcParams.update({'font.size': 22})


class ImageObject():

    ''' Image Object encapsulates the meta data related to each image
    being processed by the front end system '''

    path = ''   # Folder Path
    name = ''   # Image Name
    ext = '.png'   # Image Extension
    rgb = True  # RGB Boolean
    fc = 0      # Frame Count

    # Gamma Filter Order, Shape, and Exponentiation Parameters
    k = np.array([1, 25, 1, 30, 1, 35], dtype=float)  # Orders
    mu = np.array([4, 4, 4, 4, 4, 4], dtype=float)  # Shapes
    alpha = 3  # Exponentiation

    # Image Maps
    original = np.array([])  # Original Image
    modified = np.array([])  # Modified Image - Processed by Gamma Kernel
    patched = np.array([])  # Original Image with image patch bounding boxes
    ground_truth = np.array([])  # Ground Truth Map
    salience_map = np.array([])  # Saliency Map

    # Objects
    patch_list = []  # List of objects in the frame

    # Bounding Box Metadata
    bb_coords = []  # Bounding Box Coordinates - Ranked by order in the list
    center_coord = []  # Approximate center pixel of objects

    def __init__(self, img, rgb=rgb, fc=0):

        ''' Initialization '''

        self.bb_coords = []

        if (isinstance(img, str)):
            # Extract image name, path, and extension
            self.file_parse(img)
            image_path = self.path + self.name + self.ext
            self.original = cv2.imread(image_path, cv2.IMREAD_COLOR)
        else:
            self.original = img

        # Convert image to CIELAB Color Space
        self.image_convert()

        self.rgb = rgb
        self.fc = fc

    def file_parse(self, file):

        ''' Discretize file path, name, and extension '''

        if (file[0] == '.'):
            file = file[1:]

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

        ''' Get image from path and convert RGB image to CIELAB Color Space '''

        if (self.rgb):  # If image is RGB...
            # Read original image into object
            self.original = cv2.cvtColor(self.original, cv2.COLOR_BGR2RGB)

            # Convert RGB to CIELAB
            self.modified = color.rgb2lab(self.original)

        else:  # Image is Gray
            # Rename image object
            self.name = self.name + '_gray'

            # Reopen image as gray scale
#            gim = Image.open(self.path + self.name + self.ext).convert('LA')
            gim = self.original.convert('LA')
            self.img = np.array(gim)[:, :, 0]

    def gray_convert(self):

        ''' Convert an image to gray scale '''

        # Attempt Gray Scale Conversion - Saves new gray scale image
        try:
            # Open original image as gray scale
            gim = Image.open(self.path +
                             self.name +
                             self.ext).convert('LA')

            # Save a gray version for future reference
            gim.save(self.path + self.name + '_gray' + self.ext)
        except IOError:
            try:
                # Try saving as .png if default extension fails
                gim.save(self.path + self.name + '_gray' + '.png')
            except IOError:
                print("Gray conversion save failed")

    def display_image(self, img, title):
        plt.figure()
        plt.title(title)
        plt.imshow(img)
        plt.show()

    def plot_original_map(self):
        self.display_image(self.original, "Original Image")

    def plot_saliency_map(self):
        self.display_image(self.salience_map, "Saliency Map")

    def draw_image_patches(self, bbt=1, salmap=True, plot=False):

        ''' Apply bounding boxes to original image in the patched map. Boolean
        value decides if the patches from the image will be saved. '''

        # Copy the original image
        self.patched = np.copy(self.original)

        # Place a bounding box on max intensity regions
        for o in self.bb_coords:

            # Grab bounding coordinates
            a = o[0]
            b = o[1]
            c = o[2]
            d = o[3]

            # Draw bounding boxes on patched map
            if (self.rgb):  # RGB
                lc = [255, 150, 100]  # Line Color
                self.patched[a:b, c-bbt:c] = lc  # Left
                self.patched[a:b, d:d+bbt] = lc  # Right
                self.patched[a-bbt:a, c:d] = lc  # Top
                self.patched[b:b+bbt, c:d] = lc  # Bottom
                if (salmap):  # Salience Map
                    self.salience_map[a:b, c-bbt:c] = 1.0  # Left
                    self.salience_map[a:b, d:d+bbt] = 1.0  # Right
                    self.salience_map[a-bbt:a, c:d] = 1.0  # Top
                    self.salience_map[b:b+bbt, c:d] = 1.0  # Bottom
            else:  # Gray
                self.patched[a:b, c-bbt:c] = [255]  # Left
                self.patched[a:b, d:d+bbt] = [255]  # Right
                self.patched[a-bbt:a, c:d] = [255]  # Top
                self.patched[b:b+bbt, c:d] = [255]  # Bottom

    def save_image_patches(self, path=os.getcwd()):

        ''' Save image patches from the current image to
        specified directories. '''

        # Create image patches
        save_path = path + '/patches/'
        print("SAVE: ", save_path)

        # Create temporary image directory
        os.makedirs(save_path, exist_ok=True)
        try:
            for i in range(len(self.patch_list)):
                subf = 'obj' + str(i) + '/'
                print(subf)
                os.makedirs(save_path + subf)
        except OSError:
            pass
        i = 0

        for o in self.patch_list:
            patch_name = str(self.fc) + '_' + str(i) + self.ext
            subf = 'obj' + str(i) + '/'
            o.patch_name_list.append(patch_name)
            patch_image = self.original[o.bb_coords[-1][0]:o.bb_coords[-1][1],
                                        o.bb_coords[-1][2]:o.bb_coords[-1][3]]
            cv2.imwrite(os.path.join(save_path + subf, patch_name),
                        cv2.cvtColor(patch_image, cv2.COLOR_RGB2BGR))
            i += 1
