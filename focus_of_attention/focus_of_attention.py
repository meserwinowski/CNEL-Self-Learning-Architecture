# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 11:55:52 2018

@author: meser

fes_gamma.py - Initial front end system gamma kernel implemenation in Python.
Contains the current functional breakdown of the front end system (FES) gamma
kernel. Current implementation is intended to mimic the initial MATLAB
implementation and results created by Ryan Burt.

"""

# Standard Library Imports
import math
import os

# 3P Imports
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import io, color, transform
import scipy.io
import scipy.signal
from PIL import Image

# Local Imports

plt.rcParams.update({'font.size': 22})


class salObject():

    ''' Specific object detected in an image. The object contains coordinates
    to better track it in a pixel positional context. '''

    center_coord = []
    bb_coords = []
    patch_name_list = []

    def __init__(self, center_coord=center_coord):
        self.center_coord = [center_coord]
        self.bb_coords = []

    def center_check(self, new_center_coord):
        # Check if center_coord list is empty
        if not self.center_coord:
            self.center_coord = [new_center_coord]

        # Row Difference
        Rd = new_center_coord[0] - self.center_coord[-1][0]

        # Column Difference
        Cd = new_center_coord[1] - self.center_coord[-1][1]

        # 2D Eucledian Distance
        distance = (np.sqrt(Rd ** 2 + Cd ** 2))

        # If distance is within threshold...
        if (distance < 16):
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
        C1 = int(C - (boxSize['Column'] / 2))

        # Derive lower right coordinate of bounding region
        R2 = int(R + (boxSize['Row'] / 2))
        C2 = int(C + (boxSize['Column'] / 2))

        self.bb_coords.append([R1, R2, C1, C2])


class imageObject():

    ''' Image Object encapsulates the meta data related to each image
    being processed by the front end system '''

    path = './ICASSP Ryan/'  # Folder Path
    name = 'mario'  # Image Name
    ext = '.png'  # Image Extension
    rgb = True  # RGB Boolean
    fc = 0  # Frame Count

    # Gamma Filter Order, Shape, and Exponentiation Parameters
    k = np.array([1, 25, 1, 30, 1, 35], dtype=float)  # Orders
    mu = np.array([1, 2, 1, 2, 1, 2], dtype=float)  # Shapes
    alpha = 4  # Exponentiation

    # Image Maps
    original = np.array([])  # Original Image
    modified = np.array([])  # Modified Image - Processed by Gamma Kernel
    patched = np.array([])  # Original Image with image patch bounding boxes
    ground_truth = np.array([])  # Ground Truth Map
    salience_map = np.array([])  # Saliency Map

    # Objects
    objects = []  # List of objects in the frame

    # Bounding Box Metadata
    bb_coords = []  # Bounding Box Coordinates - Ranked by order in the list
    center_coord = []  # Approximate center pixel of objects

    def __init__(self, path=path, name=name, extension=ext, rgb=rgb, fc=fc):
        self.path = path
        self.name = name
        self.ext = extension
        self.rgb = rgb
        self.fc = fc

    def image_convert(self):

        ''' Get image from path and convert RGB image to CIELAB Color Space '''

        if (self.rgb):  # If image is RGB...
            # Read original image into object
            self.original = io.imread(self.path +
                                      self.name +
                                      self.ext)

            # Convert RGB to CIELAB
            self.modified = color.rgb2lab(self.original)

        else:  # Image is Gray
            # Rename image object
            self.name = self.name + '_gray'

            # Reopen image as gray scale
            gim = Image.open(self.path +
                             self.name +
                             self.ext).convert('LA')
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

    def display_image(self, img, t):
        plt.figure()
        plt.title(t)
        plt.imshow(img)
        plt.show()

    def plot_saliency_map(self):
        self.display_image(self.salience_map, "Saliency Map")

    def draw_image_patches(self, bbt=1, plot=False):

        ''' Apply bounding boxes to original image in the patched map. Boolean
        value decides if the patches from the image will be saved. '''

        # Copy the original image
        self.patched = np.copy(self.original)

        # Place a bounding box on max intensity regions
        for o in self.objects:
            print("imagePatch2: ", o)
            print("salScan2: ", o.bb_coords)

            # Grab bounding coordinates
            a = o.bb_coords[-1][0]
            b = o.bb_coords[-1][1]
            c = o.bb_coords[-1][2]
            d = o.bb_coords[-1][3]

            # Draw bounding boxes on patched map
            if (self.rgb):
                lc = [255, 150, 100]  # Line Color (RGB)
                self.patched[a:b, c-bbt:c] = lc
                self.patched[a:b, d:d+bbt] = lc
                self.patched[a-bbt:a, c:d] = lc
                self.patched[b:b+bbt, c:d] = lc
            else:
                self.patched[a:b, c:c-bbt] = [255]
                self.patched[a:b, d:d-bbt] = [255]
                self.patched[a-bbt:a, c:d+bbt] = [255]
                self.patched[b:b+bbt, c:d+bbt] = [255]

    def save_image_patches(self, dir_path: str):

        ''' Save image patches from the current image to the specified
        directory. '''

        # Create image patches
        for i in range(len(self.bb_coords)):
            patch_name = 'patch_' + self.fc + '_' + str(i) + self.ext
            self.patch_name_list.append(patch_name)
            patch_image = self.original[self.bb_coords[i][0]:
                                        self.bb_coords[i][1],
                                        self.bb_coords[i][2]:
                                        self.bb_coords[i][3]]
            cv2.imwrite(os.path.join(dir_path, patch_name),
                        cv2.cvtColor(patch_image, cv2.COLOR_RGB2BGR))


def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):

    ''' 2D Gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian', [shape], [sigma])
        Machine Epsilon - Smallest discrete difference between numbers
    where they are numerically the same; determined by data type '''
    # Get 2D Gaussian Dimensional Lengths
    m, n = [(ss - 1) / 2 for ss in shape[:2]]

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


def salScan2(image, rankCount=5, boundLength=32):
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
        R = indices[0][0]
        C = indices[1][0]

        # If list of objects is empty, add a new object
        flag = False
        for o in image.objects:
            # Check for an object with center coordinates near new coords
            if o.center_check([R, C]):
                # Get updated object
                obj = o
                flag = True
                break
        if not flag:
            # Create a new object if returned false
            image.objects.append(salObject([R, C]))
            obj = image.objects[-1]

        obj.build_bounding_box()
        print("salScan2: ", obj)
        print("salScan2: ", obj.center_coord)

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


def salScan(image, rankCount=5, boundLength=32):

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
    image.bb_coords = []
    total_old = 0
    for i in range(rankCount):

        # Grab Maximally Intense Pixel Coordinates (Object Center)
        indices = np.where(smap == smap.max())
        R = indices[0][0]
        C = indices[1][0]

        # Use defined gamma kernel orders to bound pixel distances
        for gk in range(1, len(image.k), 2):
            boundLength = image.k[-1]
            boxSize = {'Row': boundLength, 'Column': boundLength}

            # Derive upper left coordinate of bounding region
            R1 = int(R - (boxSize['Row'] / 2))
            C1 = int(C - (boxSize['Column'] / 2))

            # Derive lower right coordinate of bounding region
            R2 = int(R + (boxSize['Row'] / 2))
            C2 = int(C + (boxSize['Column'] / 2))

            # "Zero" the maximally intense region to avoid grabbing it again
            # Sum up and find the average intensity of the region
            total = 0
            for j in range(R1, R2):
                for k in range(C1, C2):
                    if ((j < image.original.shape[0]) and
                       (k < image.original.shape[1])):
                        total += image.salience_map[j][k]
                        smap[j][k] = 0

#            total /= (boundLength ** 2)
#            if (total >= total_old):
#                total_old = total
#                pass
#            else:
                # Append coordinates to image object's bounding box member
            image.bb_coords.append([R1, R2, C1, C2])
            image.center_coord.append([indices[0][0], indices[1][0]])
            break


def front_end_convolution(image, prior, maskSize=50):

    ''' Front End System Gamma (FES_Gamma)
    image - Input Image Object - Contains CIELAB Color Space Image
    k - vector contained kernel orders
    mu - vector containing shape parameters
    alpha - exponent on the saliency
    p1 - prior learned gaussian (data in prior.mat)

    Create a 2D gamma kernel and convolve it with the input image to generate
    a saliency map '''

    blurSize = 3  # Gaussian Blur - Odd Size Required
    k = image.k
    mu = image.mu
    alpha = image.alpha

    # Create Kernels and Kernel Mask
    g = np.zeros((len(mu), 2 * maskSize + 1, 2 * maskSize + 1))
    n = np.arange(-maskSize, maskSize + 1)
    gk = np.zeros((2 * maskSize + 1, 2 * maskSize + 1))

    # Meshgrid creates 2D coordinates out of vectors
    NX, NY = np.meshgrid(n, n)

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
        g[i] = (mu[i] ** (k[i] + 1)) / (2 * np.pi * math.factorial(k[i])) * \
               (supgrid ** ((k[i] - 1) * 0.5)) * \
               (np.exp(-mu[i] * (supgrid ** 0.5)))

    # Normalize Kernels
    for i in range(len(mu)):
        g[i] = g[i] / sum(sum(g[i]))

    ''' "For multiscale saliency measure, we simply combine multiple kernels
    of different sizes before the convolution stage. Kernel with larger center
    scale is subtracted by a surround kernel with a larger and further removed
    neighborhood, effectively searching for larger objects by comparing more
    overall area in the image. Kernel summation described in paper. '''

    # Combine Kernels - Center needs to be subtracted out each time
    for i in range(len(mu)):
        gk += g[i] * ((-1) ** i)

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
        sal_map = (saliency[:, :, 0] +
                   saliency[:, :, 1] +
                   saliency[:, :, 2]) / 3

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


def imagePatch2(image, bbt=1, p=False):
    if (p):  # If plot is True - Plot the image patches
        fig, axes = plt.subplots(1, 5, sharey=True)

    # Box max intensity regions, and place bounding box on original image
    for o in image.objects:
        print("imagePatch2: ", o)
        print("salScan2: ", o.bb_coords)
        # Grab bounding coordinates
        a = o.bb_coords[-1][0]
        b = o.bb_coords[-1][1]
        c = o.bb_coords[-1][2]
        d = o.bb_coords[-1][3]

#        # Generate intense region subplots
#        if (p):
#            try:
#                axes[i].imshow(image.original[a:b, c:d])
#                axes[i].set_title("X: {}, Y: {}".format(
#                        image.center_coord[i][0], image.center_coord[i][1]))
#            except IndexError:
#                pass

        # Update original image
        if (image.rgb):
            image.original[a:b, c-bbt:c] = [255, 150, 100]
            image.original[a:b, d:d+bbt] = [255, 150, 100]
            image.original[a-bbt:a, c:d] = [255, 100, 100]
            image.original[b:b+bbt, c:d] = [255, 100, 100]
        else:
            image.modified[a:b, c:c-bbt] = [255]
            image.modified[a:b, d:d-bbt] = [255]
            image.modified[a-bbt:a, c:d+bbt] = [255]
            image.modified[b:b+bbt, c:d+bbt] = [255]

    plt.show()


def imagePatch(image, bbt=1, p=False):

    if (p):  # If plot is True - Plot the image patches
        fig, axes = plt.subplots(1, 5, sharey=True)

    # Box max intensity regions, and place bounding box on original image
    for i in range(len(image.bb_coords)):

        # Grab bounding coordinates
        a = image.bb_coords[i][0]
        b = image.bb_coords[i][1]
        c = image.bb_coords[i][2]
        d = image.bb_coords[i][3]

        # Generate intense region subplots
        if (p):
            try:
                axes[i].imshow(image.original[a:b, c:d])
                axes[i].set_title("X: {}, Y: {}".format(
                        image.center_coord[i][0], image.center_coord[i][1]))
            except IndexError:
                pass

        # Update original image
        if (image.rgb):
            image.original[a:b, c-bbt:c] = [255, 150, 100]
            image.original[a:b, d:d+bbt] = [255, 150, 100]
            image.original[a-bbt:a, c:d] = [255, 100, 100]
            image.original[b:b+bbt, c:d] = [255, 100, 100]
        else:
            image.modified[a:b, c:c-bbt] = [255]
            image.modified[a:b, d:d-bbt] = [255]
            image.modified[a-bbt:a, c:d+bbt] = [255]
            image.modified[b:b+bbt, c:d+bbt] = [255]

    plt.show()