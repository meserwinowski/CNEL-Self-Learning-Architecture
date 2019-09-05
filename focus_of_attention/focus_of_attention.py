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
import math
import os
import errno

# 3P Imports
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import io, color, transform
import scipy.io
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

    def object_compare(self, new_coord, frame):
        for o in frame.objects:
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


class imageObject():

    ''' Image Object encapsulates the meta data related to each image
    being processed by the front end system '''

    path = ''   # Folder Path
    name = ''     # Image Name
    ext = '.'  # Image Extension
    rgb = True    # RGB Boolean
    fc = 0        # Frame Count

    # Gamma Filter Order, Shape, and Exponentiation Parameters
    k = np.array([1, 25, 1, 30, 1, 35], dtype=float)  # Orders
    mu = np.array([4, 4, 4, 4, 4, 4], dtype=float)  # Shapes
    alpha = 3 # Exponentiation

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

    def __init__(self, img, rgb=rgb, fc=0):
        
        ''' Initialization '''

        # Extract image name, path, and extension
        self.file_parse(img)
        self.original = img

        # Convert image to CIELAB Color Space
        self.image_convert()

        self.rgb = rgb
        self.fc = fc

    def file_parse(self, file):

        ''' Discretize file path, name, and extension '''

        if (file[0] == '.'):
            file = file[1:]
        s1 = file.split('.')
        self.ext += s1[-1]
        s2 = s1[0].split('/')
        self.name = s2[-1]
        s2.remove(s2[-1])
        for f in s2:
            self.path += (f + '/')
        if (self.path[0] == '/'):
            self.path = '.' + self.path

    def image_convert(self):

        ''' Get image from path and convert RGB image to CIELAB Color Space '''

        if (self.rgb):  # If image is RGB...
            # Read original image into object
            self.original = io.imread(self.path + self.name + self.ext)

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

    def display_image(self, img, t):
        plt.figure()
        plt.title(t)
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
                lc = [100, 150, 255]  # Line Color
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

        # Create temporary image directory
#        try:
        os.makedirs(save_path, exist_ok=True)
#        except OSError as e:
#            pass
        try:
            for i in range(3):
                subf = 'obj' + str(i) + '/'
                os.makedirs(save_path + subf)
        except OSError:
            pass
        i = 0
        for o in self.objects:
            patch_name = str(self.fc) + '_' + str(i) + self.ext
            subf = 'obj' + str(i) + '/'
            o.patch_name_list.append(patch_name)
            patch_image = self.original[o.bb_coords[-1][0]:o.bb_coords[-1][1],
                                        o.bb_coords[-1][2]:o.bb_coords[-1][3]]
            cv2.imwrite(os.path.join(save_path + subf, patch_name),
                        cv2.cvtColor(patch_image, cv2.COLOR_RGB2BGR))
            i += 1


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


def salience_scan(image, rankCount=3, boundLength=32):

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
        for o in image.objects:
            # Check for an object with center coordinates near new coords
            if o.center_check([R, C], image):
                # Get updated object
                obj = o
                flag = True
                break
        if not flag:
            # Create a new object if returned false
            image.objects.append(salObject([R, C]))
            obj = image.objects[-1]

        obj.build_bounding_box()

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


def salScan(image, rankCount=3, boundLength=32):

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
        try:
            R = indices[0][0]
            C = indices[1][0]
            if (smap[R, C] < 0.137):
                continue
        except IndexError:
            print("Image is probably black")
            R = boundLength
            C = boundLength

        # Use defined gamma kernel orders to bound pixel distances
        for gk in range(1, len(image.k), 2):
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
#            image.center_coord.append([indices[0][0], indices[1][0]])
            break


def gamma_kernel(image, maskSize=16, d=2):

    ''' Generate a 2D Gamma Kernel
    k - vector contained kernel orders
    mu - vector containing shape parameters '''

    k = image.k
    mu = image.mu

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

    return gk


def foa_convolution(image, kernel, prior):

    ''' Focus of Attention Convolution
    image - Input Image Object - Contains CIELAB Color Space Image
    kernel - Matrix for filtering the image
    alpha - exponent on the saliency
    prior - foveation prior

    Create a 2D gamma kernel and convolve it with the input image to generate
    a saliency map '''

    blurSize = 3  # Gaussian Blur - Odd Size Required
    alpha = image.alpha
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
