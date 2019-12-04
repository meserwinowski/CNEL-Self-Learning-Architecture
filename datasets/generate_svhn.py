# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 13:55:38 2019

@author: meser
"""

import sys
import os
import tqdm
import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import h5py

plt.rcParams.update(plt.rcParamsDefault)
sys.path.insert(0, "../focus_of_attention/")
import foa_image as foai
import foa_convolution as foac


class SVHNImage():
    def __init__(self, name, image, label, gt_coord):
        self.name = name
        self.image = image
        self.label = label
        self.gt_coord = gt_coord

    def __repr__(self):
        return "SVHN F1 Image Object: " + self.name

    def __len__(self):
        if (self.image):
            return self.image.shape
        else:
            return 0


def get_box_data(index, hdf5_data):
    """
    get `left, top, width, height` of each picture
    :param index:
    :param hdf5_data:
    :return:
    """
    meta_data = dict()
    meta_data['height'] = []
    meta_data['label'] = []
    meta_data['left'] = []
    meta_data['top'] = []
    meta_data['width'] = []

    def print_attrs(name, obj):
        vals = []
        if obj.shape[0] == 1:
            vals.append(obj[0][0])
        else:
            for k in range(obj.shape[0]):
                vals.append(int(hdf5_data[obj[k][0]][0][0]))
        meta_data[name] = vals

    box = hdf5_data['/digitStruct/bbox'][index]
    hdf5_data[box[0]].visititems(print_attrs)
    return meta_data


def get_name(index, hdf5_data):
    name = hdf5_data['/digitStruct/name']
    return ''.join([chr(v[0]) for v in hdf5_data[name[index][0]][()]])


def build_data_set(folder, data):
    size = mat_data['/digitStruct/name'].size
    # size = 10

    # Iterate through struct and extract the data to be reformatted
    data = []
    for _i in tqdm.tqdm(range(size)):
        name = get_name(_i, mat_data)  # Pull out the image name
        image_data = Image.open(folder + name)  # Open the actual image by name
        image_data.load()
        box = get_box_data(_i, mat_data)  # Pull out the label and bbox data
        label_data = box['label']  # Get the labels

        # Iterate through the labels to isolate each bounding box coordinate
        gt_coords = []
        for i, item in enumerate(label_data):
            gt_coord = {}
            gt_coord['top_left'] = np.array((int(box['top'][i]),
                                             int(box['left'][i])))
            gt_coord['bottom_right'] = np.array((int(box['top'][i] + box['height'][i]),
                                                 int(box['left'][i] + box['width'][i])))
            gt_coords.append(gt_coord)

        # Process an image with FoA as an example
        svhn_object = SVHNImage(name, image_data, label_data, gt_coords)
        data.append(svhn_object)
        # test = foai.ImageObject(image_data)
        # test.gt_coords = gt_coords
        # test.draw_image_patches()
        # test.plot_patched_map()

    return data


if __name__ == "__main__":
    """ Download and extract the SVHN Format 1 Data from http://ufldl.stanford.edu/housenumbers/
    and extract it to {path} and then run this script. """

    """ Format 1 (Multi-Resolution) Full SVHN """
    path = "./SVHN/format1/"

    # Training Data
    folder = "train/"
    mat_data = h5py.File(os.path.join(path + folder, 'digitStruct.mat'))  # Open MatLab struct with h5py
    train_data = build_data_set(path + folder, mat_data)

    # Dump to pickle file
    filename = path + "train.pickle"
    with open(filename, 'wb') as outfile:
        for d in train_data:
            pickle.dump(d, outfile, pickle.HIGHEST_PROTOCOL)

    # Test Data
    folder = "test/"
    mat_data = h5py.File(os.path.join(path + folder, 'digitStruct.mat'))  # Open MatLab struct with h5py
    test_data = build_data_set(path + folder, mat_data)

    # Dump to pickle file
    filename = path + "test.pickle"
    with open(filename, 'wb') as outfile:
        for d in test_data:
            pickle.dump(d, outfile, pickle.HIGHEST_PROTOCOL)

    # # Format 2 (32x32) Cropped SVHN
    # tv.datasets.SVHN("./SVHN", split="train", download=True)
    # svhn_data = loadmat("./SVHN/train_32x32.mat")
    # train_data = svhn_data['X']

    sys.exit(0)
