# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 09:04:10 2019

@author: meser
"""

# Standard Library Imports
import time
from os import listdir
from os.path import isfile, join

# 3P Imports
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import scipy.io
from sklearn.metrics import confusion_matrix

# Local Imports
import foa_image as foai
import foa_convolution as foac
import foa_saliency as foas

plt.rcParams.update({'font.size': 22})
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
           'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']


def compare_images(img1, img2, method):
    res = cv2.matchTemplate(img1, img2, eval(method))
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    return res.mean()


if __name__ == "__main__":
    # Import images into a list
    mypath = "./test_patches/"
    list_dir = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    print(list_dir)
    image_list = []
    for image in list_dir:
#        image_list.append(cv2.cvtColor(cv2.imread(mypath + image,
#                                                  cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))
        image_list.append(cv2.imread(mypath + image, 0))

    # Loop through all image pairs to create similarity matrix
    image_matrix = np.zeros(shape=(len(image_list), len(image_list)))
    row = 0
    for i in image_list:
#        print("row: ", row)
        col = 0
        for j in image_list:
#            print("col: ", col)
            image_matrix[row, col] = compare_images(i, j, methods[1])
            col = col + 1
        row = row + 1

    # Plot matrix
    y_true = [2, 0, 2, 2, 0, 1]
    y_pred = [0, 0, 2, 2, 0, 2]
    cm = confusion_matrix(y_true, y_pred, )

    fig, ax = plt.subplots()
    im = ax.imshow(image_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=list(range(16)), yticklabels=range(16),
           title="Image Similarity",
           ylabel='True label',
           xlabel='Predicted label')
    plt.title("Image Similarity")

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    plt.show()
