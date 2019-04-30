# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 11:55:52 2018

@author: meser

focus_of_attention_video_test.py - Open an mp4 file and apply the focus of
attention algorithm to the video.

"""

# Standard Library Imports
import sys
import os
import time
import errno

# 3P Imports
import matplotlib.pyplot as plt
import numpy as np
import cv2
import re
from PIL import Image

# Local Imports
import focus_of_attention as foa

plt.rcParams.update({'font.size': 22})

# Parameters
number_objects = 3
numbers = re.compile(r'(\d+)')


def numericalSort(x):
    parts = numbers.split(x)
    parts[1::2] = map(int, parts[1::2])
    return parts


def output_video(output, path):
    for image in sorted(os.listdir(path), key=numericalSort):
        image_path = os.path.join(path, image)  # Grab image path
        frame = cv2.imread(image_path)  # Grab image data from path
        output.write(frame)  # Write out frame to video
        if (cv2.waitKey(1) & 0xFF) == ord('q'):  # Hit 'q' to exit
            break


def save_image(img, path, name):
    os.makedirs(path, exist_ok=True)
    cv2.imwrite(os.path.join(path, name), img)


def delete_images(path):
    for image in os.listdir(path):
        os.remove(os.path.join(path, image))


def delete_directory(path):
    for image in os.listdir(path):
        os.rmdir(os.path.join(path, image))
    try:
        os.rmdir(path)  # Remove temporary image directory
    except OSError as e:
        print("Directory specified for removal probably not empty")
        print(e)


# Main Routine
if __name__ == '__main__':
    plt.close('all')

#    file = sys.argv[1]
#    file = "./SuperMarioWorld.mp4"
    file = "./SuperMarioWorld-Snes-YoshiIsland1-0001.mp4"

    print("Render File: ", file)

    # Open mp4 file
    movie = cv2.VideoCapture(file)

    frame = 0
    framerate = 1
    frame_count = 0

    # Working Directory + Name of bk2 file
    file = file[:-4].split('/')[-1]  # Get file name
    dir_path = os.getcwd() + '/Super Mario Gym/' + file + '/'
    ext = '.png'  # Image extension
    output = file + '.mp4'  # Video file extension
    output_patch = file + '_patch_' + '.mp4'

    # Create temporary image directory
    os.makedirs(dir_path, exist_ok=True)

    # Read first frame
    success, image = movie.read()
    print("FIRST READ SUCCESS: ", success)

    # Get image dimensions: Mario - 224x256x3
    height, width, channels = image.shape
    print("H: ", height, "W: ", width, "C: ", channels)
    start = time.time()

    # Generate Gaussian Blur Prior - Time ~0.0020006
    prior = foa.matlab_style_gauss2D(image.shape, 300)

    # Generate Gamma Kernel
    image_curr = foa.imageObject(image, fc=frame)
    image_prev = image_curr
    image_prev.salience_map = np.zeros(image_prev.original.shape[:-1])
    kernel = foa.gamma_kernel(image_curr)

    # Step through each movie frame
    while success:
        frame += 1
        image_name = str(frame) + ext
        image_curr = foa.imageObject(image, fc=frame)

        # Generate Saliency Map with Gamma Filter
        foa.foa_convolution(image_curr, kernel, prior)

        if (frame % 100 == 0):
            stop = time.time()
            print("Salience Map Gen ", frame, ": ", stop - start, " seconds")
            start = time.time()

        # Bound and Rank the most Salient Regions of Saliency Map
#        image_curr.salience_map = np.subtract(image_curr.salience_map, image_prev.salience_map)
#        image_curr.salience_map *= (image_curr.salience_map > 0.12)
        foa.salScan(image_curr)

        # Bounding Box images
        image_curr.draw_image_patches(salmap=False)
        save_image(image_curr.salience_map * 255, dir_path + "salience/",
                   image_name)
        save_image(image_curr.patched, dir_path + "bounding_box/",
                   image_name)
        
        # Create combined image
        img1 = Image.open(dir_path + "salience/" + image_name)
        img2 = Image.open(dir_path + "bounding_box/" + image_name)
        result = Image.new('RGB', (2 * width, height))
        result.paste(im=img1, box=(0, 0))
        result.paste(im=img2, box=(width, 0))
        save_image(cv2.cvtColor(np.array(result), cv2.COLOR_BGR2RGB),
                   dir_path + "combine/", image_name)

        # Next Frame
        success, image = movie.read()
        image_prev = image_curr

    # Convert generated images into an mp4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#    out = cv2.VideoWriter(output, fourcc, 60.0, (width, height))
#    out_patch = cv2.VideoWriter(output_patch, fourcc, 16.0, (32, 32))

    # Combined saliency map and bounding box images
    print("Saliency/Bounding Box Video +  Delete Images")
    output = file + "_combine" + ".mp4"
    out = cv2.VideoWriter(output, fourcc, 60.0, (2 * width, height))
    output_video(out, dir_path + "combine/")
    out.release()
    delete_images(dir_path + "combine/")

#    # Delete saliency map images
#    print("Saliency Video + Delete Images")
#    output = file + "_sal" + ".mp4"
#    out = cv2.VideoWriter(output, fourcc, 60.0, (width, height))
#    output_video(out, dir_path + "salience/")
#    out.release()
    delete_images(dir_path + "salience/")
#
#    # Delete  bounding box images
#    print("Bounding Box Video + Delete Images")
#    output = file + "_bb" + ".mp4"
#    out = cv2.VideoWriter(output, fourcc, 60.0, (width, height))
#    output_video(out, dir_path + "bounding_box/")
#    out.release()
    delete_images(dir_path + "bounding_box/")

    # Delete image patches
#    output_video(out_patch, dir_path + "patches/obj1/")
#    delete_images(dir_path, images_patch)

#    out_patch.release()
    movie.release()
    cv2.destroyAllWindows()  # Kills python process windows
    delete_directory(dir_path)

    print("Number of Rendered Frames: ", frame)
