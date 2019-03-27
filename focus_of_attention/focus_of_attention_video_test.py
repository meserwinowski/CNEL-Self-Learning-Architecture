# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 11:55:52 2018

@author: meser

focus_of_attention_video_test.py - Test gamma kernel focus of attention
application on a bk2 video file. Script handles opening bk2 file, applying
front end system, generating relevant images, and stitching images into a new
mp4 video.

"""

# Standard Library Imports
import sys
import errno
import os
import time

# 3P Imports
import numpy as np
import matplotlib.pyplot as plt
import retro
import cv2
import scipy.io
import scipy.signal
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
    if (len(img.shape) != 2):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if (os.path.isdir(path)):
        cv2.imwrite(os.path.join(path, name), img)
    else:
        os.makedirs(path)
        cv2.imwrite(os.path.join(path, name), img)


def delete_images(path):
    for image in os.listdir(path):
        os.remove(os.path.join(path, image))


def delete_directory(path):
    for image in os.listdir(path):
        os.rmdir(os.path.join(path, image))
    try:
        os.rmdir(dir_path)  # Remove temporary image directory
    except OSError as e:
        print("Directory specified for removal probably not empty")
        print(e)


# Main Routine
if __name__ == '__main__':
    plt.close('all')

#    file = sys.argv[1]
    file = "./Super Mario Gym/Bink Video/SuperMarioWorld.bk2"
#    file = "./Super Mario Gym/Bink Video/SuperMarioWorld-Snes-Start-0000.bk2"

    print("Render File: ", file)

    # Generate Movie Hook
    movie = retro.Movie(file)
    print("movie.players: ", movie.players)

    # Generate retro emulation
    try:
        env = retro.make(game=movie.get_game(),
                         state=retro.State.DEFAULT,
                         use_restricted_actions=retro.Actions.ALL,
                         players=movie.players)
    except RuntimeError:
        env.close()
        env = retro.make(game=movie.get_game(),
                         state=retro.State.DEFAULT,
                         use_restricted_actions=retro.Actions.ALL,
                         players=movie.players)
    env.initial_state = movie.get_state()
    env.reset()
    frame = 0
    framerate = 2
    frame_count = 0

    # Working Directory + Name of bk2 file
    file = file[:-4].split('/')[-1]
    dir_path = os.getcwd() + '/Super Mario Gym/' + file + '/'
    ext = '.png'  # Image extension
    output = file + '.mp4'  # Video file extension
    output_patch = file + '_patch_' + '.mp4'
    images = []
    images_sal = []
    images_bb = []
    images_patch = []
    try:
        os.makedirs(dir_path)  # Create temporary image directory
    except OSError as e:
        if e.errno != errno.EEXIST:  # Handle directory already existing
            raise

    first = True

    # Step through rendered environment
    while movie.step():
        keys = []  # Boolean list of keys pressed during current frame
        for i in range(env.num_buttons):
            keys.append(movie.get_key(i, 0))

        # Input key presses into emulation environment
        # obs contains the actual RGB pixel data of the current frame
        obs, rew, done, info = env.step(keys)

        if frame == framerate:
            # Open AI Gym environment render wrapper
#            env.render()
            frame = 0
            frame_count += 1

            # Use OpenCV to generate a png image of the current frame
            image_name = str(frame_count) + ext
            save_image(obs, dir_path + "original/", image_name)

            # Create an image object of the current frame to be processed
            f = dir_path + "original/" + image_name
            image_curr = foa.imageObject(f, fc=frame_count)

            if (first):
                first = False
                # Get image dimensions: Mario - 224x256x3
                height, width, channels = obs.shape
                print("H: ", height, "W: ", width, "C: ", channels)

                # Generate Gaussian Blur Prior - Time ~0.0020006
                prior = foa.matlab_style_gauss2D(image_curr.modified.shape,
                                                 300)

                # Generate Gamma Kernel
                kernel = foa.gamma_kernel(image_curr)

            # Generate Saliency Map with Gamma Filter
            start = time.time()
            foa.foa_convolution(image_curr, kernel, prior)
            stop = time.time()
            print("Salience Map Generation: ", stop - start, " seconds")

            # Bound and Rank the most Salient Regions of Saliency Map
            foa.salience_scan(image_curr)

            # Use OpenCV to generate a png image of saliency map
            image_name = str(frame_count) + ext
            save_image(image_curr.salience_map * 255,
                       dir_path + "salience/", image_name)

            # Create image patches
            image_curr.save_image_patches(dir_path)
#            for i in range(len(image_curr.bb_coords)):
#                patch_name = 'patch_' + str(frame_count) + '_' + str(i) + ext
#                images_patch.append(patch_name)
#                patch_image = image_curr.original[image_curr.bb_coords[i][0]:
#                                                  image_curr.bb_coords[i][1],
#                                                  image_curr.bb_coords[i][2]:
#                                                  image_curr.bb_coords[i][3]]
#                cv2.imwrite(os.path.join(dir_path, patch_name),
#                            cv2.cvtColor(patch_image, cv2.COLOR_RGB2BGR))

        else:
            frame += 1

    # Convert generated images into an mp4
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(output, fourcc, 16.0, (width, height))
    out_patch = cv2.VideoWriter(output_patch, fourcc, 16.0, (32, 32))

    # Delete original images
    output_video(out, dir_path + "original/")
    delete_images(dir_path + "original/")

    # Delete saliency map images
    output_video(out, dir_path + "salience/")
    delete_images(dir_path + "salience/")

    # Delete original images with bounding box
#    output_video(out, dir_path, images_bb)
#    delete_images(dir_path, images_bb)

    # Delete image patches
    output_video(out_patch, dir_path + "patches/obj1/")
#    delete_images(dir_path, images_patch)

    # Release everything if job is finished
    out.release()
    out_patch.release()
    cv2.destroyAllWindows()  # Kills python process windows
#    delete_directory(dir_path)

    print("Number of Rendered Frames: ", frame_count)

    env.close()
