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

# Local Imports
import focus_of_attention as foa

plt.rcParams.update({'font.size': 22})

# Parameters
number_objects = 3

# Main Routine
if __name__ == '__main__':
    plt.close('all')

#    file = sys.argv[1]
    file = "./Super Mario Gym/human/SuperMarioWorld-Snes/scenario/SuperMarioWorld.bk2"
#    file = "./Super Mario Gym/human/SuperMarioWorld-Snes/scenario/SuperMarioWorld-Snes-Start-0000.bk2"

    print("Render File: ", file)

    # Generate Movie Hook
    movie = retro.Movie(file)
    print("movie.players: ", movie.players)

    # Generate retro emulation
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
    dir_path = os.getcwd() + '/' + file[:-4] + '/'
    ext = '.png'  # Image extension
    output = file[:-4] + '.mp4'  # Video file extension
    images = []
    images_sal = []
    images_bb = []
    images_patch = []
    try:
        os.makedirs(dir_path)  # Create temporary image directory
    except OSError as e:
        if e.errno != errno.EEXIST:  # Handle directory already existing
            raise

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
            cv2.imwrite(os.path.join(dir_path, image_name),
                        cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
            images.append(image_name)

            # Create an image object of the current frame for FESGK
            image_curr = foa.imageObject(path=dir_path,
                                        name=str(frame_count),
                                        extension='.png',
                                        RGB=True)

            # Convert image to CIELAB Color Space - Resize Image and
            # create gray scale version if required
            foa.convert(image_curr)

            # Generate Gaussian Blur Prior - Time ~0.0020006
            prior = foa.matlab_style_gauss2D((image_curr.modified.shape[0],
                                             image_curr.modified.shape[1]),
                                            sigma=300)

            # Generate Saliency Map with Gamma Filter
            start = time.time()
            foa.FES_Gamma(image_curr, image_curr.k, image_curr.mu,
                         image_curr.alpha, prior)
            stop = time.time()
#            print("Salience Map Generation: ", stop - start, " seconds")

            # Bound and Rank the most Salient Regions of Saliency Map
            foa.salScan(image_curr, rankCount=number_objects)

#            # Draw bounding boxes on original images
#            foa.imagePatch(image_curr)
#            image_name = str(frame_count) + '_bb' + ext
#            cv2.imwrite(os.path.join(dir_path, image_name),
#                        cv2.cvtColor(image_curr.original, cv2.COLOR_RGB2BGR))
#            images_bb.append(image_name)

            # Use OpenCV to generate a png image of saliency map
            image_name = 'sal_' + str(frame_count) + ext
            cv2.imwrite(os.path.join(dir_path, image_name),
                        image_curr.salience_map * 255)
            images_sal.append(image_name)

            # Create image patches
            for i in range(len(image_curr.bb_coords)):
                patch_name = 'patch_' + str(frame_count) + '_' + str(i) + ext
                images_patch.append(patch_name)
                patch_image = image_curr.original[image_curr.bb_coords[i][0]:
                                                  image_curr.bb_coords[i][1],
                                                  image_curr.bb_coords[i][2]:
                                                  image_curr.bb_coords[i][3]]
                cv2.imwrite(os.path.join(dir_path, patch_name),
                            cv2.cvtColor(patch_image, cv2.COLOR_RGB2BGR))

        else:
            frame += 1

    # Get image dimensions
    image_path = os.path.join(dir_path, images_sal[0])
    frame = cv2.imread(image_path)
    height, width, channels = frame.shape
    print("H: ", height, "W: ", width, "C: ", channels)
    image_path = os.path.join(dir_path, images[0])
    frame = cv2.imread(image_path)
    height, width, channels = frame.shape

    # Convert generated images into an mp4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output, fourcc, 20.0, (width, height))

    # Delete saliency map images
    for image in images_sal:
        image_path = os.path.join(dir_path, image)  # Grab image path
        frame = cv2.imread(image_path)  # Grab image data from path
        out.write(frame)  # Write out frame to video
        os.remove(image_path)  # Delete png image
        if (cv2.waitKey(1) & 0xFF) == ord('q'):  # Hit 'q' to exit
            break

    # Delete regular images
    for image in images:
        image_path = os.path.join(dir_path, image)  # Grab image path
#        frame = cv2.imread(image_path)  # Grab image data from path
#        out.write(frame)  # Write out frame to video
        os.remove(image_path)  # Delete png image
        if (cv2.waitKey(1) & 0xFF) == ord('q'):  # Hit 'q' to exit
            break

    # Delete original images with bounding box
    for image in images_bb:
        image_path = os.path.join(dir_path, image)  # Grab image path
        frame = cv2.imread(image_path)  # Grab image data from path
        out.write(frame)  # Write out frame to video
        os.remove(image_path)  # Delete png image
        if (cv2.waitKey(1) & 0xFF) == ord('q'):  # Hit 'q' to exit
            break

    # Delete image patches
#    for image in images_patch:
#        image_path = os.path.join(dir_path, image)  # Grab image path
##        frame = cv2.imread(image_path)  # Grab image data from path
##        out_patches.write(frame)  # Write out frame to video
#        os.remove(image_path)  # Delete png image
#        if (cv2.waitKey(1) & 0xFF) == ord('q'):  # Hit 'q' to exit
#            break

    # Release everything if job is finished
    out.release()
    cv2.destroyAllWindows()  # Kills python process windows
    os.rmdir(dir_path)  # Remove temporary image directory

    # 224x256x3
    print("H: ", height, "W: ", width, "C: ", channels)
    print("Number of Rendered Frames: ", frame_count)

    env.close()
