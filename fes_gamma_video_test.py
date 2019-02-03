# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 11:55:52 2018

@author: meser
"""

import fes_gamma as fg
import retro
import os
import errno
import sys
import cv2
import scipy.io
import scipy.signal
import time
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})


# Main Routine
if __name__ == '__main__':
    plt.close('all')

#    file = sys.argv[1]
    file = "SuperMarioWorld.bk2"

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
    framerate = 32
    frame_count = 0

    # Working Directory + Name of bk2 file
    dir_path = os.getcwd() + '/' + file[:-4] + '/'
    ext = '.png'  # Image extension
    output = file[:-4] + '.mp4'  # Video file extension
    images = []
    images_sal = []
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
            image_curr = fg.imageObject(path=dir_path,
                                        name=str(frame_count),
                                        extension='.png',
                                        RGB=True)

            # Convert image to CIELAB Color Space - Resize Image and
            # create gray scale version if required
            fg.convert(image_curr)

        # %% Generate Gamma Kernel and Saliency Map
            # Set Gamma Filter Orders and Shape parameters
            k = np.array([1, 20, 1, 30, 1, 40], dtype=float)
            mu = np.array([2, 2, 2, 2, 2, 2], dtype=float)
            alpha = 5

            # Generate Gaussian Blur Prior - Time ~0.0020006
            prior = fg.matlab_style_gauss2D((image_curr.img.shape[0],
                                             image_curr.img.shape[1]),
                                            sigma=300)

            # Generate Saliency Map with Gamma Filter
            start = time.time()
            fg.FES_Gamma(image_curr, k, mu, alpha, prior)
            stop = time.time()
            print("Salience Map Generation: ", stop - start, " seconds")

            # Bound and Rank the most Salient Regions of Saliency Map
            fg.salScan(image_curr)

            # Use OpenCV to generate a png image of saliency map
            image_name = 'sal_' + str(frame_count) + ext
            cv2.imwrite(os.path.join(dir_path, image_name),
                        image_curr.salience_map * 255)
            images_sal.append(image_name)

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

    for image in images_sal:
        image_path = os.path.join(dir_path, image)  # Grab image path
        frame = cv2.imread(image_path)  # Grab image data from path
        out.write(frame)  # Write out frame to video
        os.remove(image_path)  # Delete png image
        if (cv2.waitKey(1) & 0xFF) == ord('q'):  # Hit 'q' to exit
            break

    for image in images:
        image_path = os.path.join(dir_path, image)  # Grab image path
        frame = cv2.imread(image_path)  # Grab image data from path
        out.write(frame)  # Write out frame to video
        os.remove(image_path)  # Delete png image
        if (cv2.waitKey(1) & 0xFF) == ord('q'):  # Hit 'q' to exit
            break

    # Release everything if job is finished
    out.release()
    cv2.destroyAllWindows()  # Kills python process windows
    os.rmdir(dir_path)  # Remove temporary image directory

    # 224x256x3
    print("H: ", height, "W: ", width, "C: ", channels)
    print("Number of Rendered Frames: ", frame_count)

    env.close()

# %% Plot Results

#    # Plot Saliency Map
#    plt.figure()
#    plt.imshow(testIMG.salience_map)
#    plt.plot()
#
#    # Plot max intensity regions, and plot bounding box on original image
##    fig, axes = plt.subplots(1, 5, sharey=True)
#    bT = 1  # Bounding Box Thickness
#    for i in range(len(testIMG.MIC)):
#
#        # Grab bounding coordinates
#        a = testIMG.MIC[i][0]
#        b = testIMG.MIC[i][1]
#        c = testIMG.MIC[i][2]
#        d = testIMG.MIC[i][3]
#
#        # Update original image
#        if (testIMG.rgb):
#            testIMG.original[a:b, c:c+bT] = [255, 150, 100]
#            testIMG.original[a:b, d:d+bT] = [255, 150, 100]
#            testIMG.original[a:a+bT, c:d+bT] = [255, 100, 100]
#            testIMG.original[b:b+bT, c:d+bT] = [255, 100, 100]
#        else:
#            testIMG.img[a:b, c:c+bT] = [255]
#            testIMG.img[a:b, d:d+bT] = [255]
#            testIMG.img[a:a+bT, c:d+bT] = [255]
#            testIMG.img[b:b+bT, c:d+bT] = [255]
#
#        # Generate intense region subplots
##        axes[i].imshow(testIMG.salience_map[a:b, c:d])
##        axes[i].set_title("X: {}, Y: {}".format(testIMG.MIC[i][1] - 25,
##                                                testIMG.MIC[i][3] - 25))
#    plt.show()
#
#    # Create a figure with 2 subplots
#    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
#    ax1.set_title('Original')
#    ax2.set_title('Saliency Map')
#    if (testIMG.rgb):
#        ax1.imshow(testIMG.original)
#    else:
#        ax1.imshow(testIMG.img)
#    ax2.imshow(testIMG.salience_map)
#    plt.show()