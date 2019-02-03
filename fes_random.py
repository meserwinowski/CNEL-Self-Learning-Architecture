# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 15:20:38 2019

@author: meser

fes_random.py - Python script created for playing back .bk2 files and
converting a select number of frames to mp4. Integrated with gamma kernel
in the fes_gamma_video.py file

Resources/Credit:
    https://github.com/openai/retro
    https://medium.com/@tristansokol/day-6-of-the-openai-retro-contest-playback-tooling-3844ba655919
    http://tsaith.github.io/combine-images-into-a-video-with-python-3-and-opencv-3.html

"""

# !/usr/bin/python

import sys
import retro
import cv2
import os
import errno
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})


def render(file):
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
    framerate = 128
    frame_count = 0

    # Working Directory + Name of bk2 file
    dir_path = os.getcwd() + '/' + file[:-4]
    ext = '.png'  # Image extension
    output = file[:-4] + '.mp4'  # Video file extension
    images = []
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
        else:
            frame += 1

    env.close()

    # Get image dimensions
    image_path = os.path.join(dir_path, images[0])
    frame = cv2.imread(image_path)
    height, width, channels = frame.shape

    # Convert generated images into an mp4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output, fourcc, 20.0, (width, height))

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


''' Run the program from a command prompt (Anaconda) and use the bk2
    as the second argument after the file name (fes_random.py)
    Full command: python fes_random.py (file name + .bk2) '''
if __name__ == '__main__':
    print("sys.argv[:]: ", sys.argv[:])
    render(sys.argv[1])

''' Code to write out image data to a text file '''
#    # Create a file to write image data to
#    f = open(file[:-4] + "_Pixel_Data.txt", "w+")
#            # Write image data dimensions to first line
#            f.write(str(len(obs)) + '; ' +
#                    str(len(obs[0])) + '; ' +
#                    str(len(obs[0, 0])) + ';\n')
#
#            # Write image data
#            for col in obs:
#                for row in col:
#                    f.write(str(row) + ', ')
#
#            # Write delimiter between frames
#            f.write('\n-\n')
