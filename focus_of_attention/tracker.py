# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 10:30:38 2019

@author: 61995
"""

#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>

import cv2
import sys
import os
import re

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


if __name__ == '__main__':

    # Set up tracker
    tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN',
                     'MOSSE', 'CSRT']
    tracker_type = tracker_types[7]

    if tracker_type == 'BOOSTING':
        tracker = cv2.TrackerBoosting_create()
    if tracker_type == 'MIL':
        tracker = cv2.TrackerMIL_create()
    if tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create()
    if tracker_type == 'TLD':
        tracker = cv2.TrackerTLD_create()
    if tracker_type == 'MEDIANFLOW':
        tracker = cv2.TrackerMedianFlow_create()
    if tracker_type == 'GOTURN':
        tracker = cv2.TrackerGOTURN_create()
    if tracker_type == 'MOSSE':
        tracker = cv2.TrackerMOSSE_create()
    if tracker_type == "CSRT":
        tracker = cv2.TrackerCSRT_create()

    # Read video
    file = "./SuperMarioWorld-Snes-YoshiIsland1-0001.mp4"
    video = cv2.VideoCapture(file)
    file = file[:-4].split('/')[-1]  # Get file name
    dir_path = os.getcwd() + '/Super Mario Gym/' + file + '/'

    # Exit if video not opened
    if not video.isOpened():
        print("Could not open video")
        sys.exit()

    # Read first frame
    ok, frame = video.read()
    height, width, channels = frame.shape
    if not ok:
        print('Cannot read video file')
        sys.exit()

    frame = cv2.resize(frame, (1024, 1024))

    # Define an initial bounding box
#    bbox = (1035,543,100,100)
    #bbox = (544,483,100,120)
    #bbox = (904,302,970-904,365-302)

    # Uncomment the line below to select a different bounding box
    bbox = cv2.selectROI(frame, False)

    # Initialize tracker with first frame and bounding box
    ok = tracker.init(frame, bbox)
    success = True
    count = 0

    while success:

        # Read a new frame
        success, frame = video.read()
        if not success:
            break

        frame = cv2.resize(frame, (1024, 1024))

        # Start timer
        timer = cv2.getTickCount()

        # Update tracker
        ok, bbox = tracker.update(frame)

        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        # Draw bounding box
        if ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))

            letter = frame[int(bbox[1]):int(bbox[1] + bbox[3]),
                           int(bbox[0]):int(bbox[0] + bbox[2])]
            #cv2.imwrite('3_'+str(i+417)+'.png', letter)

            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        else:
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # Display tracker type on frame
        cv2.putText(frame, tracker_type + " Tracker", (100, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        # Display result
        cv2.imshow("Tracking", frame)
        save_image(letter, dir_path + "patch/", str(count) + '.png')
        count += 1

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            cv2.destroyAllWindows()
            break

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output = file + "_patch" + ".mp4"
    out = cv2.VideoWriter(output, fourcc, 60.0, (width, height))
    output_video(out, dir_path + "patch/")
    out.release()
    delete_images(dir_path + "patch/")
    delete_directory(dir_path)
    video.release()
    cv2.destroyAllWindows()
