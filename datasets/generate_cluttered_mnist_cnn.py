# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 14:37:33 2019

@author: meser

generate_cluttered_mnist_cnn.py - Generate a cluttered MNIST dataset for a CNN
"""

import pickle
import os
import sys

from scipy.ndimage import gaussian_filter
from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision as tv
from tqdm import tqdm
from PIL import Image

np.random.seed(1234)


def select_digits(n_digits, data, labels, out_shape):
    n_samples = len(data)
    indices = np.random.choice(range(n_samples), replace=True, size=n_digits)
    digits = [data[i].reshape(out_shape).numpy() for i in indices]
    labels = [labels[i] for i in indices]
    return digits, labels


def plot_n_by_n_images(images, epoch=None, folder=None, n=10, shape=[28, 28]):
    """ Plot 100 MNIST images in a 10 by 10 table. Note that we crop
    the images so that they appear reasonably close together. The
    image is post-processed to give the appearance of being continued."""

    a, b = shape
    img_out = np.zeros((a * n, b * n))
    for x in range(n):
        for y in range(n):
            xa, xb = x * a, (x + 1) * b
            ya, yb = y * a, (y + 1) * b
            im = np.reshape(images[(x * n) + y], (a, b))
            img_out[xa:xb, ya:yb] = im

    img_out *= 255
    img_out = Image.fromarray(img_out.astype('uint32'))
    if folder is not None and epoch is not None:
        img_out.save(os.path.join(folder, epoch + ".png"))
    return img_out


def create_sample(digit, in_shape, out_shape, idx):
    """ Generate an {out_shape} sized sample of {digits} """

    # Dimensions of input
    input_x = in_shape[0]
    input_y = in_shape[1]

    # Dimensions of output
    output_x = out_shape[0]
    output_y = out_shape[1]

    # Initial offset
    x_offset = input_x // 2
    y_offset = input_y // 2

    # Randomized y offset between each digit
    angle = np.random.choice(range(int(-input_y * 1.5), int(input_y * 1.5)))

    output = np.zeros((output_x, output_y), dtype=np.uint8)
    i = np.random.choice(range(0, 2))

    # Starting x coordinate of the digit placement
    x_start = i * (x_offset + input_y)

    # Ending x coordinate of the digit placement
    x_end = int(x_start + input_y)

# Starting y coordinate of the digit placement
    y_start = int(y_offset + np.floor(i * angle))

    # Ending y coordinate of the digit placement
    y_end = int(y_start + input_x)

    # Boundary cases
    if (y_start < 0):
        m = y_start
        y_end -= m
        y_start -= m
    if (y_end > (out_shape[1] - 1)):
        m = out_shape[1] - y_end
        y_end += m
        y_start += m
    if (x_start < 0):
        m = x_start
        x_end -= m
        x_start -= m
    if (x_end > (out_shape[0] - 1)):
        m = out_shape[0] - x_end
        x_end += m
        x_start += m

    # Place digit in the output sample space
    output[y_start:y_end, x_start:x_end] = digit

    return output


def add_distortions(input, num_distortions, distortions):
    dist_size = distortions[0].size()
    if (num_distortions > 0):
        canvas = np.zeros_like(input)
        for i in range(num_distortions):
            rand_distortion = distortions[np.random.randint(len(distortions))]
            rand_x = np.random.randint(input.shape[0] - dist_size[0])
            rand_y = np.random.randint(input.shape[1] - dist_size[1])
            canvas[rand_y:rand_y + dist_size[1],
                   rand_x:rand_x + dist_size[0]] = rand_distortion
        canvas += input

    # return np.clip(canvas, 0, 1)
    return canvas


def generate_distortions(all_digits, num_distortions, num_digits, dist_size=(9, 9)):
    """ Generate X amount of distortions and save to a list """

    print("Create distortions")
    distortions = []
    for i in tqdm(range(num_distortions)):
        rand_digit = np.random.randint(num_digits)  # Random index
        rand_x = np.random.randint(all_digits[0].shape[0] - dist_size[0])  # Random x coord
        rand_y = np.random.randint(all_digits[0].shape[1] - dist_size[1])  # Random y coord

        # Select a random MNIST digit to grab pixels from
        digit = all_digits[rand_digit]

        # Genereate a distortion by pulling pixels from a random digit
        distortion = digit[rand_x:(rand_x + dist_size[0]), rand_y:(rand_y + dist_size[1])]

        assert distortion.shape == dist_size
        distortions.append(distortion)

    return distortions


def generate_dataset(num_samples, data, labels, in_shape, out_shape, digit_count=3):
    """ Create the Cluttered MNIST Data set with the specified parameters """

    # Declare data and label arrays
    # dataset = np.zeros((num_samples, out_shape[0], out_shape[1]))
    dataset = []
    # labelset = np.zeros((num_samples, digit_count), dtype=int)
    labelset = []

    # Generate distortion data
    distortion_count = 100000
    num_distortions = 6
    distortions = generate_distortions(data, distortion_count,
                                       num_samples, dist_size=(6, 6))

    # Create {num_samples} of samples
    print("Create samples")
    for i in tqdm(range(num_samples)):

        # Pull digits and labels from MNIST
        # _digits, _labels = select_digits(digit_count, data, labels, in_shape)
        _digits = data[i].numpy()
        _labels = labels[i].item()

        # Create a canvas sample combining the digits
        output = create_sample(_digits, in_shape, out_shape, i)

        # Labels array
        labelset.append(_labels)

        # Add distortions to canvas
        dataset.append(add_distortions(output, num_distortions, distortions))

    return (dataset, labelset)


if __name__ == "__main__":

    # Load MNIST Input data
    tv.datasets.MNIST("./", train=True, download=True)
    mnist_data = torch.load("./MNIST/processed/training.pt")
    mnist_train_data = mnist_data[0].data
    mnist_train_labels = mnist_data[1].data
    mnist_data = torch.load("./MNIST/processed/test.pt")
    mnist_test_data = mnist_data[0].data
    mnist_test_labels = mnist_data[1].data

    # Parameters
    num_samples = 1000
    all_digits = mnist_train_data
    in_shape = [mnist_train_data.shape[1], mnist_train_data.shape[2]]  # 28x28
    out_shape = [60, 60]
    distortion_count = 10000
    distortions = generate_distortions(all_digits, distortion_count, num_samples)

    # Generate example canvas
    # samples = []
    # for i in range(num_samples):
        # _digits, _labels = select_digits(3, mnist_train_data, mnist_train_labels, in_shape)
    #     output = create_sample(_digits, in_shape, out_shape, i)
    #     samples.append(add_distortions(output, 6, distortions))
    # samples_arr = np.vstack(samples).reshape(num_samples, 60, 60)
    # out = plot_n_by_n_images(samples_arr, epoch="cnn", folder="ClutteredMNIST",
    #                          n=10, shape=out_shape)

    # Generate training data set and save to pickle file
    num_samples = len(mnist_train_data)
    td, tl = generate_dataset(num_samples, mnist_train_data, mnist_train_labels,
                               in_shape, out_shape, digit_count=1)

    train_data = np.array(td)
    train_labels = np.array(tl)

    # Display a data set sample
    print(train_labels[0])
    plt.figure()
    plt.imshow(train_data[0])
    plt.figure()
    plt.imshow(train_data[1])
    plt.show()

    # Dump to pickle file
    filename = "./ClutteredMNIST/train_cnn.pickle"
    outfile = open(filename, "wb")
    pickle.dump([train_data, train_labels], outfile)
    outfile.close()

    # Generate testing data set and save to pickle file
    num_samples = len(mnist_test_data)
    all_digits = mnist_test_data
    td, tl = generate_dataset(num_samples, mnist_test_data, mnist_test_labels,
                              in_shape, out_shape, digit_count=1)

    test_data = np.array(td)
    test_labels = np.array(tl)

    # Dump to pickle file
    filename = "./ClutteredMNIST/test_cnn.pickle"
    outfile = open(filename, "wb")
    pickle.dump([test_data, test_labels], outfile)
    outfile.close()
