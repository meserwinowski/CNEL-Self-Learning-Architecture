# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 15:13:40 2019

@author: meser

test_cluttered_mnist.py - Script intended to run tests on the Cluttered MNIST
set from generate_cluttered_mnist.py. View focus of attention results and
apply metrics

"""

# Standard Imports
import pickle
import sys
import time

# 3P Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Local Imports
sys.path.insert(1, "../focus_of_attention")
import foa_image as foai
import foa_convolution as foac
import foa_saliency as foas
import foa_metrics as foam


def run_tests(image):

    # Normalized Scanpath Saliency (NSS) Test
    nss_score = foam.foa_normalized_scanpath_saliency(image)
    print(f"NSS Score: {nss_score:.4f}")

    # Similarity (SIM) Test
    sim_score = foam.foa_sim(image)
    print(f"Similarity: {sim_score:.4f}")

    # Information Gain (IG) Test
    info_gain = None

    # Kullback-Liebler Divergence (KB) Test
    kl_div = foam.foa_kl_divergence(image)
    print(f"KL Divergence: {kl_div:.4f}")

    # Correlation Coefficient (CC) Test
    cc = foam.foa_correlation_coefficient(image)
    print(f"Correlation Coefficient: {cc:.4f}")

    # ROC + AUC (sklearn) Test
    foa_auc_sk = foam.foa_roc_curve_and_auc(image, mode="sklearn", plot=False)
    print(f"ROC sklearn AUC: {foa_auc_sk:.4f}")

    # ROC + AUC (Judd) Test
    foa_auc_judd = foam.foa_roc_curve_and_auc(image, mode="judd", plot=False)
    print(f"ROC Judd AUC: {foa_auc_judd:.4f}")

    # ROC + AUC (Borji) Test
    foa_auc_borji = foam.foa_roc_curve_and_auc(image, mode="borji", plot=False)
    print(f"ROC Borji AUC: {foa_auc_borji:.4f}")

    # Mean Absolute Error (MAE) Test
    mae = foam.foa_mean_absolute_error(image)
    print(f"MAE: {mae:.4f}")

    return [nss_score, sim_score, info_gain, kl_div, cc,
            foa_auc_sk, foa_auc_judd, foa_auc_borji, mae]


def run_foa(sample, gt):

    # Import image
    test_image = foai.ImageObject(sample, rgb=False)
    test_image.ground_truth = gt

    # Generate Gaussian Blur Prior - Time ~0.0020006
    foveation_prior = foac.matlab_style_gauss2D(test_image.modified.shape, 300)

    # Generate Gamma Kernel
    k = np.array([1, 9], dtype=float)
    mu = np.array([0.2, 0.5], dtype=float)
    kernel = foac.gamma_kernel(mask_size=(14, 14), k=k, mu=mu)
    # plt.imshow(kernel)

    # Generate Saliency Map
    foac.convolution(test_image, kernel, foveation_prior)

    # Bound and Rank the most Salient Regions of Saliency Map
    foas.salience_scan(test_image, rank_count=3, bbox_size=(28, 28))

    test_image.draw_image_patches()
    test_image.plot_main()
    test_image.plot_patches()
    plt.show()

    return test_image


if __name__ == "__main__":

    # Import data
    filename = "./ClutteredMNIST/train.pickle"
    with open(filename, 'rb') as infile:
        data = pickle.load(infile)

    train_data = data[0][0]
    cropped_digits = data[0][1]
    train_labels = data[1]
    ground_truth = data[2]

    # Pandas Dataframe
    metrics = ["NSS", "SIM", "IG", "KL", "CC",
               "AUC-SK", "AUC-JUDD", "AUC-BORJI", "MAE"]
    df = pd.DataFrame(columns=metrics)

    print("Process samples")
    count = 0
    for i, sample in enumerate(train_data):
        if (count == 3):
            break
        count += 1

        # Process sample
        start = time.time()
        foa_object = run_foa(sample, ground_truth[i])
        stop = time.time()
        print(f"Saliency Map Generation: {stop - start} seconds")

        # Run tests
        results = run_tests(foa_object)
        df.loc[len(df)] = results

    df.to_csv("./cluttered_mnist_res.csv")
