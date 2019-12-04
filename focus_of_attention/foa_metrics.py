# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 14:42:28 2019

@author: meser

foa_metrics.py - Implement metrics to quantify saliency.

MIT Saliency MatLab Reference Code:
https://github.com/cvzoya/saliency/tree/master/code_forMetrics

Current metrics:
    Precision-Recall (PR) and Average Precision (AP)
    F-Measure (F1)
    ROC Curve + sAUC (Shuffled) (TODO)
    ROC Curve + AUC (Borji)
    ROC Curve + AUC (Judd)
    ROC Curve + AUC (sklearn)
    Pearson's Correlation Coefficient (CC)
    Normalized Scanpath Saliency (NSS)
    Mean Absolute Error (MAE)
    Kullback-Liebler Divergence (KL) (Fix?)
    Similarity (SIM)
    Earth Mover Distance (EMD) (TODO)
    Information Gain (IG) (TODO)

"""

# Standard Library Imports
import sys
import time

# 3P Imports
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
from tqdm import tqdm

# Local Imports
import foa_image as foai
import foa_convolution as foac
import foa_saliency as foas

plt.rcParams.update(plt.rcParamsDefault)


def squarePadImage(image):

    # Find differential between the image dimensions
    dim_diff = abs(np.array(image.shape) - max(image.shape))

    # Determine dimension to pad
    index = 0
    if (image.shape[0] > image.shape[1]):
        index = 1
    else:
        index = 0

    # Calculate amount to pad
    nmap = [(0, 0) for x in range(len(image.shape))]
    nmap[index] = (int(dim_diff[0] / 2), int(dim_diff[0] / 2))

    # Pad
    image = np.pad(image, pad_width=nmap, mode='constant')

    return image


def normalize(map=np.array):
    """ Min-max Normalization """
    assert (map is not None)

    # If max is not 1, normalize the map
    max_v = map.max()
    if (max_v != 1):
        map = (map - map.min()) / (map.max() - map.min())

    return map


def binarization(continuous_map, tf=0.5):
    assert (continuous_map is not None)

    # If max is not 1, average the map
    # max_v = continuous_map.max()
    # if (max_v != 1):
    #     continuous_map *= 1 / max_v

    continuous_map = normalize(continuous_map)

    # Binarization
    binary_map = np.where(continuous_map > tf, 1.0, 0.0)

    return binary_map


def f_measure(precision, recall, beta_square=0.3):
    f_score = ((1 + beta_square) * precision * recall) / (beta_square * precision + recall)
    return f_score.max()


def foa_precision_recall_curve(image, tf=0.5, mode="micro", plot=False):
    assert (image.salience_map is not None and image.ground_truth is not None)
    assert (image.salience_map.max() == 1.0)
    smap = image.salience_map
    gt = image.ground_truth

    # Generate binary ground truth
    binary_gt = binarization(gt, tf)

    # Calculate Precision-Recall Curve and Average Precision
    precision, recall, _ = precision_recall_curve(binary_gt.flatten(), smap.flatten())
    ap_score = average_precision_score(binary_gt.flatten(), smap.flatten(), average=mode)
    gt = normalize(gt)
    f_score = f_measure(precision, recall)

    if (plot):
        plt.figure()
        plt.plot(recall, precision, color='blue', lw=2,
                 label=f"PR Curve (AP = {ap_score:.3f}); (mode = {mode})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title(f"Precision-Recall Curve for {image.name}")
        plt.legend(loc="upper right")
        plt.show()

    return ap_score, f_score


def auc_sklearn(smap, gt):
    fpr, tpr, thresholds = roc_curve(gt.flatten(), smap.flatten())
    score = auc(fpr, tpr)

    return fpr, tpr, thresholds, score


def auc_judd(smap, gt, jitter=False, plot=False):
    """ Measures how well the saliency map of an image predicts the ground
    truth human fixations on the image.

    This version of the Area Under ROC curve measure has been called AUC-Judd
    in Riche et al. 2013. The saliency map is treated as a binary classifier to
    separate positive from negative samples at various thresholds. The true
    positive (tp) rate is the proportion of saliency map values above threshold
    at fixation locations. The false positive (fp) rate is the proportion of
    saliency map values above threshold at non-fixated pixels. In this
    implementation, the thresholds are sampled from saliency map values.

    smap - saliency map
    gt - ground truth fixation map (binary)
    jitter - will add tiny non-zero random constant to all map locations
    plot - ROC Curve plot boolean flag """

    start = time.time()
    assert (smap is not None and gt is not None)
    assert (smap.max() == 1.0)
    assert (smap.shape == gt.shape)  # Saliency map and ground truth should be the same shape
    if (jitter):
        # Jitter the saliency map slightly to disrupt ties of the same numbers
        smap += np.random.rand(smap.shape) / 1e7

    # Flatten maps to 1D arrays
    smap_flat = smap.flatten()
    gt_flat = gt.flatten()

    # Get Saliency Map values at the fixation locations
    thresholded_smap = smap_flat[gt_flat > 0]
    num_fixations = len(thresholded_smap)
    num_pixels = len(smap_flat)

    # Sort thresholds in reverse order
    all_thresholds = sorted(thresholded_smap, reverse=True)
    tpr = np.zeros((num_fixations + 2, 1))  # Initialize true positive rate vector
    fpr = np.zeros((num_fixations + 2, 1))  # Initialize false positive rate vector
    tpr[0] = 0
    tpr[-1] = 1
    fpr[0] = 0
    fpr[-1] = 1

    print("Calculating AUC Judd...")
    for i in tqdm(range(num_fixations)):

        # Number of saliency map values above threshold
        above_threshold = len(smap_flat[smap_flat > all_thresholds[i]])

        # Calculate rates
        tpr[i + 1] = (i + 1) / num_fixations  # ratio sal map values at fixation locations above threshold
        fpr[i + 1] = (above_threshold - i) / (num_pixels - num_fixations)  # ratio other sal map values above threshold

    score = np.trapz(tpr, x=fpr, axis=0)[0]
    all_thresholds = np.concatenate(([1], all_thresholds, [0]))
    stop = time.time()
    print(f"AUC Judd Score: {score}; Time: {stop - start}")

    return fpr, tpr, all_thresholds, score


def auc_borji(image, num_splits=10, step_size=0.01):

    assert (image.salience_map is not None and image.ground_truth is not None)
    assert (image.salience_map.max() == 1.0)
    smap = image.salience_map
    gt = binarization(image.ground_truth)

    # Flatten maps to 1D arrays
    smap_flat = smap.flatten()
    gt_flat = gt.flatten()

    # Get Saliency Map values at the fixation locations
    thresholded_smap = smap_flat[gt_flat > 0]
    num_fixations = len(thresholded_smap)
    num_pixels = len(smap_flat)

    # for each fixation, sample Nsplits values from anywhere on the sal map
    r = np.random.randint(1, high=num_pixels, size=[num_fixations, num_splits])
    rand_fix = smap_flat[r]  # sal map values at random locations

    # calculate AUC per random split (set of random locations)
    print("Calculating AUC Borji...")
    fpr_total = np.empty(())
    tpr_total = np.empty(())
    auc = np.zeros((num_splits, 1))
    for s in tqdm(range(num_splits)):
        cur_fix = rand_fix[:, s]
        thresh_end = np.float64(np.concatenate((thresholded_smap, cur_fix)).max())
        all_thresholds = np.flip(np.arange(0.0, thresh_end, step_size))
        tpr = np.zeros((len(all_thresholds), 1))
        fpr = np.zeros((len(all_thresholds), 1))

        for i in range(len(all_thresholds)):
            thresh = all_thresholds[i]
            fpr[i] = (cur_fix >= thresh).sum() / num_fixations
            tpr[i] = (thresholded_smap >= thresh).sum() / num_fixations

        fpr_total = np.append(fpr_total, fpr)
        tpr_total = np.append(tpr_total, tpr)
        auc[s] = np.trapz(tpr, x=fpr, axis=0)[0]
    score = auc.mean()  # mean across random splits

    return fpr_total, tpr_total, all_thresholds, score


def foa_roc_curve_and_auc(image, tf=0.5, mode="sklean", plot=False):
    assert (image.salience_map is not None and image.ground_truth is not None)
    assert (image.salience_map.max() == 1.0)
    smap = image.salience_map
    gt = normalize(image.ground_truth)

    # Generate binary ground truth
    binary_gt = binarization(gt, tf)

    # Calculate ROC Curve and AUC
    if (mode == "sklearn"):
        fpr, tpr, thresholds, foa_roc_auc = auc_sklearn(smap, binary_gt)
    elif (mode == "judd"):
        fpr, tpr, thresholds, foa_roc_auc = auc_judd(smap, binary_gt)
    elif (mode == "borji"):
        fpr, tpr, thresholds, foa_roc_auc = auc_borji(image)
    elif (mode == "shuffled"):
        raise NotImplementedError("AUC Shuffled not implemented")
    else:
        raise ValueError(f"Invalid Mode: {mode}")

    if (plot):
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f"ROC Curve ({mode} area = {foa_roc_auc:.3f})")
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve for {image.name}")
        plt.legend(loc="lower right")
        plt.show()

    return foa_roc_auc


def __correlation_coefficient(image_x, image_y):
    """ Calculate Pearson's Correlation Coefficient to quantify similarity """

    assert len(image_x.shape) == 2, "Image X should be two dimensions!"
    assert len(image_y.shape) == 2, "Image Y should be two dimensions!"

    return np.corrcoef(image_x.flat, image_y.flat)[0, 1]


def foa_correlation_coefficient(image):
    """ Focus of Attention specific Correlation Coefficient """

    assert (image.salience_map is not None and image.ground_truth is not None)
    smap = image.salience_map
    gt = normalize(image.ground_truth)
    return __correlation_coefficient(smap, gt)


def foa_normalized_scanpath_saliency(image, tf=0.5):
    """ Normalized Scanpath Saliency
    This is the normalized scanpath saliency between two different saliency
    maps. It is measured as the mean value of the normalized saliency map at
    fixation locations. """

    assert (image.salience_map is not None and image.ground_truth is not None)
    assert (image.salience_map.max() == 1.0)
    smap = image.salience_map.astype(float)
    gt = image.ground_truth.astype(float)

    # Generate binary ground truth
    binary_gt = binarization(gt, tf)

    # Normalized saliency map
    if (smap.std(ddof=1) != 0):
        smap_normalize = (smap - smap.mean()) / smap.std(ddof=1)

    # Mean value at fixation locations
    score = smap_normalize[binary_gt > 0].mean()

    return score


def foa_mean_absolute_error(image):
    """ Mean Absolute Error
    Finds the L1 Norm of the saliency map and the fixation map; Direct pixel
    based comparison """

    assert (image.salience_map is not None and image.ground_truth is not None)
    assert (image.salience_map.max() == 1.0)
    smap = image.salience_map.astype(float)
    gt = image.ground_truth.astype(float)

    # Min-max normalize fixation map
    gt = normalize(gt)

    # Number of pixels
    num_pixels = smap.size

    # Calculate mean element-wise absolute error
    error = np.abs(np.subtract(smap, gt))
    total_error = error.sum()
    mae = total_error / num_pixels

    return mae


def foa_kl_divergence(image, epsilon=1e-7):
    """ Kullback-Liebler Divergence - the divergence between two different
    saliency maps when viewed as distributions: it is a non-symmetric measure
    of the information lost when the saliency map is used to estimate the
    fixation map.

    *** Image-based KL Divergence, not fixation-based ***

    Very sensitive to order of magnitude changes in epsilon. """

    assert (image.salience_map is not None and image.ground_truth is not None)
    assert (image.salience_map.max() == 1.0)
    smap = image.salience_map.astype(float)
    gt = image.ground_truth.astype(float)

    # make sure map1 and map2 sum to 1
    if (smap.any()):
        smap = smap / smap.sum()
    if (gt.any()):
        gt = gt / gt.sum()

    # Computer KL-divergence
    score = (gt * np.log(epsilon + (gt / (smap + epsilon)))).sum()

    return score


def foa_sim(image):
    """ Similarity (SIM) Metric
    This similarity measure is also called histogram intersection and measures
    the similarity between two different saliency maps when viewed as
    distributions (SIM=1 means the distributions are identical). """

    assert (image.salience_map is not None and image.ground_truth is not None)
    assert (image.salience_map.max() == 1.0)
    smap = image.salience_map.astype(float)
    gt = image.ground_truth.astype(float)

    # Average the saliency map to sum to 1
    smap = smap / smap.sum()

    # Min-max normalize fixation map to sum to 1
    gt = normalize(gt)

    # Compute Histogram Intersection
    diff = np.fmin(smap, gt)
    score = diff.sum()

    return score


def foa_earth_mover_distance(image):
    """ Earth Mover Distance (EMD) Metric
    The Earth Mover's Distance measures the distance between two probability
    distributions by how much transformation one distribution would need to
    undergo to match another (EMD=0 for identical distributions). """
    pass


def foa_information_gain(image, baseline):
    assert (image.salience_map is not None and image.ground_truth is not None)
    assert (image.salience_map.max() == 1.0)
    smap = image.salience_map.astype(float)
    gt = image.ground_truth.astype(float)

    # gt = discretize_gt(gt)
    gt /= 255
    # assuming s_map and baseline_map are normalized
    eps = 2.2204e-16
    smap = smap / np.sum(smap)
    baseline = baseline / np.sum(baseline)

    # for all places where gt=1, calculate info gain
    temp = []
    x, y = np.where(gt > 0)
    for i in zip(x, y):
        temp.append(np.log2(eps + smap[i[0], i[1]]) - np.log2(eps + baseline[i[0], i[1]]))

    return np.mean(temp)


# Main Routine
if __name__ == '__main__':
    plt.close('all')

    # Open test images as 8-bit RGB values - Time ~0.0778813
    file = "./SMW_Test_Image.png"
    mario = foai.ImageObject(file)
    file = "../datasets/AIM/eyetrackingdata/original_images/22.jpg"
    banana = foai.ImageObject(file)
    file = "../datasets/AIM/eyetrackingdata/ground_truth/d22.jpg"
    gt_banana = Image.open(file)
    gt_banana.load()
    file = "../datasets/AIM/eyetrackingdata/original_images/120.jpg"
    corner = foai.ImageObject(file)
    file = "../datasets/AIM/eyetrackingdata/ground_truth/d120.jpg"
    gt_corner = Image.open(file)
    gt_corner.load()

    # Test Image
    test_image = corner
    test_image.ground_truth = np.array(gt_corner, dtype=np.float64)

# %% Generate Saliency Map

    # Generate Gaussian Blur Prior - Time ~0.0020006
    foveation_prior = foac.matlab_style_gauss2D(test_image.modified.shape, 300)

    # Generate Gamma Kernel
    # k = np.array([1, 26, 1, 25, 1, 19], dtype=float)
    # mu = np.array([2, 2, 1, 1, 0.5, 0.5], dtype=float)

    k = np.array([1, 20, 1, 30, 1, 40], dtype=float)
    mu = np.array([2, 2, 2, 2, 2, 2], dtype=float)
    kernel = foac.gamma_kernel(test_image, mask_size=(40, 40), k=k, mu=mu)

    # Generate Saliency Map
    start = time.time()
    foac.convolution(test_image, kernel, foveation_prior)
    stop = time.time()
    print(f"Salience Map Generation: {stop - start} seconds")

    # Bound and Rank the most Salient Regions of Saliency Map
    foas.salience_scan(test_image, rank_count=5, bbox_size=(80, 80))

# %% Plot Results
    test_image.plot_original_map()
    test_image.plot_modified_map
    test_image.plot_saliency_map()
    test_image.plot_ground_truth()
    # test_image.draw_image_patches(bbt=2)
    # for i in range(len(test_image.patched_sequence)):
    #     test_image.display_map(test_image.patched_sequence[i], f"{i}")
    # test_image.plot_patched_map()

# %% Evaluate Results

    # # Precision-Recall (PR) Test
    # ap_score, f_score = foa_precision_recall_curve(test_image, plot=True)

    # # Average Precision (AP) Score
    # print(f"Average Precision: {ap_score:.4f}")

    # # F-Measure (F1) Score
    # print(f"F-Measure: {f_score:.4f}")

    # # ROC + AUC (sklearn) Test
    # foa_auc = foa_roc_curve_and_auc(test_image, mode="sklearn", plot=False)
    # print(f"ROC sklearn AUC: {foa_auc:.4f}")

    # # ROC + AUC (Judd) Test
    # foa_auc = foa_roc_curve_and_auc(test_image, mode="judd", plot=False)
    # print(f"ROC Judd AUC: {foa_auc:.4f}")

    # # ROC + AUC (Borji) Test
    # foa_auc = foa_roc_curve_and_auc(test_image, mode="borji", plot=False)
    # print(f"ROC Borji AUC: {foa_auc:.4f}")

    # Correlation Coefficient (CC) Test
    cc = foa_correlation_coefficient(test_image)
    print(f"Correlation Coefficient: {cc:.4f}")

    # Normalized Scanpath Saliency (NSS) Test
    nss_score = foa_normalized_scanpath_saliency(test_image)
    print(f"NSS Score: {nss_score:.4f}")

    # Mean Absolute Error (MAE) Test
    mae = foa_mean_absolute_error(test_image)
    print(f"MAE: {mae:.4f}")

    # Kullback-Liebler Divergence (KB) Test
    kb_div = foa_kl_divergence(test_image)
    print(f"KL Divergence: {kb_div:.4f}")

    # Similarity (SIM) Test
    sim_score = foa_sim(test_image)
    print(f"Similarity: {sim_score:.4f}")

    # Information Gain (IG) Gaussian Test (Center Bias)
    ig_score = foa_information_gain(test_image, foveation_prior)
    print(f"Information Gain: {ig_score:.4f}")
