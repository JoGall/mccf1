# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


def mcc_fun(true_pos, false_pos, true_neg, false_neg):
    if (true_pos + false_pos) * (true_pos + false_neg) * (true_neg + false_pos) * (true_neg + false_neg) == 0:
        return 0
    else:
        return ((true_pos * true_neg) - (false_pos * false_neg)) / np.sqrt(np.float(
            (true_pos + false_pos) * (true_pos + false_neg) * (true_neg + false_pos) * (true_neg + false_neg)
        ))

def precision_fun(true_pos, false_pos):
    if (true_pos + false_pos) == 0:
        return 1
    else:
        return true_pos / (true_pos + false_pos)

def recall_fun(true_pos, false_neg):
    if (true_pos + false_neg) == 0:
        return 1
    else:
        return true_pos / (true_pos + false_neg)

def f1_score_fun(precision, recall):
    if (precision + recall) == 0:
        return 0
    else:
        return 2 * (precision * recall) / (precision + recall)

def normalise_mcc(mcc, f1_score):
    # normalise MCC to [0,1]
    mcc = (mcc + 1) / 2

    # MCC is NaN where F1 score is NaN
    f1_nan_idx = np.where(np.isnan(f1_score))[0]
    mcc[f1_nan_idx] = np.nan

    return mcc

def quantise_predictions(predicted, threshold):
    return np.where(predicted > threshold, 1, 0)

def confusion_matrix_fun(actual, predicted):
    true_pos = len(predicted[(predicted == 1) & (actual == 1)])
    false_pos = len(predicted[(predicted == 1) & (actual == 0)])
    true_neg = len(predicted[(predicted == 0) & (actual == 0)])
    false_neg = len(predicted[(predicted == 0) & (actual == 1)])

    return true_pos, false_pos, true_neg, false_neg

def bisect_mccf1_curve(mcc, f1_score):
    # get index of the point with largest normalized MCC ("point" refers to the point on the MCC-F1 curve)
    index_of_max_mcc = np.nanargmax(mcc)

    # define points on the MCC-F1 curve located on the left of the point with the highest normalized MCC as "left curve"
    mcc_left = mcc[0:index_of_max_mcc]
    f1_left = f1_score[0:index_of_max_mcc]

    # define points on the MCC-F1 curve located on the right of the point with the highest normalized MCC as "right curve"
    mcc_right = mcc[(index_of_max_mcc):len(mcc)]
    f1_right = f1_score[(index_of_max_mcc):len(f1_score)]

    return mcc_left, mcc_right, f1_left, f1_right

def get_mccf1_metrics(actual, predicted, n_bins=100):
    # define thresholds
    thresholds = np.linspace(0.001, 0.999, num=n_bins, endpoint=True)

    # init global lists
    mcc = list()
    precision = list()
    recall = list()
    f1_score = list()

    # for each threshold...
    for i, tau in enumerate(thresholds):
        # binarise probabilistic predictions
        predicted_quantised = quantise_predictions(predicted, thresholds[i])
        true_pos, false_pos, true_neg, false_neg = confusion_matrix_fun(actual, predicted_quantised)

        # matthews correlation coefficients
        mcc_i = mcc_fun(true_pos=true_pos, false_pos=false_pos, true_neg=true_neg, false_neg=false_neg)

        # precision score
        precision_i = precision_fun(true_pos=true_pos, false_pos=false_pos)

        # recall score
        recall_i = recall_fun(true_pos=true_pos, false_neg=false_neg)

        # f1 score
        f1_score_i = f1_score_fun(precision_i, recall_i)

        # append to global lists
        mcc.append(mcc_i)
        precision.append(precision_i)
        recall.append(recall_i)
        f1_score.append(f1_score_i)

    # lists to array
    mcc = np.array(mcc)
    precision = np.array(precision)
    recall = np.array(recall)
    f1_score = np.array(f1_score)

    # normalise MCC from [-1, 1] to [0, 1]; infer NaN where F1 score is NaN
    mcc = normalise_mcc(mcc, f1_score)

    return precision, recall, mcc, f1_score, thresholds

def calc_mccf1(actual, predicted, n_bins=100):
    # get validation metrics
    precision, recall, mcc, f1_score, thresholds = get_mccf1_metrics(actual, predicted, n_bins=n_bins)

    # bisect MCC-F1 curve into left, right sides by which side of point with the highest normalized MCC each points lies
    mcc_left, mcc_right, f1_left, f1_right = bisect_mccf1_curve(mcc, f1_score)

    # divide the range of normalized MCC into subranges
    unit_len = (max(mcc) - min(mcc)) / n_bins

    # calculate the sum of mean distances from the left curve to the point (1, 1)
    mean_distances_left = []
    for i in range(n_bins):
        # find all the points on the left curve with normalized MCC between unit_len*(i-1) and unit_len*i
        idx = (mcc_left >= min(mcc) + (i - 1) * unit_len) & (mcc_left <= min(mcc) + i * unit_len)

        sum_of_distance_within_subrange = 0
        for pos, mask in enumerate(idx):
            d = np.sqrt((mcc_left[pos] - 1) ** 2 + (f1_left[pos] - 1) ** 2) * mask
            d = np.where(np.isnan(d), 0, d)
            sum_of_distance_within_subrange = sum_of_distance_within_subrange + d

        mean_distances_left.append(sum_of_distance_within_subrange / np.sum(idx))

    def distance_of_bisected_curve_from_optimal(mcc_lr, f1_lr, unit_len, min_mcc, n_bins=100):
        mean_distances_lr = []
        for i in range(n_bins):
            # find all the points on the curve section with normalized MCC between unit_len*(i-1) and unit_len*i
            idx = (mcc_lr >= min_mcc+ (i - 1) * unit_len) & (mcc_lr <= min_mcc + i * unit_len)

            sum_of_distance_within_subrange = 0
            for pos, mask in enumerate(idx):
                d = np.sqrt((mcc_lr[pos] - 1) ** 2 + (f1_right[pos] - 1) ** 2) * mask
                d = np.where(np.isnan(d), 0, d)
                sum_of_distance_within_subrange = sum_of_distance_within_subrange + d

            mean_distances_lr.append(sum_of_distance_within_subrange / np.sum(idx))

        return mean_distances_lr

    distance_of_bisected_curve_from_optimal(mcc_right, f1_right, min(mcc), n_bins=n_bins)

    # calculate the sum of mean distances from the right curve to the point (1, 1)
    mean_distances_right = []
    for i in range(n_bins):
        # find all the points on the right curve with normalized MCC between unit_len*(i-1) and unit_len*i
        idx = (mcc_right >= min(mcc) + (i - 1) * unit_len) & (mcc_right <= min(mcc) + i * unit_len)

        sum_of_distance_within_subrange = 0
        for pos, mask in enumerate(idx):
            d = np.sqrt((mcc_right[pos] - 1) ** 2 + (f1_right[pos] - 1) ** 2) * mask
            d = np.where(np.isnan(d), 0, d)
            sum_of_distance_within_subrange = sum_of_distance_within_subrange + d

        mean_distances_right.append(sum_of_distance_within_subrange / np.sum(idx))

    # drop NAs and sum the mean distances
    num_of_na_left = sum(np.isnan(mean_distances_left))
    sum_of_mean_distances_left_no_na = np.nansum(mean_distances_left)
    num_of_na_right = sum(np.isnan(mean_distances_right))
    sum_of_mean_distances_right_no_na = np.nansum(mean_distances_right)

    # calculate the MCC-F1 metric
    mccf1_metric = 1 - ((sum_of_mean_distances_left_no_na + sum_of_mean_distances_right_no_na) / (
                n_bins * 2 - num_of_na_left - num_of_na_right)) / np.sqrt(2)

    # find the best threshold
    eu_distance = []
    for i in range(len(mcc)):
        eu_distance.append(np.sqrt((1 - mcc[i]) ** 2 + (1 - f1_score[i]) ** 2))

    best_threshold_idx = np.nanargmin(eu_distance)
    best_threshold = thresholds[best_threshold_idx]

    return mccf1_metric, best_threshold

def plot_mccf1(actual, predicted, n_bins=100):
    # get values for plotting
    precision, recall, mcc, f1_score, thresholds = get_mccf1_metrics(actual, predicted, n_bins=n_bins)

    # get summary statistics
    mccf1_score, best_threshold = calc_mccf1(actual, predicted, n_bins=n_bins)

    # plot
    fig, ax = plt.subplots()
    ax.plot(f1_score, mcc, c='r')
    ax.plot(0, 0, 'o', c='k')
    ax.plot(1, 1, 'o', c='k')
    plt.hlines(0.5, 0, 1, ls='--', alpha=0.5)
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('F1 score')
    ax.set_ylabel('MCC (unit-normalised)')
    ax.set_title(f'MCC-F1 score = {mccf1_score:.3f}; τ = {best_threshold:.3f}')


if __name__ == '__main__':
    import pandas as pd

    xp_df = pd.read_csv("runtime/xp_39_2020.csv")

    actual = np.array(xp_df.outcome)
    predicted = np.array(xp_df.xP)

    calc_mccf1(actual, predicted, n_bins = 100)
    # plot_mccf1(actual, predicted, n_bins=100)
