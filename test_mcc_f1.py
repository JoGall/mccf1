import numpy as np
from unittest import TestCase

from mcc_f1 import *


class TestMCCF1(TestCase):
    def test_quantise_predictions(self):
        predicted = np.array([0, 0.199, 0.201, 0.3])
        predicted_quantised = quantise_predictions(predicted, threshold = 0.2)

        np.testing.assert_array_equal(
            predicted_quantised,
            np.array([0, 0, 1, 1]))

    def test_confusion_matrix(self):
        actual = np.array([0, 1, 0, 1])
        predicted_quantised = np.array([0, 0, 1, 1])
        true_pos, false_pos, true_neg, false_neg = confusion_matrix_fun(actual, predicted_quantised)

        np.testing.assert_equal(true_pos, 1)
        np.testing.assert_equal(false_pos, 1)
        np.testing.assert_equal(true_neg, 1)
        np.testing.assert_equal(false_neg, 1)

    def test_mcc_fun(self):
        np.testing.assert_equal(
            mcc_fun(true_pos=0, false_pos=1, true_neg=1, false_neg=1),
            -0.5)
        np.testing.assert_equal(
            mcc_fun(true_pos=1, false_pos=1, true_neg=0, false_neg=0),
            0)
        np.testing.assert_equal(
            mcc_fun(true_pos=1, false_pos=0, true_neg=1, false_neg=0),
            1)
        np.testing.assert_almost_equal(
            mcc_fun(true_pos=5, false_pos=4, true_neg=5, false_neg=3),
            0.1805555)

    def test_precision_fun(self):
        np.testing.assert_equal(
            precision_fun(true_pos=3, false_pos=1),
            0.75)
        np.testing.assert_equal(
            precision_fun(true_pos=0, false_pos=0),
            1)
        np.testing.assert_equal(
            precision_fun(true_pos=100, false_pos=0),
            1)
        np.testing.assert_equal(
            precision_fun(true_pos=0, false_pos=100),
            0)

    def test_recall_fun(self):
        np.testing.assert_equal(
            recall_fun(true_pos=3, false_neg=1),
            0.75)
        np.testing.assert_equal(
            recall_fun(true_pos=0, false_neg=0),
            1)
        np.testing.assert_equal(
            recall_fun(true_pos=100, false_neg=0),
            1)
        np.testing.assert_equal(
            recall_fun(true_pos=0, false_neg=100),
            0)

    def test_f1_score_fun(self):
        np.testing.assert_equal(
            f1_score_fun(precision=1, recall=0),
            0)
        np.testing.assert_equal(
            f1_score_fun(precision=0, recall=1),
            0)
        np.testing.assert_equal(
            f1_score_fun(precision=0, recall=0),
            0)
        np.testing.assert_equal(
            f1_score_fun(precision=1, recall=1),
            1)

    def test_normalise(self):
        np.testing.assert_array_equal(
            normalise_mcc(mcc=np.array([-0.5, 0.5]), f1_score=np.array([0.5, 0.5])),
            np.array([0.25, 0.75]))

        np.testing.assert_array_equal(
            normalise_mcc(mcc=np.array([-0.5, 0.5]), f1_score=np.array([np.nan, 0.5])),
            np.array([np.nan, 0.75]))