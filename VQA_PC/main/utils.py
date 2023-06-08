# -*- coding: utf-8 -*-

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import scipy
from scipy import stats
from scipy.optimize import curve_fit
from data_loader import VideoDataset_NR_image_with_fast_features
import ResNet_mean_with_fast
import random 
import time


def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
    denominator = np.abs(bayta4) + 1e-5  # to avoid division by zero
    numerator = np.negative(X - bayta3)
    exponent = np.clip(np.divide(numerator, denominator), -500, 500)  # to avoid overflow and underflow
    logisticPart = 1 + np.exp(exponent)
    yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
    return yhat


def fit_function(y_label, y_output):
    max_val = np.max(y_label)
    min_val = np.min(y_label)
    mean_val = np.mean(y_output)
    range_val = np.max(y_output) - np.min(y_output)
    beta = [max_val, min_val, mean_val, range_val]
    popt, _ = curve_fit(logistic_func, y_output, y_label, p0=beta, maxfev=1000000)
    y_output_logistic = logistic_func(y_output, *popt)
    return y_output_logistic
