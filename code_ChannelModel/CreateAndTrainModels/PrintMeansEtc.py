import sys
sys.path.append('../Tools')
import ReadRoutines as rr

import numpy as np
import os
import glob
import xarray as xr
import pickle

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from torch.utils import data
import torch.nn as nn
from torch.autograd import Variable
import torch
from torchvision import transforms, utils

import netCDF4 as nc4

import time as time
import gc as gc

import logging

mean_std_file = '../../../Channel_nn_Outputs/IncLand_PredictNext_2d_MeanStd.npz'
mean_std_data = np.load(mean_std_file)
inputs_mean  = mean_std_data['arr_0']
inputs_std   = mean_std_data['arr_1']
inputs_range = mean_std_data['arr_2']
#targets_mean  = mean_std_data['arr_3']
#targets_std   = mean_std_data['arr_4']
#targets_range = mean_std_data['arr_5']

print(inputs_mean.shape)
print(inputs_mean)


mean_std_file = '../../../Channel_nn_Outputs/IncLand_2d_MeanStd.npz'
mean_std_data = np.load(mean_std_file)
inputs_mean  = mean_std_data['arr_0']
inputs_std   = mean_std_data['arr_1']
inputs_range = mean_std_data['arr_2']
targets_mean  = mean_std_data['arr_3']
targets_std   = mean_std_data['arr_4']
targets_range = mean_std_data['arr_5']

print(inputs_mean.shape)
print(inputs_mean)
