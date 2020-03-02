#!/usr/bin/env python
# coding: utf-8

print('import packages')
import sys
sys.path.append('/data/hpcdata/users/racfur/DynamicPrediction/code_git/')
from Tools import CreateDataName as cn
from Tools import Model_Plotting as rfplt

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from skimage.util import view_as_windows
import os
import xarray as xr
import pickle
from netCDF4 import Dataset

#------------------------
# Set plotting variables
#------------------------
point = [5,5,5]

#----------------------
rootdir = '/data/hpcdata/users/racfur/DynamicPrediction/lr_Outputs/'

level = point[0]
y_coord = point[1]
x_coord = point[2]

#-----------------------------------------------
print('reading in ds')
data_filename=rootdir+'ITERATED_PREDICTION_ARRAYS/3dLatLonDepUVSalEtaPolyDeg2_AveragedSinglePredictions.nc'
ds = xr.open_dataset(data_filename)
da_AvTempError=ds['Averaged PredTemp-Truth'][:,:,:]


z_size = da_AvTempError.shape[0]
y_size = da_AvTempError.shape[1]
x_size = da_AvTempError.shape[2]

print('da_AvTempError.shape')
print(da_AvTempError.shape)

#--------------------------
# Plot spatial depth plots
#--------------------------
fig, ax, im = rfplt.plot_depth_fld(da_AvTempError[:,:,:], 'Temperature from Model', level, title=None, min_value=None, max_value=None, diff=True)
plt.savefig(rootdir+'PLOTS/AvTempErrors_z'+str(level), bbox_inches = 'tight', pad_inches = 0.1)

#-----------------------
# Plot y-cross sections
#-----------------------
fig, ax, im = rfplt.plot_yconst_crss_sec(da_AvTempError[:,:,:], 'Temperature Predictions', y_coord, title=None, min_value=None, max_value=None, diff=True)
plt.savefig(rootdir+'PLOTS/AvTempErrors_y'+str(y_coord), bbox_inches = 'tight', pad_inches = 0.1)

#-----------------------
# Plot x-cross sections
#-----------------------
fig, ax, im = rfplt.plot_xconst_crss_sec(da_AvTempError[:,:,:], 'Temperature Predictions', x_coord, title=None, min_value=None, max_value=None, diff=True)
plt.savefig(rootdir+'PLOTS/AvTempErrors_x'+str(x_coord), bbox_inches = 'tight', pad_inches = 0.1)

