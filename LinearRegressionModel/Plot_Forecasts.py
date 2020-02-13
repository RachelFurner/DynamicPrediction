#!/usr/bin/env python
# coding: utf-8

print('import packages')
import sys
sys.path.append('/data/hpcdata/users/racfur/DynamicPrediction/code_git/')
from Tools import CreateExpName as cn
from Tools import Model_Plotting as rfplt

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from skimage.util import view_as_windows
import os
import xarray as xr
import pickle

#-----------------------------------------------
# Read in netcdf file for 'truth' and get shape
#-----------------------------------------------
print('reading in ds')
DIR  = '/data/hpcdata/users/racfur/MITGCM_OUTPUT/20000yr_Windx1.00_mm_diag/'
data_filename=DIR+'cat_tave_5000yrs_SelectedVars.nc'
ds = xr.open_dataset(data_filename)
da_T=ds['Ttave'][:12001,:,:,:]

z_size = da_T.shape[1]
y_size = da_T.shape[2]
x_size = da_T.shape[3]

print('da_T.shape')
print(da_T.shape)

#----------------------
# Read in predictions
#----------------------

pred_dir = '/data/hpcdata/users/racfur/DynamicPrediction/RegressionOutputs/ITERATED_PREDICTION_ARRAYS/'

pred1_filename = pred_dir+ 'lr_3dLatDepUVSalEtaPolyDeg2_predictions.npy'
predictions1 = np.load(pred1_filename)

pred2_filename = pred_dir+ 'lr_2dLatDepUVSalEtaPolyDeg2_predictions.npy'
predictions2 = np.load(pred2_filename)

pred3_filename = pred_dir+ 'lr_3dDepUVSalEtaPolyDeg2_predictions.npy'
predictions3 = np.load(pred3_filename)

pred4_filename = pred_dir+ 'lr_3dLatUVSalEtaPolyDeg2_predictions.npy'
predictions4 = np.load(pred4_filename)

pred5_filename = pred_dir+ 'lr_3dLatDepSalEtaPolyDeg2_predictions.npy'
predictions5 = np.load(pred5_filename)

pred6_filename = pred_dir+ 'lr_3dLatDepUVEtaPolyDeg2_predictions.npy'
predictions6 = np.load(pred6_filename)

#predictions7 = np.zeros((for_len+1, z_size, y_size, x_size))
pred7_filename = pred_dir+ 'lr_3dLatDepUVSalPolyDeg2_predictions.npy'
predictions7 = np.load(pred7_filename)

#---------------------------
# Set persistence forecasts
#---------------------------
print('Set perstistence forecast')
persistence = np.zeros((12000, z_size, y_size, x_size))
persistence[:,:,:,:] = da_T[0,:,:,:]

#--------------------------
# Plot spatial depth plots
#--------------------------
#fig, ax, im = rfplt.plot_depth_fld(predictions1[1,:,:,:], 'Temperature Predictions', 5, title=None, min_value=None, max_value=None)
#plt.show()

#-------------------------------
# Plot spatial depth diff plots
#-------------------------------
#fig = rfplt.plot_depth_fld_diff(da_T[1,:,:,:], 'GCM Temperature', predictions1[1,:,:,:], 'Temperature Predictions', 5, title=None)
#plt.show()

#-----------------------
# Plot y-cross sections
#-----------------------
#fig, ax, im = rfplt.plot_y_crss_sec(predictions1[1,:,:,:], 'Temperature Predictions', 5, title=None, min_value=None, max_value=None)
#plt.show()

#---------------------------------
# Plot y-cross section diff plots
#---------------------------------
#fig = rfplt.plot_y_crss_sec_diff(da_T[1,:,:,:], 'GCM Temperature', predictions1[1,:,:,:],'Temperature Predictions', 5, title=None)
#plt.show()

#-----------------------
# Plot x-cross sections
#-----------------------
#fig, ax, im = rfplt.plot_x_crss_sec(predictions1[1,:,:,:], 'Temperature Predictions', 5, title=None, min_value=None, max_value=None)
#plt.show()

#---------------------------------
# Plot x-cross section diff plots
#---------------------------------
#fig = rfplt.plot_x_crss_sec_diff(da_T[1,:,:,:], 'GCM Temperature', predictions1[1,:,:,:],'Temperature Predictions', 5, title=None)
#plt.show()

#---------------------------------------------------------------------------
# Plot the predictions against da_T and persistence for a variety of points
#---------------------------------------------------------------------------

fig = rfplt.plt_timeseries([5,5,5], 120, {'da_T (truth)':da_T.data, 'persistence':persistence, 'full_lr_model':predictions1, '2d':predictions2, 'no_Lat':predictions3, 'no_depth':predictions4, 'no_currents':predictions5, 'no_eta':predictions5, 'no_sal':predictions5})
plt.show()

fig = rfplt.plt_3_timeseries([5,5,5], 120, 1200, 12000, {'da_T (truth)':da_T.data, 'persistence':persistence, 'full_lr_model':predictions1, '2d':predictions2, 'no_Lat':predictions3, 'no_depth':predictions4, 'no_currents':predictions5, 'no_eta':predictions5, 'no_sal':predictions5})
plt.show()

