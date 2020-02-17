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

#----------------------------
# Set variables for this run
#----------------------------
point = [5,5,3]

#-----------------------------------------------
# Read in netcdf file for 'truth' and get shape
#-----------------------------------------------
print('reading in ds')
datadir = '/data/hpcdata/users/racfur/MITGCM_OUTPUT/20000yr_Windx1.00_mm_diag/'
data_filename=datadir+'cat_tave_5000yrs_SelectedVars.nc'
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
rootdir = '/data/hpcdata/users/racfur/DynamicPrediction/RegressionOutputs/'
pred_dir = rootdir+'ITERATED_PREDICTION_ARRAYS/'

pred1_filename = pred_dir+ 'lr_3dLatDepUVSalEtaPolyDeg2_IterativePredictions.npy'
pred_full = np.load(pred1_filename)

pred2_filename = pred_dir+ 'lr_2dLatDepUVSalEtaPolyDeg2_IterativePredictions.npy'
pred_2d = np.load(pred2_filename)

pred3_filename = pred_dir+ 'lr_3dDepUVSalEtaPolyDeg2_IterativePredictions.npy'
pred_noLat = np.load(pred3_filename)

pred4_filename = pred_dir+ 'lr_3dLatUVSalEtaPolyDeg2_IterativePredictions.npy'
pred_noDepth = np.load(pred4_filename)

pred5_filename = pred_dir+ 'lr_3dLatDepSalEtaPolyDeg2_IterativePredictions.npy'
pred_noCurrents = np.load(pred5_filename)

pred6_filename = pred_dir+ 'lr_3dLatDepUVEtaPolyDeg2_IterativePredictions.npy'
pred_noSal = np.load(pred6_filename)

pred7_filename = pred_dir+ 'lr_3dLatDepUVSalPolyDeg2_IterativePredictions.npy'
pred_noEta = np.load(pred7_filename)

#---------------------------
# Set persistence forecasts
#---------------------------
print('Set perstistence forecast')
persistence = np.zeros((12000, z_size, y_size, x_size))
persistence[:,:,:,:] = da_T[0,:,:,:]

#---------------------------------------------------------------------------
# Plot the predictions against da_T and persistence for a variety of points
#---------------------------------------------------------------------------

fig = rfplt.plt_timeseries(point, 120, {'da_T (truth)':da_T.data, 'persistence':persistence, 'full_lr_model':pred_full, '2d':pred_2d, 'no_Lat':pred_noLat, 'no_depth':pred_noDepth, 'no_currents':pred_noCurrents, 'no_sal':pred_noSal, 'no_eta':pred_noEta})
plt.savefig(rootdir+'PLOTS/multimodel_forecast_10yrs'+str(point[0])+'y'+str(point[1])+'x'+str(point[2])+'y', bbox_inches = 'tight', pad_inches = 0.1)

fig = rfplt.plt_3_timeseries(point, 120, 1200, 12000, {'da_T (truth)':da_T.data, 'persistence':persistence, 'full_lr_model':pred_full, '2d':pred_2d, 'no_Lat':pred_noLat, 'no_depth':pred_noDepth, 'no_currents':pred_noCurrents, 'no_sal':pred_noSal, 'no_eta':pred_noEta})
plt.savefig(rootdir+'PLOTS/multimodel_forecast_1000yrs_z'+str(point[0])+'y'+str(point[1])+'x'+str(point[2])+'y' , bbox_inches = 'tight', pad_inches = 0.1)

