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
point = [5,15,4]

model_type = 'lr'

rootdir = '/data/hpcdata/users/racfur/DynamicPrediction/'+model_type+'_Outputs/'
nn_dir = '/data/hpcdata/users/racfur/DynamicPrediction/nn_Outputs/'
lr_dir = '/data/hpcdata/users/racfur/DynamicPrediction/lr_Outputs/'

#-----------------------------------------------
# Read in netcdf file for 'truth' and get shape
#-----------------------------------------------
print('reading in ds')
datadir = '/data/hpcdata/users/racfur/MITGCM_OUTPUT/20000yr_Windx1.00_mm_diag/'
data_filename=datadir+'cat_tave_5000yrs_SelectedVars_masked.nc'
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
lr_pred_dir = lr_dir+'ITERATED_PREDICTION_ARRAYS/'
nn_pred_dir = nn_dir+'ITERATED_PREDICTION_ARRAYS/'

pred1_filename = lr_pred_dir+ 'lr_3dLatLonDepUVSalEtaPolyDeg2_IterativePredictions.npy'
pred_full = np.load(pred1_filename)

pred2_filename = lr_pred_dir+ 'lr_2dLatLonDepUVSalEtaPolyDeg2_IterativePredictions.npy'
pred_2d = np.load(pred2_filename)

pred3_filename = lr_pred_dir+ 'lr_3dLonDepUVSalEtaPolyDeg2_IterativePredictions.npy'
pred_noLat = np.load(pred3_filename)

pred4_filename = lr_pred_dir+ 'lr_3dLatDepUVSalEtaPolyDeg2_IterativePredictions.npy'
pred_noLon = np.load(pred4_filename)

pred5_filename = lr_pred_dir+ 'lr_3dLatLonUVSalEtaPolyDeg2_IterativePredictions.npy'
pred_noDepth = np.load(pred5_filename)

pred6_filename = lr_pred_dir+ 'lr_3dLatLonDepSalEtaPolyDeg2_IterativePredictions.npy'
pred_noCurrents = np.load(pred6_filename)

pred7_filename = lr_pred_dir+ 'lr_3dLatLonDepUVEtaPolyDeg2_IterativePredictions.npy'
pred_noSal = np.load(pred7_filename)

pred8_filename = lr_pred_dir+ 'lr_3dLatLonDepUVSalPolyDeg2_IterativePredictions.npy'
pred_noEta = np.load(pred8_filename)

#pred9_filename = nn_pred_dir+ 'nn_3dLatLonDepUVSalEtaPolyDeg2_IterativePredictions.npy'
#pred_nn = np.load(pred9_filename)

#---------------------------
# Set persistence forecasts
#---------------------------
print('Set perstistence forecast')
persistence = np.zeros((12000, z_size, y_size, x_size))
persistence[:,:,:,:] = da_T[0,:,:,:]

#---------------------------------------------------------------------------
# Plot the predictions against da_T and persistence for a variety of points
#---------------------------------------------------------------------------

#fig = rfplt.plt_timeseries(point, 120, {'da_T (truth)':da_T.data, 'persistence':persistence, 'full_lr_model':pred_full, 'full_nn_model':pred_nn})
#plt.savefig(rootdir+'PLOTS/multimodel_forecast_10yrs_z'+str(point[0])+'y'+str(point[1])+'x'+str(point[2]), bbox_inches = 'tight', pad_inches = 0.1)
#
#fig = rfplt.plt_3_timeseries(point, 120, 1200, 12000, {'da_T (truth)':da_T.data, 'persistence':persistence, 'full_lr_model':pred_full, 'full_nn_model':pred_nn})
#plt.savefig(rootdir+'PLOTS/multimodel_forecast_1000yrs_z'+str(point[0])+'y'+str(point[1])+'x'+str(point[2]), bbox_inches = 'tight', pad_inches = 0.1)

fig = rfplt.plt_timeseries(point, 120, {'da_T (truth)':da_T.data, 'persistence':persistence, 'full_lr_model':pred_full, '2d':pred_2d, 'no_Lat':pred_noLat, 'no_Lon':pred_noLon, 'no_depth':pred_noDepth, 'no_currents':pred_noCurrents, 'no_sal':pred_noSal, 'no_eta':pred_noEta})
plt.savefig(rootdir+'PLOTS/multimodel_forecast_10yrs'+str(point[0])+'y'+str(point[1])+'x'+str(point[2]), bbox_inches = 'tight', pad_inches = 0.1)

fig = rfplt.plt_3_timeseries(point, 120, 1200, 12000, {'da_T (truth)':da_T.data, 'persistence':persistence, 'full_lr_model':pred_full, '2d':pred_2d, 'no_Lat':pred_noLat, 'no_Lon':pred_noLon, 'no_depth':pred_noDepth, 'no_currents':pred_noCurrents, 'no_sal':pred_noSal, 'no_eta':pred_noEta})
plt.savefig(rootdir+'PLOTS/multimodel_forecast_1000yrs_z'+str(point[0])+'y'+str(point[1])+'x'+str(point[2]) , bbox_inches = 'tight', pad_inches = 0.1)

