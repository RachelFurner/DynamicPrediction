#!/usr/bin/env python
# coding: utf-8

print('import packages')
import sys
sys.path.append('../Tools')
import CreateDataName as cn
import Model_Plotting as rfplt

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from skimage.util import view_as_windows
import os
import xarray as xr
import pickle

plt.rcParams.update({'font.size': 14})

#----------------------------
# Set variables for this run
#----------------------------
point = [ 5,15,5]

model_type = 'lr'

time_step = '24hrs'

#------------------------------

rootdir = '../../../'+model_type+'_Outputs/'
nn_dir = '../../../nn_Outputs/'
lr_dir = '../../../lr_Outputs/'

if time_step == '1mnth':
   DIR  = '/data/hpcdata/users/racfur/MITGCM_OUTPUT/20000yr_Windx1.00_mm_diag/'
   data_filename=DIR+'cat_tave_2000yrs_SelectedVars_masked_withBolus.nc'
elif time_step == '24hrs':
   DIR  = '/data/hpcdata/users/racfur/MITGCM_OUTPUT/100yr_Windx1.00_FrequentOutput/'
   data_filename=DIR+'cat_tave_50yr_SelectedVars_masked_withBolus.nc'
else:
   print('ERROR!!! No suitable time step given!!')

#-----------------------------------------------
# Read in netcdf file for 'truth' and get shape
#-----------------------------------------------
print('reading in ds')
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


## NEED TO UPDATE LOAD COMMANDS TO USE THIS....! 

#pred_filename = lr_pred_dir+time_step+'_3dLatLonDepUVBolSalEtaDnsPolyDeg2_AB3_IterativePredictions_5.npz'
#pred_full, delT_tmp, mask_tmp = np.load(pred_filename).values()
#
#pred_filename = lr_pred_dir+time_step+'_2dLatLonDepUVBolSalEtaDnsPolyDeg2_AB3_IterativePredictions_5.npz'
#pred_2d, delT_tmp, mask_tmp = np.load(pred_filename).values()
#
#pred_filename = lr_pred_dir+time_step+'_3dLonDepUVBolSalEtaDnsPolyDeg2_AB3_IterativePredictions_5.npz'
#pred_noLat, delT_tmp, mask_tmp = np.load(pred_filename).values()
#
#pred_filename = lr_pred_dir+time_step+'_3dLatDepUVBolSalEtaDnsPolyDeg2_AB3_IterativePredictions_5.npz'
#pred_noLon, delT_tmp, mask_tmp = np.load(pred_filename).values()
#
#pred_filename = lr_pred_dir+time_step+'_3dLatLonUVBolSalEtaDnsPolyDeg2_AB3_IterativePredictions_5.npz'
#pred_noDep, delT_tmp, mask_tmp = np.load(pred_filename).values()
#
#pred_filename = lr_pred_dir+time_step+'_3dLatLonDepBolSalEtaDnsPolyDeg2_AB3_IterativePredictions_5.npz'
#pred_noCur, delT_tmp, mask_tmp = np.load(pred_filename).values()
#
#pred_filename = lr_pred_dir+time_step+'_3dLatLonDepUVSalEtaDnsPolyDeg2_AB3_IterativePredictions_5.npz'
#pred_noBol, delT_tmp, mask_tmp = np.load(pred_filename).values()
#
#pred_filename = lr_pred_dir+time_step+'_3dLatLonDepUVBolEtaDnsPolyDeg2_AB3_IterativePredictions_5.npz'
#pred_noSal, delT_tmp, mask_tmp = np.load(pred_filename).values()
#
#pred_filename = lr_pred_dir+time_step+'_3dLatLonDepUVBolSalDnsPolyDeg2_AB3_IterativePredictions_5.npz'
#pred_noEta, delT_tmp, mask_tmp = np.load(pred_filename).values()
#
#pred_filename = lr_pred_dir+time_step+'_3dLatLonDepUVBolSalEtaPolyDeg2_AB3_IterativePredictions_5.npz'
#pred_noDen, delT_tmp, mask_tmp = np.load(pred_filename).values()
#
#pred_filename = lr_pred_dir+time_step+'_3dLatLonDepUVBolSalEtaDnsPolyDeg1_AB3_IterativePredictions_5.npz'
#pred_noPoly, delT_tmp, mask_tmp = np.load(pred_filename).values()
#
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

fig = rfplt.plt_timeseries(point, 360, {'MITGCM':da_T.data, 'full_lr_model':pred_full, '2d':pred_2d,
                                        #'no_Lat':pred_noLat, 'no_Lon':pred_noLon, 'no_depth':pred_noDep, 
                                        'no_currents':pred_noCur, 'no_bolus_velocity':pred_noBol,
                                        'no_sal':pred_noSal, 'no_eta':pred_noEta, 'no_density':pred_noDen, 'no_polynomial_terms':pred_noPoly},
                           time_step=time_step)
plt.savefig(rootdir+'PLOTS/multimodel_forecast_1yr_z'+str(point[0])+'y'+str(point[1])+'x'+str(point[2]), bbox_inches = 'tight', pad_inches = 0.1)
