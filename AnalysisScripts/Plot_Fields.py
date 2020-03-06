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

plt.rcParams.update({'font.size': 14})

#----------------------------
# Set variables for this run
#----------------------------
run_vars={'dimension':3, 'lat':True , 'lon':True , 'dep':True , 'current':True , 'sal':True , 'eta':True , 'poly_degree':2}
model_type = 'lr'
exp_prefix = ''

time = 10
point = [5,15,4]

compare_w_truth = False

#----------------------
rootdir = '/data/hpcdata/users/racfur/DynamicPrediction/'+model_type+'_Outputs/'

data_name = cn.create_dataname(run_vars)
exp_name = exp_prefix+data_name

level = point[0]
y_coord = point[1]
x_coord = point[2]

#-------------------------------------------
# Read in netcdf file for 'truth' and shape
#-------------------------------------------
print('reading in ds')
datadir  = '/data/hpcdata/users/racfur/MITGCM_OUTPUT/20000yr_Windx1.00_mm_diag/'
data_filename=datadir+'cat_tave_2000yrs_SelectedVars_masked.nc'
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
pred_dir = rootdir+'ITERATED_PREDICTION_ARRAYS/'

pred_filename = pred_dir+model_type+'_'+exp_name+'_IterativePredictions.npy'
field = np.load(pred_filename)

#--------------------------
# Plot spatial depth plots
#--------------------------
fig, ax, im = rfplt.plot_depth_fld(field[time,:,:,:], 'Temperature Predictions', level, title=None, min_value=None, max_value=None)
plt.savefig(rootdir+'PLOTS/'+model_type+'_'+exp_name+'_predictions_z'+str(level), bbox_inches = 'tight', pad_inches = 0.1)

#-----------------------
# Plot y-cross sections
#-----------------------
fig, ax, im = rfplt.plot_yconst_crss_sec(field[time,:,:,:], 'Temperature Predictions', y_coord, title=None, min_value=None, max_value=None)
plt.savefig(rootdir+'PLOTS/'+model_type+'_'+exp_name+'_predictions_y'+str(y_coord), bbox_inches = 'tight', pad_inches = 0.1)

#-----------------------
# Plot x-cross sections
#-----------------------
fig, ax, im = rfplt.plot_xconst_crss_sec(field[time,:,:,:], 'Temperature Predictions', x_coord, title=None, min_value=None, max_value=None)
plt.savefig(rootdir+'PLOTS/'+model_type+'_'+exp_name+'_predictions_x'+str(x_coord), bbox_inches = 'tight', pad_inches = 0.1)

if compare_w_truth:
   #-------------------------------
   # Plot spatial depth diff plots
   #-------------------------------
   fig = rfplt.plot_depth_fld_diff(da_T[time,:,:,:], 'GCM Temperature', field[time,:,:,:], 'Temperature Predictions', level, title=None)
   plt.savefig(rootdir+'PLOTS/'+model_type+'_'+exp_name+'_Diff_z'+str(level), bbox_inches = 'tight', pad_inches = 0.1)
   
   #---------------------------------
   # Plot y-cross section diff plots
   #---------------------------------
   fig = rfplt.plot_yconst_crss_sec_diff(da_T[time,:,:,:], 'GCM Temperature', field[time,:,:,:],'Temperature Predictions', y_coord, title=None)
   plt.savefig(rootdir+'PLOTS/'+model_type+'_'+exp_name+'_Diff_y'+str(y_coord), bbox_inches = 'tight', pad_inches = 0.1)
   
   #---------------------------------
   # Plot x-cross section diff plots
   #---------------------------------
   fig = rfplt.plot_xconst_crss_sec_diff(da_T[time,:,:,:], 'GCM Temperature', field[time,:,:,:],'Temperature Predictions', x_coord, title=None)
   plt.savefig(rootdir+'PLOTS/'+model_type+'_'+exp_name+'_Diff_x'+str(x_coord), bbox_inches = 'tight', pad_inches = 0.1)
   
   #---------------------------
   # Set persistence forecasts
   #---------------------------
   print('Set perstistence forecast')
   persistence = np.zeros((12000, z_size, y_size, x_size))
   persistence[:,:,:,:] = da_T[0,:,:,:]
   
   #---------------------------------------------------------------------------
   # Plot the predictions against da_T and persistence
   #---------------------------------------------------------------------------
   
   fig = rfplt.plt_timeseries(point, 120, {'da_T (truth)':da_T.data, 'persistence':persistence, model_type: field})
   plt.savefig(rootdir+'PLOTS/'+model_type+'_'+exp_name+'_multimodel_forecast_10yrs_z'+str(point[0])+'y'+str(point[1])+'x'+str(point[2]), bbox_inches = 'tight', pad_inches = 0.1)
   
   fig = rfplt.plt_3_timeseries(point, 120, 1200, 12000, {'da_T (truth)':da_T.data, 'persistence':persistence, 'model':field})
   plt.savefig(rootdir+'PLOTS/'+model_type+'_'+exp_name+'_multimodel_forecast_1000yrs_z'+str(point[0])+'y'+str(point[1])+'x'+str(point[2]), bbox_inches = 'tight', pad_inches = 0.1)
   
