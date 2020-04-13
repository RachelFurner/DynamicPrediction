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
run_vars={'dimension':3, 'lat':True , 'lon':True , 'dep':True , 'current':True , 'sal':True , 'eta':True , 'density':True , 'poly_degree':2}
model_type = 'lr'
data_prefix = 'WithThroughFlow_'
exp_prefix = ''
ItMethod = 'AB1'

time = 10
point = [30,55,5]

compare_w_truth = True 

#----------------------
rootdir = '/data/hpcdata/users/racfur/DynamicPrediction/'+model_type+'_Outputs/'

data_name = cn.create_dataname(run_vars)
data_name = data_prefix+data_name
exp_name = ItMethod+'_'+data_name
exp_name = exp_prefix+exp_name

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
land_mask=ds['Mask'].values

z_size = da_T.shape[1]
y_size = da_T.shape[2]
x_size = da_T.shape[3]

print('da_T.shape')
print(da_T.shape)

#----------------------
# Read in predictions
#----------------------
pred_dir = rootdir+'ITERATED_PREDICTION_ARRAYS/'

pred_filename = pred_dir+exp_name+'_IterativePredictions_0.npz'
field, delT = np.load(pred_filename).values()

#-----------
# mask data
#-----------
MITGCM_temp  = np.where(land_mask==1, da_T,  np.nan)
pred_temp = np.where(land_mask==1, field, np.nan)
pred_delT  = np.where(land_mask==1, delT,  np.nan)

#--------------------------
# Plot spatial depth plots
#--------------------------
fig, ax, im = rfplt.plot_depth_fld(pred_temp[time,:,:,:], 'Predicted Temperature', level, title=None, min_value=None, max_value=None)
plt.savefig(rootdir+'PLOTS/'+exp_name+'_predictions_z'+str(level), bbox_inches = 'tight', pad_inches = 0.1)

#-----------------------
# Plot y-cross sections
#-----------------------
fig, ax, im = rfplt.plot_yconst_crss_sec(pred_temp[time,:,:,:], 'Predicted Temperature', y_coord, title=None, min_value=None, max_value=None)
plt.savefig(rootdir+'PLOTS/'+exp_name+'_predictions_y'+str(y_coord), bbox_inches = 'tight', pad_inches = 0.1)

#-----------------------
# Plot x-cross sections
#-----------------------
fig, ax, im = rfplt.plot_xconst_crss_sec(pred_temp[time,:,:,:], 'Predicted Temperature', x_coord, title=None, min_value=None, max_value=None)
plt.savefig(rootdir+'PLOTS/'+exp_name+'_predictions_x'+str(x_coord), bbox_inches = 'tight', pad_inches = 0.1)

if compare_w_truth:
   #-------------------------------
   # Plot spatial depth diff plots
   #-------------------------------
   fig = rfplt.plot_depth_fld_diff(MITGCM_temp[time,:,:,:], 'GCM Temperature', pred_temp[time,:,:,:], 'Predicted Temperature', level, title=None)
   plt.savefig(rootdir+'PLOTS/'+exp_name+'_Diff_z'+str(level), bbox_inches = 'tight', pad_inches = 0.1)
   
   #---------------------------------
   # Plot y-cross section diff plots
   #---------------------------------
   fig = rfplt.plot_yconst_crss_sec_diff(MITGCM_temp[time,:,:,:], 'GCM Temperature', pred_temp[time,:,:,:],'Predicted Temperature', y_coord, title=None)
   plt.savefig(rootdir+'PLOTS/'+exp_name+'_Diff_y'+str(y_coord), bbox_inches = 'tight', pad_inches = 0.1)
   
   #---------------------------------
   # Plot x-cross section diff plots
   #---------------------------------
   fig = rfplt.plot_xconst_crss_sec_diff(MITGCM_temp[time,:,:,:], 'GCM Temperature', pred_temp[time,:,:,:],'Predicted Temperature', x_coord, title=None)
   plt.savefig(rootdir+'PLOTS/'+exp_name+'_Diff_x'+str(x_coord), bbox_inches = 'tight', pad_inches = 0.1)
   
   #---------------------------
   # Set persistence forecasts
   #---------------------------
   print('Set perstistence forecast')
   persistence = np.zeros((12000, z_size, y_size, x_size))
   persistence[:,:,:,:] = MITGCM_temp[0,:,:,:]
   
   #---------------------------------------------------------------------------
   # Plot the predictions against MITGCM_temp and persistence
   #---------------------------------------------------------------------------
   
   fig = rfplt.plt_timeseries(point, 120, {'MITGCM_temp (truth)':MITGCM_temp, 'persistence':persistence, model_type: pred_temp})
   plt.savefig(rootdir+'PLOTS/'+exp_name+'_10yrs_z'+str(point[0])+'y'+str(point[1])+'x'+str(point[2]), bbox_inches = 'tight', pad_inches = 0.1)
   
   fig = rfplt.plt_3_timeseries(point, 120, 1200, 12000, {'MITGCM_temp (truth)':MITGCM_temp, 'persistence':persistence, 'model':pred_temp})
   plt.savefig(rootdir+'PLOTS/'+exp_name+'_1000yrs_z'+str(point[0])+'y'+str(point[1])+'x'+str(point[2]), bbox_inches = 'tight', pad_inches = 0.1)
   
