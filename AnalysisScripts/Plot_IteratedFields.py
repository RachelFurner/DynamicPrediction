#!/usr/bin/env python
# coding: utf-8

print('import packages')
import sys
sys.path.append('../')
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
run_vars={'dimension':2, 'lat':True , 'lon':True , 'dep':True , 'current':True , 'bolus_vel':True , 'sal':True , 'eta':True , 'density':True , 'poly_degree':2}
model_type = 'lr'

time_step = '24hrs'
data_prefix = ''
model_prefix = '' 
exp_prefix = ''
ItMethod = 'AB3'

time = 100
point = [ 5,10, 5]

compare_w_truth = True 

#----------------------
rootdir = '../'+model_type+'_Outputs/'

data_name = cn.create_dataname(run_vars)
data_name = data_prefix+data_name+'_'+time_step 
model_name = model_prefix+data_name
exp_name = exp_prefix+model_name+'_'+ItMethod

level = point[0]
y_coord = point[1]
x_coord = point[2]

if time_step == '1mnth':
   DIR  = '/data/hpcdata/users/racfur/MITGCM_OUTPUT/20000yr_Windx1.00_mm_diag/'
   data_filename=DIR+'cat_tave_2000yrs_SelectedVars_masked_withBolus.nc'
elif time_step == '24hrs':
   DIR  = '/data/hpcdata/users/racfur/MITGCM_OUTPUT/100yr_Windx1.00_FrequentOutput/'
   data_filename=DIR+'cat_tave_50yr_SelectedVars_masked_withBolus.nc'
else:
   print('ERROR!!! No suitable time step chosen!!')

#-------------------------------------------
# Read in netcdf file for 'truth' and shape
#-------------------------------------------
print('reading in ds')
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

pred_filename = pred_dir+exp_name+'_IterativePredictions_5.npz'
#field, delT, mask = np.load(pred_filename).values()
pred_data = np.load(pred_filename)
field = pred_data['arr_0']
delT  = pred_data['arr_1']
mask  = pred_data['arr_2']

#-----------
# mask data
#-----------
MITGCM_temp  = np.where(land_mask==1, da_T,  np.nan)
pred_temp = np.where(land_mask==1, field, np.nan)
pred_delT  = np.where(land_mask==1, delT,  np.nan)

#--------------------------
# Plot spatial depth plots
#--------------------------
fig, ax, im = rfplt.plot_depth_fld(pred_temp[time,:,:,:], 'Predicted Temperature', level, ds['X'].values, ds['Y'].values,
                                   ds['Z'].values, title=None, min_value=None, max_value=None)
plt.savefig(rootdir+'PLOTS/'+model_name+'/'+exp_name+'_IteratedData_predictions_z'+str(level)+'_'+str(time), bbox_inches = 'tight', pad_inches = 0.1)

#-----------------------
# Plot y-cross sections
#-----------------------
fig, ax, im = rfplt.plot_yconst_crss_sec(pred_temp[time,:,:,:], 'Predicted Temperature', y_coord, ds['X'].values, 
                                         ds['Y'].values, ds['Z'].values, title=None, min_value=None, max_value=None)
plt.savefig(rootdir+'PLOTS/'+model_name+'/'+exp_name+'_IteratedData_predictions_y'+str(y_coord)+'_'+str(time), bbox_inches = 'tight', pad_inches = 0.1)

#-----------------------
# Plot x-cross sections
#-----------------------
fig, ax, im = rfplt.plot_xconst_crss_sec(pred_temp[time,:,:,:], 'Predicted Temperature', x_coord, ds['X'].values, 
                                         ds['Y'].values, ds['Z'].values, title=None, min_value=None, max_value=None)
plt.savefig(rootdir+'PLOTS/'+model_name+'/'+exp_name+'_IteratedData_predictions_x'+str(x_coord)+'_'+str(time), bbox_inches = 'tight', pad_inches = 0.1)

if compare_w_truth:
   #-------------------------------
   # Plot spatial depth diff plots
   #-------------------------------
   fig = rfplt.plot_depth_fld_diff(MITGCM_temp[time,:,:,:], 'GCM Temperature', pred_temp[time,:,:,:], 'Predicted Temperature', 
                                   level, ds['X'].values, ds['Y'].values, ds['Z'].values, title=None)
   plt.savefig(rootdir+'PLOTS/'+model_name+'/'+exp_name+'_IteratedData_Diff_z'+str(level)+'_'+str(time), bbox_inches = 'tight', pad_inches = 0.1)
   
   #---------------------------------
   # Plot y-cross section diff plots
   #---------------------------------
   fig = rfplt.plot_yconst_crss_sec_diff(MITGCM_temp[time,:,:,:], 'GCM Temperature', pred_temp[time,:,:,:],'Predicted Temperature', 
                                         y_coord, ds['X'].values, ds['Y'].values, ds['Z'].values, title=None)
   plt.savefig(rootdir+'PLOTS/'+model_name+'/'+exp_name+'_IteratedData_Diff_y'+str(y_coord)+'_'+str(time), bbox_inches = 'tight', pad_inches = 0.1)
   
   #---------------------------------
   # Plot x-cross section diff plots
   #---------------------------------
   fig = rfplt.plot_xconst_crss_sec_diff(MITGCM_temp[time,:,:,:], 'GCM Temperature', pred_temp[time,:,:,:],'Predicted Temperature', 
                                         x_coord, ds['X'].values, ds['Y'].values, ds['Z'].values, title=None)
   plt.savefig(rootdir+'PLOTS/'+model_name+'/'+exp_name+'_IteratedData_Diff_x'+str(x_coord)+'_'+str(time), bbox_inches = 'tight', pad_inches = 0.1)
   
   #---------------------------
   # Set persistence forecasts
   #---------------------------
   print('Set perstistence forecast')
   persistence = np.zeros((12000, z_size, y_size, x_size))
   persistence[:,:,:,:] = MITGCM_temp[0,:,:,:]
   
   #---------------------------------------------------------------------------
   # Plot timeseries of the predictions against MITGCM_temp and persistence
   #---------------------------------------------------------------------------
   
   if time_step == '1mnth':
      length = 120
      title_time = '10yrs'
   elif time_step == '24hrs':
      length = 180
      title_time = '6mnths'
   fig = rfplt.plt_timeseries(point, length, {'MITGCM_temp (truth)':MITGCM_temp, model_type: pred_temp},
                              ylim=(19.4 ,19.8 ), time_step=time_step)
                              #ylim=(None), time_step=time_step)
   plt.savefig(rootdir+'PLOTS/'+model_name+'/'+exp_name+'_'+title_time+'_z'+str(point[0])+'y'+str(point[1])+'x'+str(point[2]), bbox_inches = 'tight', pad_inches = 0.1)
   
