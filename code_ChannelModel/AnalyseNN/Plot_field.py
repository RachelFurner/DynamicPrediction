#!/usr/bin/env python
# coding: utf-8

print('import packages')
import sys
sys.path.append('../')
from Tools import Channel_Model_Plotting as ChnPlt

import numpy as np
import matplotlib.pyplot as plt
import os
import xarray as xr
from netCDF4 import Dataset

plt.rcParams.update({'font.size': 14})

#----------------------------
# Set variables for this run
#----------------------------

dir_name = 'Spits_UNet2dtransp_histlen1_seed30475'
epochs_list = ['200']
trainval = 'training'

rootdir = '../../../Channel_nn_Outputs/'+dir_name

point = [ 2, 8, 6]
time = 5
level = point[0]
y_coord = point[1]
x_coord = point[2]

for epochs in epochs_list:
   model_name = dir_name+'_'+epochs+'epochs'
   
   #------------------------
   print('reading in data')
   #------------------------
   stats_data_filename=rootdir+'/STATS/'+model_name+'_StatsOutput_'+trainval+'.nc'
   stats_ds = xr.open_dataset(stats_data_filename)
   da_True_Temp = stats_ds['True_Temp']
   da_True_U    = stats_ds['True_U']
   da_True_V    = stats_ds['True_V']
   da_True_Eta  = stats_ds['True_Eta']
   da_Pred_Temp = stats_ds['Pred_Temp']
   da_Pred_U    = stats_ds['Pred_U']
   da_Pred_V    = stats_ds['Pred_V']
   da_Pred_Eta  = stats_ds['Pred_Eta']
   da_X = stats_ds['X']
   da_Y = stats_ds['Y']
   da_Z = stats_ds['Z']

   #---------------------
   print('Create Masks')
   #---------------------
   grid_filename = '/data/hpcdata/users/racfur/MITgcm/verification/MundayChannelConfig10km_LandSpits/runs/50yr_Cntrl_orig/grid.nc'
   mask_ds   = xr.open_dataset(grid_filename, lock=False)
   masks = np.ones(( da_Z.shape[0], da_Y.shape[0], da_X.shape[0]))

   HfacC = mask_ds['HFacC'].values
   masks = np.where( HfacC > 0., 1, 0 )
 
   #--------------------------
   # Plot spatial depth plots
   #--------------------------
   fig = ChnPlt.plot_depth_fld_diff(np.where(masks[level,:,:]==1, da_True_Temp.values[time,level,:,:], np.nan), 'True Temperature',
                                    np.where(masks[level,:,:]==1, da_Pred_Temp.values[time,level,:,:], np.nan), 'Predicted Temperature',
                                    level, da_X.values, da_Y.values, da_Z.values, title=None)
   plt.savefig(rootdir+'/PLOTS/'+model_name+'_Temp_diff_z'+str(level)+'.png', bbox_inches = 'tight', pad_inches = 0.1)
   
   fig = ChnPlt.plot_depth_fld_diff(np.where(masks[level,:,:]==1, da_True_U.values[time,level,:,:], np.nan), 'True East-West Velocity', 
                                    np.where(masks[level,:,:]==1, da_Pred_U.values[time,level,:,:], np.nan), 'Predicted East-West Velocity',
                                    level, da_X.values, da_Y.values, da_Z.values)
   plt.savefig(rootdir+'/PLOTS/'+model_name+'_U_diff_z'+str(level)+'.png', bbox_inches = 'tight', pad_inches = 0.1)
   
   fig = ChnPlt.plot_depth_fld_diff(np.where(masks[level,:,:]==1, da_True_V.values[time,level,:,:], np.nan), 'True North-South Velocity',
                                    np.where(masks[level,:,:]==1, da_Pred_V.values[time,level,:,:], np.nan), 'Predicted North-South Velocity',
                                    level, da_X.values, da_Y.values, da_Z.values, title=None)
   plt.savefig(rootdir+'/PLOTS/'+model_name+'_V_diff_z'+str(level)+'.png', bbox_inches = 'tight', pad_inches = 0.1)
   
   fig = ChnPlt.plot_depth_fld_diff(np.where(masks[level,:,:]==1, da_True_Eta.values[time,:,:], np.nan), 'True Sea Surface Height',
                                    np.where(masks[level,:,:]==1, da_Pred_Eta.values[time,:,:], np.nan), 'Predicted Sea Surface Height',
                                    0, da_X.values, da_Y.values, da_Z.values, title=None)
   plt.savefig(rootdir+'/PLOTS/'+model_name+'_Eta_diff_z'+str(level)+'.png', bbox_inches = 'tight', pad_inches = 0.1)
