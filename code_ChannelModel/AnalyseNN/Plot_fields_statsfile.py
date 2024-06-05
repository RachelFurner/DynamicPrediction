#!/usr/bin/env python
# coding: utf-8

# Script to plot predictions, truth and the difference from random samples across the 
# training/validation/test set taken from the STATS file. Data is read in and masked.
# The true fields are plotted, the predicted fields are plotted, and the errors are
# plotted. Then a final plot is done which shows the truth, prediction and error all
# together in one.png.

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

dir_name = 'Spits12hrly_UNet2dtransp_histlen1_rolllen1_seed30475'
#epochs_list = ['10', '50', '200']
epochs_list = ['200']
trainval = 'test'

rootdir = '../../../Channel_nn_Outputs/'+dir_name+'/STATS/'

point = [ 2, 8, 6]
time = 5
level = point[0]
y_coord = point[1]
x_coord = point[2]


#---------------------

for epochs in epochs_list:
   model_name = dir_name+'_'+epochs+'epochs'
   
   #------------------------
   print('reading in data')
   #------------------------
   stats_data_filename=rootdir+model_name+'_StatsOutput_'+trainval+'.nc'
   stats_ds = xr.open_dataset(stats_data_filename)
   da_True_Temp = stats_ds['True_Temp']
   da_True_U    = stats_ds['True_U']
   da_True_V    = stats_ds['True_V']
   da_True_Eta  = stats_ds['True_Eta']
   da_True_Temp_Tend = stats_ds['True_Temp_tend']
   da_True_U_Tend    = stats_ds['True_U_tend']
   da_True_V_Tend    = stats_ds['True_V_tend']
   da_True_Eta_Tend  = stats_ds['True_Eta_tend']
   da_Pred_Temp = stats_ds['Pred_Temp']
   da_Pred_U    = stats_ds['Pred_U']
   da_Pred_V    = stats_ds['Pred_V']
   da_Pred_Eta  = stats_ds['Pred_Eta']
   da_Pred_Temp_Tend = stats_ds['Pred_Temp_tend']
   da_Pred_U_Tend    = stats_ds['Pred_U_tend']
   da_Pred_V_Tend    = stats_ds['Pred_V_tend']
   da_Pred_Eta_Tend  = stats_ds['Pred_Eta_tend']
   da_X = stats_ds['X']
   da_Y = stats_ds['Y']
   da_Z = stats_ds['Z']

   #---------------------
   print('Create Masks')
   #---------------------
   grid_filename = '/data/hpcdata/users/racfur/MITgcm/verification/MundayChannelConfig10km_LandSpits/runs/50yr_Cntrl/grid.nc'
   mask_ds   = xr.open_dataset(grid_filename, lock=False)
   masks = np.ones(( da_Z.shape[0], da_Y.shape[0], da_X.shape[0]))
   
   HfacC = mask_ds['HFacC'].values
   masks = np.where( HfacC > 0., 1, 0 )
   
   depths = mask_ds['RC']

   #------------------
   # Plot true fields 
   #------------------
   print('check')
   fig = ChnPlt.plot_depth_fld(np.where(masks[level,:,:]==1, da_True_Temp.values[time,level,:,:], np.nan), 'MITgcm Temperature ('+u'\xb0'+'C)',
                               level, da_X.values, da_Y.values, depths, title=None,
                               minmax = [0, 6.3])
   plt.savefig(rootdir+'EXAMPLE_FIELDS/'+model_name+'_TrueTemp_z'+str(level)+'_'+trainval+'.png', bbox_inches = 'tight', pad_inches = 0.1)
   plt.close()
   
   fig = ChnPlt.plot_depth_fld(np.where(masks[level,:,:]==1, da_True_U.values[time,level,:,:], np.nan), 'MITgcm Eastward Velocity (m/s)', 
                               level, da_X.values, da_Y.values, depths, cmap='PRGn',
                               minmax = [-0.5, 0.5], extend='both')
   plt.savefig(rootdir+'EXAMPLE_FIELDS/'+model_name+'_TrueU_z'+str(level)+'_'+trainval+'.png', bbox_inches = 'tight', pad_inches = 0.1)
   plt.close()
   
   fig = ChnPlt.plot_depth_fld(np.where(masks[level,:,:]==1, da_True_V.values[time,level,:,:], np.nan), 'MITgcm Northward Velocity (m/s)',
                               level, da_X.values, da_Y.values, depths, title=None, cmap='PRGn',
                               minmax = [-0.5, 0.5], extend='both')
   plt.savefig(rootdir+'EXAMPLE_FIELDS/'+model_name+'_TrueV_z'+str(level)+'_'+trainval+'.png', bbox_inches = 'tight', pad_inches = 0.1)
   plt.close()
   
   fig = ChnPlt.plot_depth_fld(np.where(masks[level,:,:]==1, da_True_Eta.values[time,:,:], np.nan), 'MITgcm Sea Surface Height (m)',
                               0, da_X.values, da_Y.values, depths, title=None, cmap='PRGn',
                               minmax = [-0.8, 0.8])
   plt.savefig(rootdir+'EXAMPLE_FIELDS/'+model_name+'_TrueEta_'+trainval+'.png', bbox_inches = 'tight', pad_inches = 0.1)
   plt.close()

   #----------------------
   # Plot true increments
   #----------------------
   fig = ChnPlt.plot_depth_fld(np.where(masks[level,:,:]==1, da_True_Temp_Tend.values[time,level,:,:], np.nan),
                               'MITgcm Temperature Increment ('+u'\xb0'+'C)',
                               level, da_X.values, da_Y.values, depths, title=None, cmap='PRGn',
                               minmax = [-0.6, 0.6], extend='both')
   plt.savefig(rootdir+'EXAMPLE_FIELDS/'+model_name+'_TrueTempTend_z'+str(level)+'_'+trainval+'.png', bbox_inches = 'tight', pad_inches = 0.1)
   plt.close()
   
   fig = ChnPlt.plot_depth_fld(np.where(masks[level,:,:]==1, da_True_U_Tend.values[time,level,:,:], np.nan),
                               'MITgcm Eastward Velocity Increment (m/s)', 
                               level, da_X.values, da_Y.values, depths, cmap='PRGn',
                               minmax = [-0.2, 0.2], extend='both')
   plt.savefig(rootdir+'EXAMPLE_FIELDS/'+model_name+'_TrueUTend_z'+str(level)+'_'+trainval+'.png', bbox_inches = 'tight', pad_inches = 0.1)
   plt.close()
   
   fig = ChnPlt.plot_depth_fld(np.where(masks[level,:,:]==1, da_True_V_Tend.values[time,level,:,:], np.nan),
                               'MITgcm Northward Velocity Increment (m/s)',
                               level, da_X.values, da_Y.values, depths, title=None, cmap='PRGn',
                               minmax = [-0.3, 0.3], extend='both')
   plt.savefig(rootdir+'EXAMPLE_FIELDS/'+model_name+'_TrueVTend_z'+str(level)+'_'+trainval+'.png', bbox_inches = 'tight', pad_inches = 0.1)
   plt.close()
   
   fig = ChnPlt.plot_depth_fld(np.where(masks[level,:,:]==1, da_True_Eta_Tend.values[time,:,:], np.nan),
                               'MITgcm Sea Surface Height Increment (m)',
                               0, da_X.values, da_Y.values, depths, title=None, cmap='PRGn',
                               minmax = [-0.08, 0.08], extend='both')
   plt.savefig(rootdir+'EXAMPLE_FIELDS/'+model_name+'_TrueEtaTend_'+trainval+'.png', bbox_inches = 'tight', pad_inches = 0.1)
   plt.close()

   #-----------------------
   # Plot predicted fields
   #-----------------------
   fig = ChnPlt.plot_depth_fld(np.where(masks[level,:,:]==1, da_Pred_Temp.values[time,level,:,:], np.nan), 'Predicted Temperature ('+u'\xb0'+'C)',
                               level, da_X.values, da_Y.values, depths, title=None,
                               minmax = [0.0, 6.3] )
   plt.savefig(rootdir+'EXAMPLE_FIELDS/'+model_name+'_PredTemp_z'+str(level)+'_'+trainval+'.png', bbox_inches = 'tight', pad_inches = 0.1)
   plt.close()
   
   fig = ChnPlt.plot_depth_fld(np.where(masks[level,:,:]==1, da_Pred_U.values[time,level,:,:], np.nan), 'Predicted Eastward Velocity (m/s)', 
                               level, da_X.values, da_Y.values, depths, cmap='PRGn',
                               minmax = [-0.5, 0.5], extend='both')
   plt.savefig(rootdir+'EXAMPLE_FIELDS/'+model_name+'_PredU_z'+str(level)+'_'+trainval+'.png', bbox_inches = 'tight', pad_inches = 0.1)
   plt.close()
   
   fig = ChnPlt.plot_depth_fld(np.where(masks[level,:,:]==1, da_Pred_V.values[time,level,:,:], np.nan), 'Predicted Northward Velocity (m/s)',
                               level, da_X.values, da_Y.values, depths, title=None, cmap='PRGn',
                               minmax = [-0.5, 0.5], extend='both')
   plt.savefig(rootdir+'EXAMPLE_FIELDS/'+model_name+'_PredV_z'+str(level)+'_'+trainval+'.png', bbox_inches = 'tight', pad_inches = 0.1)
   plt.close()
   
   fig = ChnPlt.plot_depth_fld(np.where(masks[level,:,:]==1, da_Pred_Eta.values[time,:,:], np.nan), 'Predicted Sea Surface Height (m)',
                               0, da_X.values, da_Y.values, depths, title=None, cmap='PRGn',
                               minmax = [-0.8, 0.8])
   plt.savefig(rootdir+'EXAMPLE_FIELDS/'+model_name+'_PredEta_'+trainval+'.png', bbox_inches = 'tight', pad_inches = 0.1)
   plt.close()

   #---------------------------
   # Plot predicted increments
   #---------------------------
   fig = ChnPlt.plot_depth_fld(np.where(masks[level,:,:]==1, da_Pred_Temp_Tend.values[time,level,:,:], np.nan),
                               'Predicted Temperature Increment ('+u'\xb0'+'C)',
                               level, da_X.values, da_Y.values, depths, title=None, cmap='PRGn',
                               minmax = [-0.6, 0.6], extend='both')
   plt.savefig(rootdir+'EXAMPLE_FIELDS/'+model_name+'_PredTempTend_z'+str(level)+'_'+trainval+'.png', bbox_inches = 'tight', pad_inches = 0.1)
   plt.close()
   
   fig = ChnPlt.plot_depth_fld(np.where(masks[level,:,:]==1, da_Pred_U_Tend.values[time,level,:,:], np.nan),
                               'Predicted Eastward Velocity Increment (m/s)', 
                               level, da_X.values, da_Y.values, depths, cmap='PRGn',
                               minmax = [-0.2, 0.2], extend='both')
   plt.savefig(rootdir+'EXAMPLE_FIELDS/'+model_name+'_PredUTend_z'+str(level)+'_'+trainval+'.png', bbox_inches = 'tight', pad_inches = 0.1)
   plt.close()
   
   fig = ChnPlt.plot_depth_fld(np.where(masks[level,:,:]==1, da_Pred_V_Tend.values[time,level,:,:], np.nan),
                               'Predicted Northward Velocity Increment (m/s)',
                               level, da_X.values, da_Y.values, depths, title=None, cmap='PRGn',
                               minmax = [-0.3, 0.3], extend='both')
   plt.savefig(rootdir+'EXAMPLE_FIELDS/'+model_name+'_PredVTend_z'+str(level)+'_'+trainval+'.png', bbox_inches = 'tight', pad_inches = 0.1)
   plt.close()
   
   fig = ChnPlt.plot_depth_fld(np.where(masks[level,:,:]==1, da_Pred_Eta_Tend.values[time,:,:], np.nan),
                               'Predicted Sea Surface Height Increment (m)',
                               0, da_X.values, da_Y.values, depths, title=None, cmap='PRGn',
                               minmax = [-0.08, 0.08], extend='both')
   plt.savefig(rootdir+'EXAMPLE_FIELDS/'+model_name+'_PredEtaTend_'+trainval+'.png', bbox_inches = 'tight', pad_inches = 0.1)
   plt.close()

   #-------------
   # Plot errors
   #-------------
   fig = ChnPlt.plot_depth_fld(np.where(masks[level,:,:]==1, (da_Pred_Temp_Tend.values[time,level,:,:]-da_True_Temp_Tend.values[time,level,:,:]), np.nan),
                               'Temperature Errors ('+u'\xb0'+'C)',
                               level, da_X.values, da_Y.values, depths, title=None, cmap='bwr',
                               minmax = [-0.05, 0.05], extend='both')
                               #minmax = [-0.1, 0.1], extend='both')
   plt.savefig(rootdir+'EXAMPLE_FIELDS/'+model_name+'_TempError_z'+str(level)+'_'+trainval+'.png', bbox_inches = 'tight', pad_inches = 0.1)
   plt.close()
   
   fig = ChnPlt.plot_depth_fld(np.where(masks[level,:,:]==1, (da_Pred_U_Tend.values[time,level,:,:]-da_True_U_Tend.values[time,level,:,:]), np.nan),
                               'Eastward Velocity Errors (m/s)', 
                               level, da_X.values, da_Y.values, depths, cmap='bwr',
                               minmax = [-0.01, 0.01], extend='both')
                               #minmax = [-0.01, 0.01], extend='both')
   plt.savefig(rootdir+'EXAMPLE_FIELDS/'+model_name+'_UError_z'+str(level)+'_'+trainval+'.png', bbox_inches = 'tight', pad_inches = 0.1)
   plt.close()
   
   fig = ChnPlt.plot_depth_fld(np.where(masks[level,:,:]==1, (da_Pred_V_Tend.values[time,level,:,:]-da_True_V_Tend.values[time,level,:,:]), np.nan),
                               'Northward Velocity Errors (m/s)',
                               level, da_X.values, da_Y.values, depths, title=None, cmap='bwr',
                               minmax = [-0.01, 0.01], extend='both')
                               #minmax = [-0.01, 0.01], extend='both')
   plt.savefig(rootdir+'EXAMPLE_FIELDS/'+model_name+'_VError_z'+str(level)+'_'+trainval+'.png', bbox_inches = 'tight', pad_inches = 0.1)
   plt.close()
   
   fig = ChnPlt.plot_depth_fld(np.where(masks[level,:,:]==1, (da_Pred_Eta_Tend.values[time,:,:]-da_True_Eta_Tend.values[time,:,:]), np.nan),
                               'Sea Surface Height Errors (m)',
                               0, da_X.values, da_Y.values, depths, title=None, cmap='bwr',
                               minmax = [-0.006, 0.006], extend='both')
                               #minmax = [-0.005, 0.005], extend='both')
   plt.savefig(rootdir+'EXAMPLE_FIELDS/'+model_name+'_EtaError_'+trainval+'.png', bbox_inches = 'tight', pad_inches = 0.1)
   plt.close()

   #-------------------------------------
   # Plot spatial depth difference plots
   # These plots have either 3 panes - the truth, the prediction and the diff, or two panes - the prediction and diff from truth
   #-------------------------------------
   fig = ChnPlt.plot_depth_fld_diff(np.where(masks[level,:,:]==1, da_True_Temp.values[time,level,:,:], np.nan), 'MITgcm Temperature',
                                    np.where(masks[level,:,:]==1, da_Pred_Temp.values[time,level,:,:], np.nan), 'Predicted Temperature',
                                    level, da_X.values, da_Y.values, depths,
                                    flds_minmax= [-0.6, 0.6], diff_minmax = [-0.1, 0.1], panes=2, extend='both')
   plt.savefig(rootdir+'EXAMPLE_FIELDS/'+model_name+'_Temp_diff_z'+str(level)+'_'+trainval+'.png', bbox_inches = 'tight', pad_inches = 0.1)
   plt.close()
   
   fig = ChnPlt.plot_depth_fld_diff(np.where(masks[level,:,:]==1, da_True_U.values[time,level,:,:], np.nan), 'MITgcm Eastward Velocity', 
                                    np.where(masks[level,:,:]==1, da_Pred_U.values[time,level,:,:], np.nan), 'Predicted Eastward Velocity',
                                    level, da_X.values, da_Y.values, depths, cmap='PRGn',
                                    flds_minmax = [-0.2, 0.2], diff_minmax = [-0.025, 0.025], panes=2, extend='both')
   plt.savefig(rootdir+'EXAMPLE_FIELDS/'+model_name+'_U_diff_z'+str(level)+'_'+trainval+'.png', bbox_inches = 'tight', pad_inches = 0.1)
   plt.close()
   
   fig = ChnPlt.plot_depth_fld_diff(np.where(masks[level,:,:]==1, da_True_V.values[time,level,:,:], np.nan), 'MITgcm Northward Velocity',
                                    np.where(masks[level,:,:]==1, da_Pred_V.values[time,level,:,:], np.nan), 'Predicted Northward Velocity',
                                    level, da_X.values, da_Y.values, depths, cmap='PRGn',
                                    flds_minmax = [-0.3, 0.3], diff_minmax = [-0.025, 0.025], panes=2, extend='both')
   plt.savefig(rootdir+'EXAMPLE_FIELDS/'+model_name+'_V_diff_z'+str(level)+'_'+trainval+'.png', bbox_inches = 'tight', pad_inches = 0.1)
   plt.close()
   
   fig = ChnPlt.plot_depth_fld_diff(np.where(masks[level,:,:]==1, da_True_Eta.values[time,:,:], np.nan), 'MITgcm Sea Surface Height',
                                    np.where(masks[level,:,:]==1, da_Pred_Eta.values[time,:,:], np.nan), 'Predicted Sea Surface Height',
                                    0, da_X.values, da_Y.values, depths, cmap='PRGn',
                                    flds_minmax = [-0.08, 0.08], diff_minmax = [-0.006, 0.006], panes=2, extend='both')
   plt.savefig(rootdir+'EXAMPLE_FIELDS/'+model_name+'_Eta_diff_'+trainval+'.png', bbox_inches = 'tight', pad_inches = 0.1)
   plt.close()

   #-------------------------------------
   # Plot spatial depth difference plots for increments 
   # These plots have either 3 panes - the truth, the prediction and the diff, or two panes - the prediction and diff from truth
   #-------------------------------------
   fig = ChnPlt.plot_depth_fld_diff(np.where(masks[level,:,:]==1, da_True_Temp_Tend.values[time,level,:,:], np.nan), 'MITgcm Temperature Increment',
                                    np.where(masks[level,:,:]==1, da_Pred_Temp_Tend.values[time,level,:,:], np.nan), 'Predicted Temperature Increment',
                                    level, da_X.values, da_Y.values, depths,
                                    flds_minmax = [-0.6, 0.6], diff_minmax = [-0.1, 0.1],
                                    panes=2, cmap='PRGn', extend='both')
   plt.savefig(rootdir+'EXAMPLE_FIELDS/'+model_name+'_TempTend_diff_z'+str(level)+'_'+trainval+'.png', bbox_inches = 'tight', pad_inches = 0.1)
   plt.close()
   
   fig = ChnPlt.plot_depth_fld_diff(np.where(masks[level,:,:]==1, da_True_U_Tend.values[time,level,:,:], np.nan), 'MITgcm Eastward Velocity Increment', 
                                    np.where(masks[level,:,:]==1, da_Pred_U_Tend.values[time,level,:,:], np.nan), 'Predicted Eastward Velocity Increment',
                                    level, da_X.values, da_Y.values, depths,
                                    flds_minmax = [-0.2, 0.2], diff_minmax = [-0.025, 0.025],
                                    panes=2, cmap='PRGn', extend='both')
   plt.savefig(rootdir+'EXAMPLE_FIELDS/'+model_name+'_UTend_diff_z'+str(level)+'_'+trainval+'.png', bbox_inches = 'tight', pad_inches = 0.1)
   plt.close()
   
   fig = ChnPlt.plot_depth_fld_diff(np.where(masks[level,:,:]==1, da_True_V_Tend.values[time,level,:,:], np.nan), 'MITgcm Northward Velocity Increment',
                                    np.where(masks[level,:,:]==1, da_Pred_V_Tend.values[time,level,:,:], np.nan), 'Predicted Northward Velocity Increment',
                                    level, da_X.values, da_Y.values, depths,
                                    flds_minmax = [-0.3, 0.3], diff_minmax = [-0.025, 0.025],
                                    panes=2, cmap='PRGn', extend='both')
   plt.savefig(rootdir+'EXAMPLE_FIELDS/'+model_name+'_VTend_diff_z'+str(level)+'_'+trainval+'.png', bbox_inches = 'tight', pad_inches = 0.1)
   plt.close()
   
   fig = ChnPlt.plot_depth_fld_diff(np.where(masks[level,:,:]==1, da_True_Eta_Tend.values[time,:,:], np.nan), 'MITgcm Sea Surface Height Increment',
                                    np.where(masks[level,:,:]==1, da_Pred_Eta_Tend.values[time,:,:], np.nan), 'Predicted Sea Surface Height Increment',
                                    0, da_X.values, da_Y.values, depths,
                                    flds_minmax = [-0.08, 0.08], diff_minmax = [-0.006, 0.006],
                                    panes=2, cmap='PRGn', extend='both')
   plt.savefig(rootdir+'EXAMPLE_FIELDS/'+model_name+'_EtaTend_diff_'+trainval+'.png', bbox_inches = 'tight', pad_inches = 0.1)
   plt.close()


