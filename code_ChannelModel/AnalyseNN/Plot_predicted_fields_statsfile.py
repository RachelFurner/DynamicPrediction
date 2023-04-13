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

dir_name = 'Spits12hrly_UNet2dtransp_histlen1_predlen1_seed30475'
epochs_list = ['10', '20', '40', '50', '100', '150', '200']
trainval = 'test'

rootdir = '../../../Channel_nn_Outputs/'+dir_name+'/STATS/'

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
 
   #-----------------------------------
   # Plot spatial depth plots of truth
   #-----------------------------------
   print('check')
   fig = ChnPlt.plot_depth_fld(np.where(masks[level,:,:]==1, da_True_Temp.values[time,level,:,:], np.nan), 'True Temperature',
                               level, da_X.values, da_Y.values, da_Z.values, title=None)
   plt.savefig(rootdir+'/EXAMPLE_PREDICTIONS/'+model_name+'_TrueTemp_z'+str(level)+'_'+trainval+'.png', bbox_inches = 'tight', pad_inches = 0.1)
   plt.close()
   
   fig = ChnPlt.plot_depth_fld(np.where(masks[level,:,:]==1, da_True_U.values[time,level,:,:], np.nan), 'True East-West Velocity', 
                               level, da_X.values, da_Y.values, da_Z.values, cmap='PRGn')
   plt.savefig(rootdir+'/EXAMPLE_PREDICTIONS/'+model_name+'_TrueU_z'+str(level)+'_'+trainval+'.png', bbox_inches = 'tight', pad_inches = 0.1)
   plt.close()
   
   fig = ChnPlt.plot_depth_fld(np.where(masks[level,:,:]==1, da_True_V.values[time,level,:,:], np.nan), 'True North-South Velocity',
                               level, da_X.values, da_Y.values, da_Z.values, title=None, cmap='PRGn')
   plt.savefig(rootdir+'/EXAMPLE_PREDICTIONS/'+model_name+'_TrueV_z'+str(level)+'_'+trainval+'.png', bbox_inches = 'tight', pad_inches = 0.1)
   plt.close()
   
   fig = ChnPlt.plot_depth_fld(np.where(masks[level,:,:]==1, da_True_Eta.values[time,:,:], np.nan), 'True Sea Surface Height',
                               0, da_X.values, da_Y.values, da_Z.values, title=None, cmap='PRGn')
   plt.savefig(rootdir+'/EXAMPLE_PREDICTIONS/'+model_name+'_TrueEta_'+trainval+'.png', bbox_inches = 'tight', pad_inches = 0.1)
   plt.close()

   fig = ChnPlt.plot_depth_fld(np.where(masks[level,:,:]==1, da_True_Temp_Tend.values[time,level,:,:], np.nan), 'True Temperature Increment',
                               level, da_X.values, da_Y.values, da_Z.values, title=None, cmap='PRGn', min_value=-0.6, max_value=0.6, extend='both')
   plt.savefig(rootdir+'/EXAMPLE_PREDICTIONS/'+model_name+'_TrueTempTend_z'+str(level)+'_'+trainval+'.png', bbox_inches = 'tight', pad_inches = 0.1)
   plt.close()
   
   fig = ChnPlt.plot_depth_fld(np.where(masks[level,:,:]==1, da_True_U_Tend.values[time,level,:,:], np.nan), 'True East-West Velocity Increment', 
                               level, da_X.values, da_Y.values, da_Z.values, cmap='PRGn',  min_value=-0.2, max_value=0.2, extend='both')
   plt.savefig(rootdir+'/EXAMPLE_PREDICTIONS/'+model_name+'_TrueUTend_z'+str(level)+'_'+trainval+'.png', bbox_inches = 'tight', pad_inches = 0.1)
   plt.close()
   
   fig = ChnPlt.plot_depth_fld(np.where(masks[level,:,:]==1, da_True_V_Tend.values[time,level,:,:], np.nan), 'True North-South Velocity Increment',
                               level, da_X.values, da_Y.values, da_Z.values, title=None, cmap='PRGn', min_value=-0.3, max_value=0.3, extend='both')
   plt.savefig(rootdir+'/EXAMPLE_PREDICTIONS/'+model_name+'_TrueVTend_z'+str(level)+'_'+trainval+'.png', bbox_inches = 'tight', pad_inches = 0.1)
   plt.close()
   
   fig = ChnPlt.plot_depth_fld(np.where(masks[level,:,:]==1, da_True_Eta_Tend.values[time,:,:], np.nan), 'True Sea Surface Height Increment',
                               0, da_X.values, da_Y.values, da_Z.values, title=None, cmap='PRGn',  min_value=-0.08, max_value=0.08, extend='both')
   plt.savefig(rootdir+'/EXAMPLE_PREDICTIONS/'+model_name+'_TrueEtaTend_'+trainval+'.png', bbox_inches = 'tight', pad_inches = 0.1)
   plt.close()

   #-----------------------------------------
   # Plot spatial depth plots of predictions
   #-----------------------------------------
   fig = ChnPlt.plot_depth_fld(np.where(masks[level,:,:]==1, da_Pred_Temp.values[time,level,:,:], np.nan), 'Predicted Temperature',
                               level, da_X.values, da_Y.values, da_Z.values, title=None, min_value=0.0, max_value=6.3)
   plt.savefig(rootdir+'/EXAMPLE_PREDICTIONS/'+model_name+'_PredTemp_z'+str(level)+'_'+trainval+'.png', bbox_inches = 'tight', pad_inches = 0.1)
   plt.close()
   
   fig = ChnPlt.plot_depth_fld(np.where(masks[level,:,:]==1, da_Pred_U.values[time,level,:,:], np.nan), 'Predicted East-West Velocity', 
                               level, da_X.values, da_Y.values, da_Z.values, cmap='PRGn', min_value = -0.5, max_value = 0.5, extend='both')
   plt.savefig(rootdir+'/EXAMPLE_PREDICTIONS/'+model_name+'_PredU_z'+str(level)+'_'+trainval+'.png', bbox_inches = 'tight', pad_inches = 0.1)
   plt.close()
   
   fig = ChnPlt.plot_depth_fld(np.where(masks[level,:,:]==1, da_Pred_V.values[time,level,:,:], np.nan), 'Predicted North-South Velocity',
                               level, da_X.values, da_Y.values, da_Z.values, title=None, cmap='PRGn', min_value = -0.5, max_value = 0.5, extend='both')
   plt.savefig(rootdir+'/EXAMPLE_PREDICTIONS/'+model_name+'_PredV_z'+str(level)+'_'+trainval+'.png', bbox_inches = 'tight', pad_inches = 0.1)
   plt.close()
   
   fig = ChnPlt.plot_depth_fld(np.where(masks[level,:,:]==1, da_Pred_Eta.values[time,:,:], np.nan), 'Predicted Sea Surface Height',
                               0, da_X.values, da_Y.values, da_Z.values, title=None, cmap='PRGn', min_value = -0.8, max_value = 0.8)
   plt.savefig(rootdir+'/EXAMPLE_PREDICTIONS/'+model_name+'_PredEta_'+trainval+'.png', bbox_inches = 'tight', pad_inches = 0.1)
   plt.close()

   fig = ChnPlt.plot_depth_fld(np.where(masks[level,:,:]==1, da_Pred_Temp_Tend.values[time,level,:,:], np.nan), 'Predicted Temperature Increment',
                               level, da_X.values, da_Y.values, da_Z.values, title=None, cmap='PRGn')
   plt.savefig(rootdir+'/EXAMPLE_PREDICTIONS/'+model_name+'_PredTempTend_z'+str(level)+'_'+trainval+'.png', bbox_inches = 'tight', pad_inches = 0.1)
   plt.close()
   
   fig = ChnPlt.plot_depth_fld(np.where(masks[level,:,:]==1, da_Pred_U_Tend.values[time,level,:,:], np.nan), 'Predicted East-West Velocity Increment', 
                               level, da_X.values, da_Y.values, da_Z.values, cmap='PRGn')
   plt.savefig(rootdir+'/EXAMPLE_PREDICTIONS/'+model_name+'_PredUTend_z'+str(level)+'_'+trainval+'.png', bbox_inches = 'tight', pad_inches = 0.1)
   plt.close()
   
   fig = ChnPlt.plot_depth_fld(np.where(masks[level,:,:]==1, da_Pred_V_Tend.values[time,level,:,:], np.nan), 'Predicted North-South Velocity Increment',
                               level, da_X.values, da_Y.values, da_Z.values, title=None, cmap='PRGn')
   plt.savefig(rootdir+'/EXAMPLE_PREDICTIONS/'+model_name+'_PredVTend_z'+str(level)+'_'+trainval+'.png', bbox_inches = 'tight', pad_inches = 0.1)
   plt.close()
   
   fig = ChnPlt.plot_depth_fld(np.where(masks[level,:,:]==1, da_Pred_Eta_Tend.values[time,:,:], np.nan), 'Predicted Sea Surface Height Increment',
                               0, da_X.values, da_Y.values, da_Z.values, title=None, cmap='PRGn')
   plt.savefig(rootdir+'/EXAMPLE_PREDICTIONS/'+model_name+'_PredEtaTend_'+trainval+'.png', bbox_inches = 'tight', pad_inches = 0.1)
   plt.close()

   #-------------------------------------
   # Plot spatial depth difference plots
   # These plots have either 3 panes - the truth, the prediction and the diff, or two panes - the prediction and diff from truth
   #-------------------------------------
   fig = ChnPlt.plot_depth_fld_diff(np.where(masks[level,:,:]==1, da_True_Temp.values[time,level,:,:], np.nan), 'True Temperature',
                                    np.where(masks[level,:,:]==1, da_Pred_Temp.values[time,level,:,:], np.nan), 'Predicted Temperature',
                                    level, da_X.values, da_Y.values, da_Z.values,
                                    flds_max_value=-0.6, flds_min_value=0.6, diff_min_value=-0.1, diff_max_value=0.1, panes=2, extend='both')
   plt.savefig(rootdir+'/EXAMPLE_PREDICTIONS/'+model_name+'_Temp_diff_z'+str(level)+'_'+trainval+'.png', bbox_inches = 'tight', pad_inches = 0.1)
   plt.close()
   
   fig = ChnPlt.plot_depth_fld_diff(np.where(masks[level,:,:]==1, da_True_U.values[time,level,:,:], np.nan), 'True East-West Velocity', 
                                    np.where(masks[level,:,:]==1, da_Pred_U.values[time,level,:,:], np.nan), 'Predicted East-West Velocity',
                                    level, da_X.values, da_Y.values, da_Z.values, cmap='PRGn',
                                    flds_max_value=-0.2, flds_min_value=0.2, diff_min_value=-0.025, diff_max_value=0.025, panes=2, extend='both')
   plt.savefig(rootdir+'/EXAMPLE_PREDICTIONS/'+model_name+'_U_diff_z'+str(level)+'_'+trainval+'.png', bbox_inches = 'tight', pad_inches = 0.1)
   plt.close()
   
   fig = ChnPlt.plot_depth_fld_diff(np.where(masks[level,:,:]==1, da_True_V.values[time,level,:,:], np.nan), 'True North-South Velocity',
                                    np.where(masks[level,:,:]==1, da_Pred_V.values[time,level,:,:], np.nan), 'Predicted North-South Velocity',
                                    level, da_X.values, da_Y.values, da_Z.values, cmap='PRGn',
                                    flds_max_value=-0.3, flds_min_value=0.3, diff_min_value=-0.025, diff_max_value=0.025, panes=2, extend='both')
   plt.savefig(rootdir+'/EXAMPLE_PREDICTIONS/'+model_name+'_V_diff_z'+str(level)+'_'+trainval+'.png', bbox_inches = 'tight', pad_inches = 0.1)
   plt.close()
   
   fig = ChnPlt.plot_depth_fld_diff(np.where(masks[level,:,:]==1, da_True_Eta.values[time,:,:], np.nan), 'True Sea Surface Height',
                                    np.where(masks[level,:,:]==1, da_Pred_Eta.values[time,:,:], np.nan), 'Predicted Sea Surface Height',
                                    0, da_X.values, da_Y.values, da_Z.values, cmap='PRGn',
                                    flds_max_value=-0.08, flds_min_value=0.08, diff_min_value=-0.006, diff_max_value=0.006, panes=2, extend='both')
   plt.savefig(rootdir+'/EXAMPLE_PREDICTIONS/'+model_name+'_Eta_diff_'+trainval+'.png', bbox_inches = 'tight', pad_inches = 0.1)
   plt.close()

   fig = ChnPlt.plot_depth_fld_diff(np.where(masks[level,:,:]==1, da_True_Temp_Tend.values[time,level,:,:], np.nan), 'True Temperature Increment',
                                    np.where(masks[level,:,:]==1, da_Pred_Temp_Tend.values[time,level,:,:], np.nan), 'Predicted Temperature Increment',
                                    level, da_X.values, da_Y.values, da_Z.values,
                                    diff_min_value=-0.1, diff_max_value=0.1, panes=2, cmap='PRGn')
   plt.savefig(rootdir+'/EXAMPLE_PREDICTIONS/'+model_name+'_TempTend_diff_z'+str(level)+'_'+trainval+'.png', bbox_inches = 'tight', pad_inches = 0.1)
   plt.close()
   
   fig = ChnPlt.plot_depth_fld_diff(np.where(masks[level,:,:]==1, da_True_U_Tend.values[time,level,:,:], np.nan), 'True East-West Velocity Increment', 
                                    np.where(masks[level,:,:]==1, da_Pred_U_Tend.values[time,level,:,:], np.nan), 'Predicted East-West Velocity Increment',
                                    level, da_X.values, da_Y.values, da_Z.values,
                                    diff_min_value=-0.025, diff_max_value=0.025, panes=2, cmap='PRGn')
   plt.savefig(rootdir+'/EXAMPLE_PREDICTIONS/'+model_name+'_UTend_diff_z'+str(level)+'_'+trainval+'.png', bbox_inches = 'tight', pad_inches = 0.1)
   plt.close()
   
   fig = ChnPlt.plot_depth_fld_diff(np.where(masks[level,:,:]==1, da_True_V_Tend.values[time,level,:,:], np.nan), 'True North-South Velocity Increment',
                                    np.where(masks[level,:,:]==1, da_Pred_V_Tend.values[time,level,:,:], np.nan), 'Predicted North-South Velocity Increment',
                                    level, da_X.values, da_Y.values, da_Z.values,
                                    diff_min_value=-0.025, diff_max_value=0.025, panes=2, cmap='PRGn')
   plt.savefig(rootdir+'/EXAMPLE_PREDICTIONS/'+model_name+'_VTend_diff_z'+str(level)+'_'+trainval+'.png', bbox_inches = 'tight', pad_inches = 0.1)
   plt.close()
   
   fig = ChnPlt.plot_depth_fld_diff(np.where(masks[level,:,:]==1, da_True_Eta_Tend.values[time,:,:], np.nan), 'True Sea Surface Height Increment',
                                    np.where(masks[level,:,:]==1, da_Pred_Eta_Tend.values[time,:,:], np.nan), 'Predicted Sea Surface Height Increment',
                                    0, da_X.values, da_Y.values, da_Z.values,
                                    diff_min_value=-0.006, diff_max_value=0.006, panes=2, cmap='PRGn')
   plt.savefig(rootdir+'/EXAMPLE_PREDICTIONS/'+model_name+'_EtaTend_diff_'+trainval+'.png', bbox_inches = 'tight', pad_inches = 0.1)
   plt.close()

