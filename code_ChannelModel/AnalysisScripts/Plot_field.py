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
point = [ 2, 8, 6]

dir_name_prefix = 'ReflPadNS_CircPadEW_'
model_name_prefix = dir_name_prefix
model_style = 'UNet'
learning_rate = '0.0001'
epochs = '99'

#-----------

dir_name = dir_name_prefix+model_style+'_lr'+learning_rate
model_name = model_name_prefix+model_style+'_lr'+learning_rate+'_'+epochs+'epochs'

rootdir = '../../../Channel_nn_Outputs/'+dir_name

time = 1
level = point[0]
y_coord = point[1]
x_coord = point[2]

#------------------------
print('reading in data')
#------------------------
stats_data_filename=rootdir+'/STATS/'+model_name+'_StatsOutput.nc'
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

#--------------------------
# Plot spatial depth plots
#--------------------------
fig = ChnPlt.plot_depth_fld_diff(da_True_Temp.values[time,level,:,:], 'True Temperature',
                                 da_Pred_Temp.values[time,level,:,:], 'Predicted Temperature',
                                 level, da_X.values, da_Y.values, da_Z.values, title=None)
plt.savefig(rootdir+'/PLOTS/'+model_name+'_Temp_diff_z'+str(level)+'.png', bbox_inches = 'tight', pad_inches = 0.1)

fig = ChnPlt.plot_depth_fld_diff(da_True_U.values[time,level,:,:], 'True U', 
                                 da_Pred_U.values[time,level,:,:], 'Predicted U',
                                 level, da_X.values, da_Y.values, da_Z.values)
plt.savefig(rootdir+'/PLOTS/'+model_name+'_U_diff_z'+str(level)+'.png', bbox_inches = 'tight', pad_inches = 0.1)

fig = ChnPlt.plot_depth_fld_diff(da_True_V.values[time,level,:,:], 'True V',
                                 da_Pred_V.values[time,level,:,:], 'Predicted V',
                                 level, da_X.values, da_Y.values, da_Z.values, title=None)
plt.savefig(rootdir+'/PLOTS/'+model_name+'_V_diff_z'+str(level)+'.png', bbox_inches = 'tight', pad_inches = 0.1)

fig = ChnPlt.plot_depth_fld_diff(da_True_Eta.values[time,:,:], 'True Eta',
                                 da_Pred_Eta.values[time,:,:], 'Predicted Eta',
                                 0, da_X.values, da_Y.values, da_Z.values, title=None)
plt.savefig(rootdir+'/PLOTS/'+model_name+'_Eta_diff_z'+str(level)+'.png', bbox_inches = 'tight', pad_inches = 0.1)



###-----------------------
### Plot y-cross sections
###-----------------------
##fig, ax, im = ChnPlt.plot_yconst_crss_sec(da_Temp_RMS.values, 'Temperature RMS Errors', y_coord,
##                                          da_X.values, da_Y.values, da_Z.values,
##                                          title=None, min_value=None, max_value=None)
##plt.savefig(rootdir+'/PLOTS/'+model_name+'_Temp_RMS_y'+str(y_coord)+'.png', bbox_inches = 'tight', pad_inches = 0.1)
##
##fig, ax, im = ChnPlt.plot_yconst_crss_sec(da_U_RMS.values, 'U Vel RMS Errors', y_coord,
##                                          da_X.values, da_Y.values, da_Z.values,
##                                          title=None, min_value=None, max_value=None)
##plt.savefig(rootdir+'/PLOTS/'+model_name+'_U_RMS_y'+str(y_coord)+'.png', bbox_inches = 'tight', pad_inches = 0.1)
##
##fig, ax, im = ChnPlt.plot_yconst_crss_sec(da_V_RMS.values, 'V Vel RMS Errors', y_coord,
##                                          da_X.values, da_Y.values, da_Z.values,
##                                          title=None, min_value=None, max_value=None)
##plt.savefig(rootdir+'/PLOTS/'+model_name+'_V_RMS_y'+str(y_coord)+'.png', bbox_inches = 'tight', pad_inches = 0.1)
##
##
##fig, ax, im = ChnPlt.plot_yconst_crss_sec(da_Temp_CC.values, 'Temperature CC Errors', y_coord,
##                                          da_X.values, da_Y.values, da_Z.values,
##                                          title=None, min_value=None, max_value=None)
##plt.savefig(rootdir+'/PLOTS/'+model_name+'_Temp_CC_y'+str(y_coord)+'.png', bbox_inches = 'tight', pad_inches = 0.1)
##
##fig, ax, im = ChnPlt.plot_yconst_crss_sec(da_U_CC.values, 'U Vel CC Errors', y_coord,
##                                          da_X.values, da_Y.values, da_Z.values,
##                                          title=None, min_value=None, max_value=None)
##plt.savefig(rootdir+'/PLOTS/'+model_name+'_U_CC_y'+str(y_coord)+'.png', bbox_inches = 'tight', pad_inches = 0.1)
##
##fig, ax, im = ChnPlt.plot_yconst_crss_sec(da_V_CC.values, 'V Vel CC Errors', y_coord,
##                                          da_X.values, da_Y.values, da_Z.values,
##                                          title=None, min_value=None, max_value=None)
##plt.savefig(rootdir+'/PLOTS/'+model_name+'_V_CC_y'+str(y_coord)+'.png', bbox_inches = 'tight', pad_inches = 0.1)
##
###-----------------------
### Plot x-cross sections
###-----------------------
##fig, ax, im = ChnPlt.plot_xconst_crss_sec(da_Temp_RMS.values, 'Temperature RMS Errors', x_coord,
##                                          da_X.values, da_Y.values, da_Z.values,
##                                          title=None, min_value=None, max_value=None)
##plt.savefig(rootdir+'/PLOTS/'+model_name+'_Temp_RMS_x'+str(x_coord)+'.png', bbox_inches = 'tight', pad_inches = 0.1)
##
##fig, ax, im = ChnPlt.plot_xconst_crss_sec(da_U_RMS.values, 'U Vel RMS Errors', x_coord,
##                                          da_X.values, da_Y.values, da_Z.values,
##                                          title=None, min_value=None, max_value=None)
##plt.savefig(rootdir+'/PLOTS/'+model_name+'_U_RMS_x'+str(x_coord)+'.png', bbox_inches = 'tight', pad_inches = 0.1)
##
##fig, ax, im = ChnPlt.plot_xconst_crss_sec(da_V_RMS.values, 'V Vel RMS Errors', x_coord,
##                                          da_X.values, da_Y.values, da_Z.values,
##                                          title=None, min_value=None, max_value=None)
##plt.savefig(rootdir+'/PLOTS/'+model_name+'_V_RMS_x'+str(x_coord)+'.png', bbox_inches = 'tight', pad_inches = 0.1)
##
##
##fig, ax, im = ChnPlt.plot_xconst_crss_sec(da_Temp_CC.values, 'Temperature CC Errors', x_coord,
##                                          da_X.values, da_Y.values, da_Z.values,
##                                          title=None, min_value=None, max_value=None)
##plt.savefig(rootdir+'/PLOTS/'+model_name+'_Temp_CC_x'+str(x_coord)+'.png', bbox_inches = 'tight', pad_inches = 0.1)
##
##fig, ax, im = ChnPlt.plot_xconst_crss_sec(da_U_CC.values, 'U Vel CC Errors', x_coord,
##                                          da_X.values, da_Y.values, da_Z.values,
##                                          title=None, min_value=None, max_value=None)
##plt.savefig(rootdir+'/PLOTS/'+model_name+'_U_CC_x'+str(x_coord)+'.png', bbox_inches = 'tight', pad_inches = 0.1)
##
##fig, ax, im = ChnPlt.plot_xconst_crss_sec(da_V_CC.values, 'V Vel CC Errors', x_coord,
##                                          da_X.values, da_Y.values, da_Z.values,
##                                          title=None, min_value=None, max_value=None)
##plt.savefig(rootdir+'/PLOTS/'+model_name+'_V_CC_x'+str(x_coord)+'.png', bbox_inches = 'tight', pad_inches = 0.1)
##
