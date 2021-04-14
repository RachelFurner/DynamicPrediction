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

dir_name_prefix = 'ConsPadNS_CircPadEW_'
model_name_prefix = dir_name_prefix
model_style = 'UNet'
learning_rate = '0.0001'
epochs = '99'

#-----------

dir_name = dir_name_prefix+model_style+'_lr'+learning_rate
model_name = model_name_prefix+model_style+'_lr'+learning_rate+'_'+epochs+'epochs'

rootdir = '../../../Channel_nn_Outputs/'+dir_name

level = point[0]
y_coord = point[1]
x_coord = point[2]

#------------------------
print('reading in data')
#------------------------
if plot_stats:
   stats_data_filename=rootdir+'/STATS/'+model_name+'_StatsOutput.nc'
   stats_ds = xr.open_dataset(stats_data_filename)
   da_Temp_RMS = stats_ds['TempRMS']
   da_U_RMS    = stats_ds['U_RMS']
   da_V_RMS    = stats_ds['V_RMS']
   da_Eta_RMS  = stats_ds['EtaRMS']
   da_Temp_CC  = stats_ds['TempCC']
   da_U_CC     = stats_ds['U_CC']
   da_V_CC     = stats_ds['V_CC']
   da_Eta_CC   = stats_ds['EtaCC']
   da_X = stats_ds['X']
   da_Y = stats_ds['Y']
   da_Z = stats_ds['Z']

#--------------------------
# Plot spatial depth plots
#--------------------------
fig, ax, im = ChnPlt.plot_depth_fld(da_Temp_RMS.values[level,:,:], 'Temperature RMS Errors', level,
                                   da_X.values, da_Y.values, da_Z.values,
                                   title=None, min_value=None, max_value=None)
plt.savefig(rootdir+'/PLOTS/'+model_name+'_Temp_RMS_z'+str(level)+'.png', bbox_inches = 'tight', pad_inches = 0.1)

fig, ax, im = ChnPlt.plot_depth_fld(da_U_RMS.values[level,:,:], 'U Vel RMS Errors', level,
                                   da_X.values, da_Y.values, da_Z.values,
                                   title=None, min_value=None, max_value=None)
plt.savefig(rootdir+'/PLOTS/'+model_name+'_U_RMS_z'+str(level)+'.png', bbox_inches = 'tight', pad_inches = 0.1)

fig, ax, im = ChnPlt.plot_depth_fld(da_V_RMS.values[level,:,:], 'V Vel RMS Errors', level,
                                   da_X.values, da_Y.values, da_Z.values,
                                   title=None, min_value=None, max_value=None)
plt.savefig(rootdir+'/PLOTS/'+model_name+'_V_RMS_z'+str(level)+'.png', bbox_inches = 'tight', pad_inches = 0.1)

fig, ax, im = ChnPlt.plot_depth_fld(da_Eta_RMS.values[:,:], 'Eta RMS Errors', 0,
                                   da_X.values, da_Y.values, da_Z.values,
                                   title=None, min_value=None, max_value=None)
plt.savefig(rootdir+'/PLOTS/'+model_name+'_Eta_RMS.png', bbox_inches = 'tight', pad_inches = 0.1)


fig, ax, im = ChnPlt.plot_depth_fld(da_Temp_CC.values[level,:,:], 'Temperature CC Errors', level,
                                   da_X.values, da_Y.values, da_Z.values,
                                   title=None, min_value=None, max_value=None)
plt.savefig(rootdir+'/PLOTS/'+model_name+'_Temp_CC_z'+str(level)+'.png', bbox_inches = 'tight', pad_inches = 0.1)

fig, ax, im = ChnPlt.plot_depth_fld(da_U_CC.values[level,:,:], 'U Vel CC Errors', level,
                                   da_X.values, da_Y.values, da_Z.values,
                                   title=None, min_value=None, max_value=None)
plt.savefig(rootdir+'/PLOTS/'+model_name+'_U_CC_z'+str(level)+'.png', bbox_inches = 'tight', pad_inches = 0.1)

fig, ax, im = ChnPlt.plot_depth_fld(da_V_CC.values[level,:,:], 'V Vel CC Errors', level,
                                   da_X.values, da_Y.values, da_Z.values,
                                   title=None, min_value=None, max_value=None)
plt.savefig(rootdir+'/PLOTS/'+model_name+'_V_CC_z'+str(level)+'.png', bbox_inches = 'tight', pad_inches = 0.1)

fig, ax, im = ChnPlt.plot_depth_fld(da_Eta_CC.values[:,:], 'Eta CC Errors', 0,
                                   da_X.values, da_Y.values, da_Z.values,
                                   title=None, min_value=None, max_value=None)
plt.savefig(rootdir+'/PLOTS/'+model_name+'_Eta_CC.png', bbox_inches = 'tight', pad_inches = 0.1)


#-----------------------
# Plot y-cross sections
#-----------------------
fig, ax, im = ChnPlt.plot_yconst_crss_sec(da_Temp_RMS.values, 'Temperature RMS Errors', y_coord,
                                          da_X.values, da_Y.values, da_Z.values,
                                          title=None, min_value=None, max_value=None)
plt.savefig(rootdir+'/PLOTS/'+model_name+'_Temp_RMS_y'+str(y_coord)+'.png', bbox_inches = 'tight', pad_inches = 0.1)

fig, ax, im = ChnPlt.plot_yconst_crss_sec(da_U_RMS.values, 'U Vel RMS Errors', y_coord,
                                          da_X.values, da_Y.values, da_Z.values,
                                          title=None, min_value=None, max_value=None)
plt.savefig(rootdir+'/PLOTS/'+model_name+'_U_RMS_y'+str(y_coord)+'.png', bbox_inches = 'tight', pad_inches = 0.1)

fig, ax, im = ChnPlt.plot_yconst_crss_sec(da_V_RMS.values, 'V Vel RMS Errors', y_coord,
                                          da_X.values, da_Y.values, da_Z.values,
                                          title=None, min_value=None, max_value=None)
plt.savefig(rootdir+'/PLOTS/'+model_name+'_V_RMS_y'+str(y_coord)+'.png', bbox_inches = 'tight', pad_inches = 0.1)


fig, ax, im = ChnPlt.plot_yconst_crss_sec(da_Temp_CC.values, 'Temperature CC Errors', y_coord,
                                          da_X.values, da_Y.values, da_Z.values,
                                          title=None, min_value=None, max_value=None)
plt.savefig(rootdir+'/PLOTS/'+model_name+'_Temp_CC_y'+str(y_coord)+'.png', bbox_inches = 'tight', pad_inches = 0.1)

fig, ax, im = ChnPlt.plot_yconst_crss_sec(da_U_CC.values, 'U Vel CC Errors', y_coord,
                                          da_X.values, da_Y.values, da_Z.values,
                                          title=None, min_value=None, max_value=None)
plt.savefig(rootdir+'/PLOTS/'+model_name+'_U_CC_y'+str(y_coord)+'.png', bbox_inches = 'tight', pad_inches = 0.1)

fig, ax, im = ChnPlt.plot_yconst_crss_sec(da_V_CC.values, 'V Vel CC Errors', y_coord,
                                          da_X.values, da_Y.values, da_Z.values,
                                          title=None, min_value=None, max_value=None)
plt.savefig(rootdir+'/PLOTS/'+model_name+'_V_CC_y'+str(y_coord)+'.png', bbox_inches = 'tight', pad_inches = 0.1)

#-----------------------
# Plot x-cross sections
#-----------------------
fig, ax, im = ChnPlt.plot_xconst_crss_sec(da_Temp_RMS.values, 'Temperature RMS Errors', x_coord,
                                          da_X.values, da_Y.values, da_Z.values,
                                          title=None, min_value=None, max_value=None)
plt.savefig(rootdir+'/PLOTS/'+model_name+'_Temp_RMS_x'+str(x_coord)+'.png', bbox_inches = 'tight', pad_inches = 0.1)

fig, ax, im = ChnPlt.plot_xconst_crss_sec(da_U_RMS.values, 'U Vel RMS Errors', x_coord,
                                          da_X.values, da_Y.values, da_Z.values,
                                          title=None, min_value=None, max_value=None)
plt.savefig(rootdir+'/PLOTS/'+model_name+'_U_RMS_x'+str(x_coord)+'.png', bbox_inches = 'tight', pad_inches = 0.1)

fig, ax, im = ChnPlt.plot_xconst_crss_sec(da_V_RMS.values, 'V Vel RMS Errors', x_coord,
                                          da_X.values, da_Y.values, da_Z.values,
                                          title=None, min_value=None, max_value=None)
plt.savefig(rootdir+'/PLOTS/'+model_name+'_V_RMS_x'+str(x_coord)+'.png', bbox_inches = 'tight', pad_inches = 0.1)


fig, ax, im = ChnPlt.plot_xconst_crss_sec(da_Temp_CC.values, 'Temperature CC Errors', x_coord,
                                          da_X.values, da_Y.values, da_Z.values,
                                          title=None, min_value=None, max_value=None)
plt.savefig(rootdir+'/PLOTS/'+model_name+'_Temp_CC_x'+str(x_coord)+'.png', bbox_inches = 'tight', pad_inches = 0.1)

fig, ax, im = ChnPlt.plot_xconst_crss_sec(da_U_CC.values, 'U Vel CC Errors', x_coord,
                                          da_X.values, da_Y.values, da_Z.values,
                                          title=None, min_value=None, max_value=None)
plt.savefig(rootdir+'/PLOTS/'+model_name+'_U_CC_x'+str(x_coord)+'.png', bbox_inches = 'tight', pad_inches = 0.1)

fig, ax, im = ChnPlt.plot_xconst_crss_sec(da_V_CC.values, 'V Vel CC Errors', x_coord,
                                          da_X.values, da_Y.values, da_Z.values,
                                          title=None, min_value=None, max_value=None)
plt.savefig(rootdir+'/PLOTS/'+model_name+'_V_CC_x'+str(x_coord)+'.png', bbox_inches = 'tight', pad_inches = 0.1)

