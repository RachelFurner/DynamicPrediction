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
point = [ 2, 48, 120]

dir_name_prefix = 'ExcLand_None_2d_'
model_name_prefix = dir_name_prefix
#model_style = 'UNet'
model_style = 'UNet_transp'
learning_rate = '0.0001'
epochs = '199'
for_len = 6*30
animation_start = 0
animation_end = 6*30

plot_timeseries = True 
make_animation_plots = True

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
iter_data_filename=rootdir+'/ITERATED_FORECAST/'+model_name+'_Forecast'+str(for_len)+'.nc'
iter_ds = xr.open_dataset(iter_data_filename)
da_true_Temp = iter_ds['True_Temp']
da_true_U    = iter_ds['True_U']
da_true_V    = iter_ds['True_V']
da_true_Eta  = iter_ds['True_Eta']
da_pred_Temp = iter_ds['Pred_Temp']
da_pred_U    = iter_ds['Pred_U']
da_pred_V    = iter_ds['Pred_V']
da_pred_Eta  = iter_ds['Pred_Eta']
da_Temp_errors = iter_ds['Temp_Errors']
da_U_errors    = iter_ds['U_Errors']
da_V_errors    = iter_ds['V_Errors']
da_Eta_errors  = iter_ds['Eta_Errors']
da_X = iter_ds['X']
da_Y = iter_ds['Y']
da_Z = iter_ds['Z']

#----------------------------
# Plot timeseries at a point
#----------------------------
if plot_timeseries:
   fig = ChnPlt.plt_timeseries( point, da_pred_Temp.shape[0], {'True Temp':da_true_Temp.values, 'Pred Temp':da_pred_Temp.values }, y_label='Temperature ('+u'\xb0'+'C)' )
   plt.savefig(rootdir+'/ITERATED_FORECAST/'+model_name+'_Temp_timeseries_z'+str(point[0])+'y'+str(point[1])+'x'+str(point[2])+
               '_'+str(for_len)+'.png', bbox_inches = 'tight', pad_inches = 0.1)

   fig = ChnPlt.plt_timeseries( point, da_pred_U.shape[0], {'True U':da_true_U.values, 'Pred U':da_pred_U.values }, y_label='U Vel $(m s^{-1})$' )
   plt.savefig(rootdir+'/ITERATED_FORECAST/'+model_name+'_U_timeseries_z'+str(point[0])+'y'+str(point[1])+'x'+str(point[2])+
               '_'+str(for_len)+'.png', bbox_inches = 'tight', pad_inches = 0.1)

   fig = ChnPlt.plt_timeseries( point, da_pred_V.shape[0], {'True V':da_true_V.values, 'Pred V':da_pred_V.values }, y_label='V Vel $(m s^{-1})$')
   plt.savefig(rootdir+'/ITERATED_FORECAST/'+model_name+'_V_timeseries_z'+str(point[0])+'y'+str(point[1])+'x'+str(point[2])+
               '_'+str(for_len)+'.png', bbox_inches = 'tight', pad_inches = 0.1)

   fig = ChnPlt.plt_timeseries( point[1:], da_pred_Eta.shape[0], {'True Eta':da_true_Eta.values, 'Pred Eta':da_pred_Eta.values }, y_label='SSH $(m)$')
   plt.savefig(rootdir+'/ITERATED_FORECAST/'+model_name+'_Eta_timeseries_z'+str(point[0])+'y'+str(point[1])+'x'+str(point[2])+
               '_'+str(for_len)+'.png', bbox_inches = 'tight', pad_inches = 0.1)

#-------------------------------
# Plot spatial plots to animate
#-------------------------------
if make_animation_plots:
   for time in range(animation_start,animation_end):
      fig = ChnPlt.plot_depth_fld_diff(da_true_Temp.values[time,level,:,:], 'True Temperature',
                                       da_pred_Temp.values[time,level,:,:], 'Predicted Temperature',
                                       level, da_X.values, da_Y.values, da_Z.values, title=None,
                                       flds_min_value=0., flds_max_value=6.5, diff_min_value=-2, diff_max_value=2 )
      plt.savefig(rootdir+'/ITERATED_FORECAST/PLOTS/'+model_name+'_Temp_level'+str(level)+'_time'+f'{time:03}'+'.png',
                  bbox_inches = 'tight', pad_inches = 0.1)
      plt.close()
      
      fig = ChnPlt.plot_depth_fld_diff(da_true_U.values[time,level,:,:], 'True U',
                                       da_pred_U.values[time,level,:,:], 'Predicted U',
                                       level, da_X.values, da_Y.values, da_Z.values,
                                       flds_min_value=-1.2, flds_max_value=1.2, diff_min_value=-1, diff_max_value=1 )
      plt.savefig(rootdir+'/ITERATED_FORECAST/PLOTS/'+model_name+'_U_level'+str(level)+'_time'+f'{time:03}'+'.png',
                  bbox_inches = 'tight', pad_inches = 0.1)
      plt.close()
      
      fig = ChnPlt.plot_depth_fld_diff(da_true_V.values[time,level,:,:], 'True V',
                                       da_pred_V.values[time,level,:,:], 'Predicted V',
                                       level, da_X.values, da_Y.values, da_Z.values, title=None,
                                       flds_min_value=-1.2, flds_max_value=1.2, diff_min_value=-1, diff_max_value=1 )
      plt.savefig(rootdir+'/ITERATED_FORECAST/PLOTS/'+model_name+'_V_level'+str(level)+'_time'+f'{time:03}'+'.png',
                  bbox_inches = 'tight', pad_inches = 0.1)
      plt.close()
      
      fig = ChnPlt.plot_depth_fld_diff(da_true_Eta.values[time,:,:], 'True Eta',
                                       da_pred_Eta.values[time,:,:], 'Predicted Eta',
                                       0, da_X.values, da_Y.values, da_Z.values, title=None,
                                       flds_min_value=-1, flds_max_value=1, diff_min_value=-1, diff_max_value=1 )
      plt.savefig(rootdir+'/ITERATED_FORECAST/PLOTS/'+model_name+'_Eta_level'+str(level)+'_time'+f'{time:03}'+'.png', 
                  bbox_inches = 'tight', pad_inches = 0.1)
      plt.close()

