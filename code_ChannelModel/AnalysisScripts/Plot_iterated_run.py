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

dir_name = 'Spits_UNet2dtransp_histlen1_seed30475'
epochs = '200'
iteration_len = 180

plot_timeseries = True 
make_animation_plots = True 
animation_end = 1*30
ts_end = 1*30

#-----------

model_name = dir_name+'_'+epochs+'epochs'

rootdir = '../../../Channel_nn_Outputs/'+dir_name

level = point[0]
y_coord = point[1]
x_coord = point[2]

#------------------------
print('reading in data')
#------------------------
iter_data_filename=rootdir+'/ITERATED_FORECAST/'+model_name+'_Forecast'+str(iteration_len)+'.nc'
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
da_Temp_mask   = iter_ds['Temp_Mask']
da_U_mask      = iter_ds['U_Mask']
da_V_mask      = iter_ds['V_Mask']
da_Eta_mask    = iter_ds['Eta_Mask']
da_X = iter_ds['X']
da_Y = iter_ds['Y']
da_Z = iter_ds['Z']

masked_True_Temp = np.where( da_Temp_mask.values==0, np.nan, da_true_Temp.values )
masked_Pred_Temp = np.where( da_Temp_mask.values==0, np.nan, da_pred_Temp.values )
masked_True_U    = np.where( da_U_mask.values==0, np.nan, da_true_U.values )
masked_Pred_U    = np.where( da_U_mask.values==0, np.nan, da_pred_U.values )
masked_True_V    = np.where( da_V_mask.values==0, np.nan, da_true_V.values )
masked_Pred_V    = np.where( da_V_mask.values==0, np.nan, da_pred_V.values )
masked_True_Eta  = np.where( da_Eta_mask.values==0, np.nan, da_true_Eta.values )
masked_Pred_Eta  = np.where( da_Eta_mask.values==0, np.nan, da_pred_Eta.values )

#----------------------------
# Plot timeseries at a point
#----------------------------
if plot_timeseries:
   fig = ChnPlt.plt_timeseries( point, ts_end, 
                                {'True Temp':masked_True_Temp[:ts_end], 'Pred Temp':masked_Pred_Temp[:ts_end] },
                                y_label='Temperature ('+u'\xb0'+'C)' )
   plt.savefig(rootdir+'/ITERATED_FORECAST/'+model_name+'_Temp_timeseries_z'+str(point[0])+'y'+str(point[1])+'x'+str(point[2])+
               '_'+str(ts_end)+'.png', bbox_inches = 'tight', pad_inches = 0.1, min_value=3., max_value=4.)

   fig = ChnPlt.plt_timeseries( point, ts_end, 
                                {'True U':masked_True_U[:ts_end], 'Pred U':masked_Pred_U[:ts_end] },
                                y_label='U Vel $(m s^{-1})$' )
   plt.savefig(rootdir+'/ITERATED_FORECAST/'+model_name+'_U_timeseries_z'+str(point[0])+'y'+str(point[1])+'x'+str(point[2])+
               '_'+str(ts_end)+'.png', bbox_inches = 'tight', pad_inches = 0.1, min_value=0., max_value=0.5 )

   fig = ChnPlt.plt_timeseries( point, ts_end, 
                                {'True V':masked_True_V[:ts_end], 'Pred V':masked_Pred_V[:ts_end] },
                                y_label='V Vel $(m s^{-1})$')
   plt.savefig(rootdir+'/ITERATED_FORECAST/'+model_name+'_V_timeseries_z'+str(point[0])+'y'+str(point[1])+'x'+str(point[2])+
               '_'+str(ts_end)+'.png', bbox_inches = 'tight', pad_inches = 0.1, min_value=-0.5, max_value=0.5)

   fig = ChnPlt.plt_timeseries( point[1:], ts_end, 
                                {'True Eta':masked_True_Eta[:ts_end], 'Pred Eta':masked_Pred_Eta[:ts_end] },
                                y_label='SSH $(m)$')
   plt.savefig(rootdir+'/ITERATED_FORECAST/'+model_name+'_Eta_timeseries_z'+str(point[0])+'y'+str(point[1])+'x'+str(point[2])+
               '_'+str(ts_end)+'.png', bbox_inches = 'tight', pad_inches = 0.1, min_value=-0.3, max_value=0.3)

#-------------------------------
# Plot spatial plots to animate
#-------------------------------
if make_animation_plots:
   for time in range(0,animation_end):
      fig = ChnPlt.plot_depth_fld_diff(masked_True_Temp[time,level,:,:], 'True Temperature',
                                       masked_Pred_Temp[time,level,:,:], 'Predicted Temperature',
                                       level, da_X.values, da_Y.values, da_Z.values, title=None,
                                       flds_min_value=0., flds_max_value=6.5, diff_min_value=-1, diff_max_value=1 )
      plt.savefig(rootdir+'/ITERATED_FORECAST/PLOTS/'+model_name+'_Temp_level'+str(level)+'_time'+f'{time:03}'+'.png',
                  bbox_inches = 'tight', pad_inches = 0.1)
      plt.close()
      
      fig = ChnPlt.plot_depth_fld_diff(masked_True_U[time,level,:,:], 'True U',
                                       masked_Pred_U[time,level,:,:], 'Predicted U',
                                       level, da_X.values, da_Y.values, da_Z.values,
                                       flds_min_value=-1.2, flds_max_value=1.2, diff_min_value=-1, diff_max_value=1 )
      plt.savefig(rootdir+'/ITERATED_FORECAST/PLOTS/'+model_name+'_U_level'+str(level)+'_time'+f'{time:03}'+'.png',
                  bbox_inches = 'tight', pad_inches = 0.1)
      plt.close()
      
      fig = ChnPlt.plot_depth_fld_diff(masked_True_V[time,level,:,:], 'True V',
                                       masked_Pred_V[time,level,:,:], 'Predicted V',
                                       level, da_X.values, da_Y.values, da_Z.values, title=None,
                                       flds_min_value=-1.2, flds_max_value=1.2, diff_min_value=-1, diff_max_value=1 )
      plt.savefig(rootdir+'/ITERATED_FORECAST/PLOTS/'+model_name+'_V_level'+str(level)+'_time'+f'{time:03}'+'.png',
                  bbox_inches = 'tight', pad_inches = 0.1)
      plt.close()
      
      fig = ChnPlt.plot_depth_fld_diff(masked_True_Eta[time,:,:], 'True Eta',
                                       masked_Pred_Eta[time,:,:], 'Predicted Eta',
                                       0, da_X.values, da_Y.values, da_Z.values, title=None,
                                       flds_min_value=-1, flds_max_value=1, diff_min_value=-1, diff_max_value=1 )
      plt.savefig(rootdir+'/ITERATED_FORECAST/PLOTS/'+model_name+'_Eta_level'+str(level)+'_time'+f'{time:03}'+'.png', 
                  bbox_inches = 'tight', pad_inches = 0.1)
      plt.close()

