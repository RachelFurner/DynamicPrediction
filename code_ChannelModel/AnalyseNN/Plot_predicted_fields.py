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
iteration_len = 60  

animation_end = 60   

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

#-------------------------------
# Plot spatial plots to animate
#-------------------------------
#for time in range(0,animation_end):
for time in range(50,51):

   fig = ChnPlt.plot_depth_fld(masked_Pred_Temp[time,level,:,:], 'Temperature',
                               level, da_X.values, da_Y.values, da_Z.values, title=None,
                               min_value=0., max_value=6.5)
   plt.savefig(rootdir+'/ITERATED_FORECAST/PLOTS/predfields_'+model_name+'_Temp_level'+str(level)+'_time'+f'{time:03}'+'.eps',
               bbox_inches = 'tight', pad_inches = 0.1)
   plt.close()
   
   fig = ChnPlt.plot_depth_fld(masked_Pred_U[time,level,:,:], 'East-West Velocity',
                               level, da_X.values, da_Y.values, da_Z.values,
                               min_value=-1.2, max_value=1.2)
   plt.savefig(rootdir+'/ITERATED_FORECAST/PLOTS/predfields_'+model_name+'_U_level'+str(level)+'_time'+f'{time:03}'+'.eps',
               bbox_inches = 'tight', pad_inches = 0.1)
   plt.close()
   
   fig = ChnPlt.plot_depth_fld(masked_Pred_V[time,level,:,:], 'North-South Velocity',
                               level, da_X.values, da_Y.values, da_Z.values, title=None,
                               min_value=-1.2, max_value=1.2)
   plt.savefig(rootdir+'/ITERATED_FORECAST/PLOTS/predfields_'+model_name+'_V_level'+str(level)+'_time'+f'{time:03}'+'.eps',
               bbox_inches = 'tight', pad_inches = 0.1)
   plt.close()
   
   fig = ChnPlt.plot_depth_fld(masked_Pred_Eta[time,:,:], 'Sea Surface Height',
                               0, da_X.values, da_Y.values, da_Z.values, title=None,
                               min_value=-1, max_value=1)
   plt.savefig(rootdir+'/ITERATED_FORECAST/PLOTS/predfields_'+model_name+'_Eta_level'+str(level)+'_time'+f'{time:03}'+'.eps', 
               bbox_inches = 'tight', pad_inches = 0.1)
   plt.close()

