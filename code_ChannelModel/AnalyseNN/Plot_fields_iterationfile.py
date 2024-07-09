#!/usr/bin/env python
# coding: utf-8

# Script to plot the predicted fields, and true fields from the netcdf iteration file.
# Reads in data and masks, and plots spatial plots for a series of time points at the set depth level. These
# can be catted into sets of variables for each time stamp, and animated to give a video of temporal evolution
# using the cat_animate_iterations.sh script.
# This script differs in purpose to Analyse_Plot_iterated_run.py. This script plots only the predicted fields
# and then the true fields, as individual.png files, whereas Analyse_Plot_iterated_run.py plots the predicted, 
# true and difference fields together in a single.png, as well as timeseries and calculates some stats

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
level = 2
dir_name = 'IncLand12hrly_UNet2dtransp_histlen1_rolllen1_seed30475'
#dir_name = 'MultiModel_average_IncLand12hrly_UNet2dtransp_histlen1_rolllen1'
epochs = '200'
iteration_len = 180  
animation_end = 85   
model_name = dir_name+'_'+epochs+'epochs'
filename = model_name+'_simple_smth20stps0'

#-----------
rootdir = '../../../Channel_nn_Outputs/'+dir_name

#------------------------
print('reading in data')
#------------------------
iter_data_filename=rootdir+'/ITERATED_FORECAST/'+filename+'_Forlen'+str(iteration_len)+'.nc'
iter_ds = xr.open_dataset(iter_data_filename)
da_true_Temp = iter_ds['True_Temp']
da_true_U    = iter_ds['True_U']
da_true_V    = iter_ds['True_V']
da_true_Eta  = iter_ds['True_Eta']
da_pred_Temp = iter_ds['Pred_Temp']
da_pred_U    = iter_ds['Pred_U']
da_pred_V    = iter_ds['Pred_V']
da_pred_Eta  = iter_ds['Pred_Eta']
da_Temp_mask   = iter_ds['Temp_Mask']
da_U_mask      = iter_ds['U_Mask']
da_V_mask      = iter_ds['V_Mask']
da_Eta_mask    = iter_ds['Eta_Mask']
da_X = iter_ds['X']
da_Y = iter_ds['Y']
da_Z = iter_ds['Z']

masked_True_Temp = np.where( da_Temp_mask.values==0, np.nan, da_true_Temp.values )
masked_True_U    = np.where( da_U_mask.values==0, np.nan, da_true_U.values )
masked_True_V    = np.where( da_V_mask.values==0, np.nan, da_true_V.values )
masked_True_Eta  = np.where( da_Eta_mask.values==0, np.nan, da_true_Eta.values )

masked_Pred_Temp = np.where( da_Temp_mask.values==0, np.nan, da_pred_Temp.values )
masked_Pred_U    = np.where( da_U_mask.values==0, np.nan, da_pred_U.values )
masked_Pred_V    = np.where( da_V_mask.values==0, np.nan, da_pred_V.values )
masked_Pred_Eta  = np.where( da_Eta_mask.values==0, np.nan, da_pred_Eta.values )

#-------------------------------
# Plot spatial plots to animate
#-------------------------------
for time in range(0,animation_end):

   #-----------------------
   # Plot predicted fields
   #-----------------------

   fig = ChnPlt.plot_depth_fld(masked_Pred_Temp[time,level,:,:], 'Temperature',
                               level, da_X.values, da_Y.values, da_Z.values, title=None,
                               minmax=[0,6.5], extend='max')
   plt.savefig(rootdir+'/ITERATED_FORECAST/PLOTS/predfields_'+filename+'_Temp_level'+str(level)+'_time'+f'{time:03}'+'.png',
               bbox_inches = 'tight', pad_inches = 0.1)
   plt.close()
   
   #fig = ChnPlt.plot_depth_fld(masked_Pred_U[time,level,:,:], 'Eastward Velocity',
   #                            level, da_X.values, da_Y.values, da_Z.values,
   #                            minmax=[-1.2,1.2])
   #plt.savefig(rootdir+'/ITERATED_FORECAST/PLOTS/predfields_'+filename+'_U_level'+str(level)+'_time'+f'{time:03}'+'.png',
   #            bbox_inches = 'tight', pad_inches = 0.1)
   #plt.close()
   #
   #fig = ChnPlt.plot_depth_fld(masked_Pred_V[time,level,:,:], 'Northward Velocity',
   #                            level, da_X.values, da_Y.values, da_Z.values, title=None,
   #                            min_value=-1.2, max_value=1.2)
   #plt.savefig(rootdir+'/ITERATED_FORECAST/PLOTS/predfields_'+filename+'_V_level'+str(level)+'_time'+f'{time:03}'+'.png',
   #            bbox_inches = 'tight', pad_inches = 0.1)
   #plt.close()
   #
   #fig = ChnPlt.plot_depth_fld(masked_Pred_Eta[time,:,:], 'Sea Surface Height',
   #                            0, da_X.values, da_Y.values, da_Z.values, title=None,
   #                            min_value=-1, max_value=1)
   #plt.savefig(rootdir+'/ITERATED_FORECAST/PLOTS/predfields_'+filename+'_Eta_level'+str(level)+'_time'+f'{time:03}'+'.png', 
   #            bbox_inches = 'tight', pad_inches = 0.1)
   #plt.close()

   #------------------
   # Plot true fields
   #------------------

   fig = ChnPlt.plot_depth_fld(masked_True_Temp[time,level,:,:], 'Temperature',
                               level, da_X.values, da_Y.values, da_Z.values, title=None,
                               minmax=[0.,6.5], extend='max')
                               #min_value=0., max_value=0.04, extend='max')
   plt.savefig(rootdir+'/ITERATED_FORECAST/PLOTS/truefields_'+model_name+'_Temp_level'+str(level)+'_time'+f'{time:03}'+'.png',
               bbox_inches = 'tight', pad_inches = 0.1)
   plt.close()
   
   #fig = ChnPlt.plot_depth_fld(masked_True_U[time,level,:,:], 'Eastward Velocity',
   #                            level, da_X.values, da_Y.values, da_Z.values,
   #                            min_value=-1.2, max_value=1.2, extend='both')
   #                            #min_value=-0.3, max_value=0.3, extend='both')
   #plt.savefig(rootdir+'/ITERATED_FORECAST/PLOTS/truefields_'+model_name+'_U_level'+str(level)+'_time'+f'{time:03}'+'.png',
   #            bbox_inches = 'tight', pad_inches = 0.1)
   #plt.close()
   #
   #fig = ChnPlt.plot_depth_fld(masked_True_V[time,level,:,:], 'Northward Velocity',
   #                            level, da_X.values, da_Y.values, da_Z.values, title=None,
   #                            min_value=-1.2, max_value=1.2, extend = 'both')
   #                            #min_value=-0.3, max_value=0.3, extend = 'both')
   #plt.savefig(rootdir+'/ITERATED_FORECAST/PLOTS/truefields_'+model_name+'_V_level'+str(level)+'_time'+f'{time:03}'+'.png',
   #            bbox_inches = 'tight', pad_inches = 0.1)
   #plt.close()
   #
   #fig = ChnPlt.plot_depth_fld(masked_True_Eta[time,:,:], 'Sea Surface Height',
   #                            0, da_X.values, da_Y.values, da_Z.values, title=None,
   #                            min_value=-1., max_value=1., extend = 'both')
   #plt.savefig(rootdir+'/ITERATED_FORECAST/PLOTS/truefields_'+model_name+'_Eta_level'+str(level)+'_time'+f'{time:03}'+'.png', 
   #            bbox_inches = 'tight', pad_inches = 0.1)
   #plt.close()

   #------------------
   # Plot diff fields
   #------------------

   fig = ChnPlt.plot_depth_fld(masked_Pred_Temp[time,level,:,:] - masked_True_Temp[time,level,:,:], 'Temperature',
                               level, da_X.values, da_Y.values, da_Z.values, title=None,
                               minmax=[-1,1], extend='both', cmap='bwr')
   plt.savefig(rootdir+'/ITERATED_FORECAST/PLOTS/difffields_'+filename+'_Temp_level'+str(level)+'_time'+f'{time:03}'+'.png',
               bbox_inches = 'tight', pad_inches = 0.1)
   plt.close()
   

