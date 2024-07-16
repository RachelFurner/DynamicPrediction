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
smooth_levels = ['20', '40', '60', '80', '100', '125', '150']
#smooth_levels = ['20']
level = 2
iteration_len = 180 
animation_end = 85   
times = [11+14, 11+28, 11+56, 11+84]

plotdir = '/data/hpcdata/users/racfur/DynamicPrediction/MITGCM_Analysis_Channel/PertPlots'
#------------------------
print('reading in data')
#------------------------
truth_filename = '/data/hpcdata/users/racfur/MITgcm/verification/MundayChannelConfig10km_noSpits/runs/50yr_Cntrl/12hrly_data.nc'
truth_ds = xr.open_dataset(truth_filename)
da_true_Temp = truth_ds['THETA']
da_X = truth_ds['X']
da_Y = truth_ds['Y'][:]
grid_filename = '/data/hpcdata/users/racfur/MITgcm/verification/MundayChannelConfig10km_noSpits/runs/50yr_Cntrl/grid.nc' 
grid_ds = xr.open_dataset(grid_filename)
HfacC = grid_ds['HFacC']
da_Z = grid_ds['RC']

masked_True_Temp = np.where( HfacC.values>0, da_true_Temp.values, np.nan )

for smooth in smooth_levels: 

   print(smooth)
   pert_filename ='/data/hpcdata/users/racfur/MITgcm/verification/MundayChannelConfig10km_noSpits/runs/50yr_smooth'+smooth+'/12hrly_data.nc'
   pert_ds = xr.open_dataset(pert_filename)
   da_pert_Temp = pert_ds['THETA']
   masked_Pert_Temp = np.where( HfacC>0, da_pert_Temp.values, np.nan )

   for time in times:

      print(time)
      #-----------------------
      # Plot pertubed fields
      #-----------------------
   
      fig = ChnPlt.plot_depth_fld(masked_Pert_Temp[time,level,:,:], 'Temperature',
                                  level, da_X.values, da_Y.values, da_Z.values, title=None,
                                  minmax=[0,6.5], extend='max')
      plt.savefig(plotdir+'/pertfields_smooth'+smooth+'Temp_level'+str(level)+'_time'+f'{time:03}'+'.png',
                  bbox_inches = 'tight', pad_inches = 0.1)
      plt.close()
      
      #------------------
      # Plot true fields
      #------------------
   
      fig = ChnPlt.plot_depth_fld(masked_True_Temp[time,level,:,:], 'Temperature',
                                  level, da_X.values, da_Y.values, da_Z.values, title=None,
                                  minmax=[0.,6.5], extend='max')
                                  #min_value=0., max_value=0.04, extend='max')
      plt.savefig(plotdir+'/truefields_smooth'+smooth+'Temp_level'+str(level)+'_time'+f'{time:03}'+'.png',
                  bbox_inches = 'tight', pad_inches = 0.1)
      plt.close()
      
      #------------------
      # Plot diff fields
      #------------------
   
      fig = ChnPlt.plot_depth_fld(masked_Pert_Temp[time,level,:,:] - masked_True_Temp[time,level,:,:], 'Temperature',
                                  level, da_X.values, da_Y.values, da_Z.values, title=None,
                                  minmax=[-1,1], extend='both', cmap='bwr')
      plt.savefig(plotdir+'/difffields_smooth'+smooth+'Temp_level'+str(level)+'_time'+f'{time:03}'+'.png',
                  bbox_inches = 'tight', pad_inches = 0.1)
      plt.close()
      
   
