#!/usr/bin/env python
# coding: utf-8

print('import packages')
import sys
sys.path.append('../Tools')
import Channel_Model_Plotting as ChnlPlt

import numpy as np
import matplotlib.pyplot as plt
import os
import xarray as xr
import pickle
from netCDF4 import Dataset

#------------------------
# Set plotting variables
#------------------------
# Note shape is 38, 108, 240 (z,y,x) and 36000 time stpng (100 years)
point = [ 2, 50, 100]

datadir  = '/data/hpcdata/users/racfur/MITgcm/verification/MundayChannelConfig10km_noSpits/runs/'
restart_times = ['2592072', '2592144', '2592216', '2592288', '2592360', '2592504', '2592720', '2593080', '2593440']
timestep = [1, 2, 3, 4, 5, 7, 10, 15, 20]

Cntrl_filename = '/data/hpcdata/users/racfur/MITgcm/verification/MundayChannelConfig10km_noSpits/runs/50yr_Cntrl/1yr_dataset.nc'
Cntrl_ds = xr.open_dataset(Cntrl_filename)
Cntrl_da_T = Cntrl_ds['THETA'][:,:,:,:]
grid_filename = '/data/hpcdata/users/racfur/MITgcm/verification/MundayChannelConfig10km_noSpits/runs/50yr_Cntrl/grid.nc'
grid_ds = xr.open_dataset(grid_filename)
HfacC = grid_ds['HFacC'].values
HfacW = grid_ds['HFacW'].values
HfacS = grid_ds['HFacS'].values
Cntrl_Masked_Temp = np.where( HfacC > 0., Cntrl_da_T, 0 )
#------------------------------------
# Read in netcdf files and get shape
#------------------------------------
fig1, axs1 = plt.subplots(9, 1, figsize=(15, 12))
fig2, axs2, = plt.subplots(9, 1, figsize=(15, 12))
fig3, axs3, = plt.subplots(9, 1, figsize=(15, 12))
for i, ax1 in enumerate(axs1.flatten()):
   data_filename=datadir+'6mnthPerturbed'+restart_times[i]+'/12hrly_data.nc'
   print(data_filename)
   ds = xr.open_dataset(data_filename)
   da_T = ds['THETA'][:,:,:,:]
   da_X = ds['X']
   da_Y = ds['Y'][:]
   Masked_Temp = np.where( HfacC > 0., da_T, 0 )

   ax1.plot(Masked_Temp[:180,2,50,100], color='black')
   ax1.plot(Cntrl_Masked_Temp[timestep[i]:timestep[i]+180,2,50,100], color='blue')

   Temp_RMS = np.sqrt(np.nanmean( np.square(Masked_Temp[:180,:,:,:]-Cntrl_Masked_Temp[timestep[i]:timestep[i]+180,:,:,:]), axis=(1,2,3) ))
   ax2.plot(Temp_RMS, color='black')

   Temp_CC = np.zeros((180))
   for time in range(180):
      Temp_CC[time] = np.corrcoef( np.reshape(Masked_Temp[time,:,:,:][~np.isnan(Masked_Temp[time,:,:,:])],(-1)),
                                   np.reshape(Cntrl_Masked_Temp[timestep[i]+time,:,:,:][~np.isnan(Cntrl_Masked_Temp[timestep[i]+time,:,:,:])],(-1)) )[0,1]
   ax3.plot(Temp_CC, color='red')

fig1.savefig('Perturbed_timseries.png', bbox_inches = 'tight')
fig2.savefig('Perturbed_RMS.png', bbox_inches = 'tight')
fig3.savefig('Perturbed_CC.png', bbox_inches = 'tight')

