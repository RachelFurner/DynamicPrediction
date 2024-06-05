#!/usr/bin/env python
# coding: utf-8

# Script to create plots for analysing MITgcm run. 
# Plots means and sd from netcdf file created from AnalyseMITgcm.py, spatial fields at a set depth,
# the difference between consecutive times at a set depth, y and x cross sections, time series of 
# min and max temperatures, and a time series of the mean and max KE

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
time = 5 
point = [ 2, 50, 100]

time_step = '12hrs'

#----------------------

datadir1  = '/data/hpcdata/users/racfur/MITgcm/verification/MundayChannelConfig10km_noSpits/runs/50yr_Cntrl/'
mon_file1 = datadir1 + 'monitor.nc'
ds_mon1 = xr.open_dataset(mon_file1)
ds_mon1 = Dataset(mon_file1)

datadir2  = '/data/hpcdata/users/racfur/MITgcm/verification/MundayChannelConfig10km_noSpits/runs/10yr_increasedBF/'
mon_file2 = datadir2 + 'monitor.nc'
ds_mon2 = xr.open_dataset(mon_file2)
ds_mon2 = Dataset(mon_file2)

rootdir = '/data/hpcdata/users/racfur/DynamicPrediction/MITGCM_Analysis_Channel/12hrs/'
#-----------------------------------------
# Plot min and max temp over whole domain
#-----------------------------------------
print('plot min and max temp plots')
fig = plt.figure(figsize=(15,3))
plt.plot(ds_mon1.variables['dynstat_theta_min'][0:])
plt.plot(ds_mon2.variables['dynstat_theta_min'][0:])
plt.title('Minimum Temperature')
plt.ylabel('Temperature (degrees C)')
plt.xlabel('Time (years)')
fig.savefig(rootdir+'minT.png', bbox_inches = 'tight', pad_inches = 0.1)
plt.close()

fig = plt.figure(figsize=(15,3))
plt.plot(ds_mon1.variables['dynstat_theta_max'][0:])
plt.plot(ds_mon2.variables['dynstat_theta_max'][0:])
plt.title('Maximum Temperature')
plt.ylabel('Temperature (degrees C)')
plt.xlabel('Time (years)')
fig.savefig(rootdir+'maxT.png', bbox_inches = 'tight', pad_inches = 0.1)
plt.close()

#-------------------------------------
# Plot time series of max and mean KE
#-------------------------------------
print('plot max and mean KE plots')
fig = plt.figure(figsize=(15,3))
plt.plot(ds_mon1.variables['ke_max'][0:])
plt.plot(ds_mon2.variables['ke_max'][0:])
plt.title('Maximum KE over domain')
plt.ylabel('KE')
plt.xlabel('Time (years)')
fig.savefig(rootdir+'maxKE.png', bbox_inches = 'tight', pad_inches = 0.1)
plt.close()

fig = plt.figure(figsize=(15,3))
plt.plot(ds_mon1.variables['ke_mean'][0:])
plt.plot(ds_mon2.variables['ke_mean'][0:])
plt.title('Mean KE over domain')
plt.ylabel('KE')
plt.xlabel('Time (years)')
plt.xticks(np.arange(100))
fig.savefig(rootdir+'meanKE.png', bbox_inches = 'tight', pad_inches = 0.1)
plt.close()


