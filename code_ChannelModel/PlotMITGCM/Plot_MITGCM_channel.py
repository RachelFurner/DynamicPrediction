#!/usr/bin/env python
# coding: utf-8

print('import packages')
import sys
sys.path.append('../Tools')
import CreateDataName as cn
import Channel_Model_Plotting as ChnlPlt

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from skimage.util import view_as_windows
import os
import xarray as xr
import pickle
from netCDF4 import Dataset

#------------------------
# Set plotting variables
#------------------------
# Note shape is 38, 108, 240 (z,y,x) and 36000 time steps (100 years)
time = 50 
point = [ 2, 50, 100]

time_step = '24hrs'

#----------------------

datadir  = '/data/hpcdata/users/racfur/MITgcm/verification/MundayChannelConfig10km_SmallDomain/runs/100yrs/'
data_filename=datadir + 'daily_ave_50yrs.nc'
grid_filename=datadir + 'grid.nc'
mon_file = datadir + 'monitor.nc'
rootdir = '../../../MITGCM_Analysis_Channel/'

level = point[0]
y_coord = point[1]
x_coord = point[2]

#-----------------------------------------------
# Read in netcdf file and get shape
#-----------------------------------------------
print('reading in ds')
ds = xr.open_dataset(data_filename)
# ignore last 8 y -points as these are land
da_T = ds['THETA'][:,:,:-8,:]
da_W = ds['WVEL'][:,:,:-8,:]
da_X = ds['X']
da_Y = ds['Y'][:-8]
ds_grid = xr.open_dataset(grid_filename)
da_Z = ds_grid['RC']

z_size = da_T.shape[1]
y_size = da_T.shape[2]
x_size = da_T.shape[3]

print('da_T.shape')
print(da_T.shape)

#----------------------
# Read in monitor file
#----------------------
ds_mon=xr.open_dataset(mon_file)
nc_mon = Dataset(mon_file)

#--------------------------
# Plot spatial depth plots
#--------------------------
print('plot spatial depth plots')
fig, ax, im = ChnlPlt.plot_depth_fld(da_T[time,:,:,:], 'Temperature from Model', level,
                                   da_X.values, da_Y.values, da_Z.values,
                                   title=None, min_value=None, max_value=None)
plt.savefig(rootdir+'PLOTS/'+time_step+'_Temperature_z'+str(level), bbox_inches = 'tight', pad_inches = 0.1)

#-----------------------
# Plot y-cross sections
#-----------------------
print('plot y cross section plots')
fig, ax, im = ChnlPlt.plot_yconst_crss_sec(da_T[time,:,:,:], 'Temperature', y_coord,
                                         da_X.values, da_Y.values, da_Z.values,
                                         title=None, min_value=None, max_value=None)
plt.savefig(rootdir+'PLOTS/'+time_step+'_Temperature_y'+str(y_coord), bbox_inches = 'tight', pad_inches = 0.1)

#-----------------------
# Plot x-cross sections
#-----------------------
print('plot x cross section plots')
fig, ax, im = ChnlPlt.plot_xconst_crss_sec(da_T[time,:,:,:], 'Temperature', x_coord,
                                         da_X.values, da_Y.values, da_Z.values,
                                         title=None, min_value=None, max_value=None)
plt.savefig(rootdir+'PLOTS/'+time_step+'_Temperature_x'+str(x_coord), bbox_inches = 'tight', pad_inches = 0.1)

fig, ax, im = ChnlPlt.plot_xconst_crss_sec(da_W[time,:,:,:], 'Vertical Velocity', x_coord,
                                         da_X.values, da_Y.values, da_Z.values,
                                         title=None, min_value=-0.00003, max_value=0.00003, cmap='bwr')
plt.savefig(rootdir+'PLOTS/'+time_step+'_Wvel_x'+str(x_coord), bbox_inches = 'tight', pad_inches = 0.1)

#----------------------------------
# Plot diffs between time t and t+1
#----------------------------------
print('plot diffs between t and t+1')
fig = ChnlPlt.plot_depth_fld_diff(da_T[time,:,:,:], 'Temp at time t', da_T[time+1,:,:,:], 'Temp at time t+1', level,
                                da_X.values, da_Y.values, da_Z.values,
                                title=None)
plt.savefig(rootdir+'PLOTS/'+time_step+'_Temperature_DiffInTime_z'+str(level), bbox_inches = 'tight', pad_inches = 0.1)

fig = ChnlPlt.plot_yconst_crss_sec_diff(da_T[time,:,:,:], 'Temp at time t', da_T[time+1,:,:,:], 'Temp at time t+1', y_coord,
                                      da_X.values, da_Y.values, da_Z.values,
                                      title=None)
plt.savefig(rootdir+'PLOTS/'+time_step+'_Temperature_DiffInTime_y'+str(y_coord), bbox_inches = 'tight', pad_inches = 0.1)

fig = ChnlPlt.plot_xconst_crss_sec_diff(da_T[time,:,:,:], 'Temp at time t', da_T[time+1,:,:,:], 'Temp at time t+1', x_coord,
                                      da_X.values, da_Y.values, da_Z.values,
                                      title=None)
plt.savefig(rootdir+'PLOTS/'+time_step+'_Temperature_DiffInTime_x'+str(x_coord), bbox_inches = 'tight', pad_inches = 0.1)

#-----------------------------------------
# Plot min and max temp over whole domain
#-----------------------------------------
print('plot min and max temp plots')
fig = plt.figure(figsize=(15,3))
plt.plot(nc_mon.variables['dynstat_theta_min'][0:])
plt.title('Minimum Temperature')
plt.ylabel('Temperature (degrees C)')
plt.xlabel('Time (years)')
fig.savefig(rootdir+'PLOTS/'+time_step+'_minT.png', bbox_inches = 'tight', pad_inches = 0.1)
plt.close()

fig = plt.figure(figsize=(15,3))
plt.plot(nc_mon.variables['dynstat_theta_max'][0:])
plt.title('Maximum Temperature')
plt.ylabel('Temperature (degrees C)')
plt.xlabel('Time (years)')
fig.savefig(rootdir+'PLOTS/'+time_step+'_maxT.png', bbox_inches = 'tight', pad_inches = 0.1)
plt.close()

#-------------------------------------
# Plot time series of max and mean KE
#-------------------------------------
print('plot max and mean KE plots')
fig = plt.figure(figsize=(15,3))
plt.plot(nc_mon.variables['ke_max'][0:])
plt.title('Maximum KE over domain')
plt.ylabel('KE')
plt.xlabel('Time (years)')
fig.savefig(rootdir+'PLOTS/'+time_step+'_maxKE.png', bbox_inches = 'tight', pad_inches = 0.1)
plt.close()

fig = plt.figure(figsize=(15,3))
plt.plot(nc_mon.variables['ke_mean'][0:])
plt.title('Mean KE over domain')
plt.ylabel('KE')
plt.xlabel('Time (years)')
fig.savefig(rootdir+'PLOTS/'+time_step+'_meanKE.png', bbox_inches = 'tight', pad_inches = 0.1)
plt.close()

##---------------------------------------------------------------------------
## Plot timeseries predictions
##---------------------------------------------------------------------------
#print('plot timeseries')
#fig = ChnlPlt.plt_timeseries(point, 360, {'MITGCM':da_T.values}, time_step=time_step)
#plt.savefig(rootdir+'PLOTS/'+time_step+'_timeseries_1yr_z'+str(point[0])+'y'+str(point[1])+'x'+str(point[2]), bbox_inches = 'tight', pad_inches = 0.1)
#fig = ChnlPlt.plt_2_timeseries(point, 360, 18000, {'MITGCM':da_T.values}, time_step=time_step)
#plt.savefig(rootdir+'PLOTS/'+time_step+'_timeseries_50yrs_z'+str(point[0])+'y'+str(point[1])+'x'+str(point[2]), bbox_inches = 'tight', pad_inches = 0.1)

