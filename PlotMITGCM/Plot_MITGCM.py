#!/usr/bin/env python
# coding: utf-8

print('import packages')
import sys
sys.path.append('/data/hpcdata/users/racfur/DynamicPrediction/code_git/')
from Tools import CreateDataName as cn
from Tools import Model_Plotting as rfplt

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

time = 12000
point = [5,15,4]

#----------------------
rootdir = '/data/hpcdata/users/racfur/DynamicPrediction/MITGCM_Analysis/'

level = point[0]
y_coord = point[1]
x_coord = point[2]

#-----------------------------------------------
# Read in netcdf file and get shape
#-----------------------------------------------
print('reading in ds')
datadir  = '/data/hpcdata/users/racfur/MITGCM_OUTPUT/20000yr_Windx1.00_mm_diag/'
data_filename=datadir+'cat_tave_2000yrs_SelectedVars_masked.nc'
ds = xr.open_dataset(data_filename)
da_T=ds['Ttave'][:,:,:,:]

z_size = da_T.shape[1]
y_size = da_T.shape[2]
x_size = da_T.shape[3]

print('da_T.shape')
print(da_T.shape)

#----------------------
# Read in monitor file
#----------------------
mon_file = datadir + 'monitor.0000000000.t001_5000yrs.nc'
ds_mon=xr.open_dataset(mon_file)
nc_mon = Dataset(mon_file)

#--------------------------
# Plot spatial depth plots
#--------------------------
fig, ax, im = rfplt.plot_depth_fld(da_T[time,:,:,:], 'Temperature from Model', level, title=None, min_value=None, max_value=None)
plt.savefig(rootdir+'PLOTS/Temperature_z'+str(level), bbox_inches = 'tight', pad_inches = 0.1)

#---------------------------------
#Plot diff between time t and t+1
#---------------------------------
fig = rfplt.plot_depth_fld_diff(da_T[time,:,:,:], 'Temp at time t', da_T[time+1,:,:,:], 'Temp at time t+1', level, title=None)
plt.savefig(rootdir+'PLOTS/Temperature_DiffInTime_z'+str(level), bbox_inches = 'tight', pad_inches = 0.1)

#-----------------------
# Plot y-cross sections
#-----------------------
fig, ax, im = rfplt.plot_yconst_crss_sec(da_T[time,:,:,:], 'Temperature Predictions', y_coord, title=None, min_value=None, max_value=None)
plt.savefig(rootdir+'PLOTS/Temperature_y'+str(y_coord), bbox_inches = 'tight', pad_inches = 0.1)

#-----------------------
# Plot x-cross sections
#-----------------------
fig, ax, im = rfplt.plot_xconst_crss_sec(da_T[time,:,:,:], 'Temperature Predictions', x_coord, title=None, min_value=None, max_value=None)
plt.savefig(rootdir+'PLOTS/Temperature_x'+str(x_coord), bbox_inches = 'tight', pad_inches = 0.1)

#---------------------------------------------------------------------------
# Plot timeseries predictions
#---------------------------------------------------------------------------

fig = rfplt.plt_timeseries(point, 120, {'da_T (truth)':da_T.data})
plt.savefig(rootdir+'PLOTS/timeseries_10yrs_z'+str(point[0])+'y'+str(point[1])+'x'+str(point[2]), bbox_inches = 'tight', pad_inches = 0.1)

fig = rfplt.plt_2_timeseries(point, 120, 24000, {'da_T (truth)':da_T.data})
plt.savefig(rootdir+'PLOTS/timeseries_2000yrs_z'+str(point[0])+'y'+str(point[1])+'x'+str(point[2]), bbox_inches = 'tight', pad_inches = 0.1)

#-----------------------------------------
# Plot min and max temp over whole domain
#-----------------------------------------

fig = plt.figure(figsize=(20,3))
plt.plot(nc_mon.variables['dynstat_theta_min'][0:200])
plt.title('Minimum Temperature')
plt.ylabel('Temperature (degrees C)')
plt.xlabel('Time (decades)')
plt.show()
fig.savefig(rootdir+'PLOTS/minT.png', bbox_inches = 'tight', pad_inches = 0.1)
plt.close()

fig = plt.figure(figsize=(20,3))
plt.plot(nc_mon.variables['dynstat_theta_max'][0:200])
plt.title('Maximum Temperature')
plt.ylabel('Temperature (degrees C)')
plt.xlabel('Time (decades)')
plt.show()
fig.savefig(rootdir+'PLOTS/maxT.png', bbox_inches = 'tight', pad_inches = 0.1)
plt.close()
