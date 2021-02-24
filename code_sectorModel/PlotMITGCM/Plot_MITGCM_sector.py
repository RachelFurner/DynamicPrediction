#!/usr/bin/env python
# coding: utf-8

print('import packages')
import sys
sys.path.append('../Tools')
import CreateDataName as cn
import Model_Plotting as SecPlt

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

time = 2500
point = [ 2, 8, 6]

time_step = '24hrs'
#----------------------

if time_step == '24hrs':
   datadir  = '/data/hpcdata/users/racfur/MITgcm/verification/MundaySectorConfig_2degree/runs/100yrs/mnc_test_0002/'
   data_filename=datadir + 'cat_tave.nc'
   mon_file = datadir + 'cat_monitor.nc'
else:
   print('####################################')
   print('WARNING - IMPROPER TIME STEP GIVEN!!')
   print('####################################')

rootdir = '../../../MITGCM_Analysis_Sector/'

level = point[0]
y_coord = point[1]
x_coord = point[2]

#-----------------------------------------------
# Read in netcdf file and get shape
#-----------------------------------------------
print('reading in ds')
ds = xr.open_dataset(data_filename)
da_T_unmasked = ds['Ttave'][:,:,:,:]
da_W_unmasked = ds['wVeltave'][:,:,:,:]
da_Mask = ds['Mask']
# Apply the Mask
da_T = np.where(da_Mask==1, da_T_unmasked, np.nan)
da_W = np.where(da_Mask==1, da_W_unmasked, np.nan)
da_X = ds['X']
da_Y = ds['Y']
da_Z = ds['Z']

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
fig, ax, im = SecPlt.plot_depth_fld(da_T[time,:,:,:], 'Temperature from Model', level,
                                   da_X.values, da_Y.values, da_Z.values,
                                   title=None, min_value=None, max_value=None, cbar_label='Temperature ('+u'\xb0'+'C)')
plt.savefig(rootdir+'PLOTS/'+time_step+'_Temperature_z'+str(level), bbox_inches = 'tight', pad_inches = 0.1)

#-----------------------
# Plot y-cross sections
#-----------------------
fig, ax, im = SecPlt.plot_yconst_crss_sec(da_T[time,:,:,:], 'Temperature', y_coord,
                                         da_X.values, da_Y.values, da_Z.values,
                                         title=None, min_value=None, max_value=None, cbar_label='Temperature ('+u'\xb0'+'C)')
plt.savefig(rootdir+'PLOTS/'+time_step+'_Temperature_y'+str(y_coord), bbox_inches = 'tight', pad_inches = 0.1)

#-----------------------
# Plot x-cross sections
#-----------------------
fig, ax, im = SecPlt.plot_xconst_crss_sec(da_T[time,:,:,:], 'Temperature', x_coord,
                                         da_X.values, da_Y.values, da_Z.values,
                                         title=None, min_value=None, max_value=None, cbar_label='Temperature ('+u'\xb0'+'C)')
plt.savefig(rootdir+'PLOTS/'+time_step+'_Temperature_x'+str(x_coord), bbox_inches = 'tight', pad_inches = 0.1)

fig, ax, im = SecPlt.plot_xconst_crss_sec(da_W[time,:,:,:], 'Vertical Velocity', x_coord,
                                         da_X.values, da_Y.values, da_Z.values,
                                         title=None, min_value=-0.00003, max_value=0.00003, cmap='bwr', cbar_label='Temperature ('+u'\xb0'+'C)')
plt.savefig(rootdir+'PLOTS/'+time_step+'_Wvel_x'+str(x_coord), bbox_inches = 'tight', pad_inches = 0.1)

#----------------------------------
# Plot diffs between time t and t+1
#----------------------------------
fig = SecPlt.plot_depth_fld_diff(da_T[time,:,:,:], 'Temperature\n', da_T[time+1,:,:,:], 'Temperature\n', level,
                                 da_X.values, da_Y.values, da_Z.values, title=None,
                                 cbar_label='Temperature ('+u'\xb0'+'C)', cbar_diff_label='Temperature Change ('+u'\xb0'+'C)', Sci=True)
plt.savefig(rootdir+'PLOTS/'+time_step+'_Temperature_DiffInTime_z'+str(level), bbox_inches = 'tight', pad_inches = 0.1)

fig = SecPlt.plot_yconst_crss_sec_diff(da_T[time,:,:,:], 'Temperature', da_T[time+1,:,:,:], 'Temperature', y_coord,
                                      da_X.values, da_Y.values, da_Z.values, title=None,
                                      cbar_label='Temperature ('+u'\xb0'+'C)',  cbar_diff_label='Temperature Change ('+u'\xb0'+'C)', Sci=True)
plt.savefig(rootdir+'PLOTS/'+time_step+'_Temperature_DiffInTime_y'+str(y_coord), bbox_inches = 'tight', pad_inches = 0.1)

fig = SecPlt.plot_xconst_crss_sec_diff(da_T[time,:,:,:], 'Temperature', da_T[time+1,:,:,:], 'Temperature', x_coord,
                                      da_X.values, da_Y.values, da_Z.values, title=None,
                                      cbar_label='Temperature ('+u'\xb0'+'C)', cbar_diff_label='Temperature Change ('+u'\xb0'+'C)', Sci=True)
plt.savefig(rootdir+'PLOTS/'+time_step+'_Temperature_DiffInTime_x'+str(x_coord), bbox_inches = 'tight', pad_inches = 0.1)

#---------------------------------------------------------------------------
# Plot timeseries predictions
#---------------------------------------------------------------------------
if time_step == '24hrs':
   fig = SecPlt.plt_timeseries(point, 360, {'MITGCM':da_T}, time_step=time_step)
   plt.savefig(rootdir+'PLOTS/'+time_step+'_timeseries_1yr_z'+str(point[0])+'y'+str(point[1])+'x'+str(point[2]), bbox_inches = 'tight', pad_inches = 0.1)
   fig = SecPlt.plt_2_timeseries(point, 360, 18000, {'MITGCM':da_T}, time_step=time_step)
   plt.savefig(rootdir+'PLOTS/'+time_step+'_timeseries_50yrs_z'+str(point[0])+'y'+str(point[1])+'x'+str(point[2]), bbox_inches = 'tight', pad_inches = 0.1)
else:
   fig = SecPlt.plt_timeseries(point, 120, {'MITGCM':da_T}, time_step=time_step)
   plt.savefig(rootdir+'PLOTS/'+time_step+'_timeseries_10yrs_z'+str(point[0])+'y'+str(point[1])+'x'+str(point[2]), bbox_inches = 'tight', pad_inches = 0.1)
   fig = SecPlt.plt_2_timeseries(point, 120, 6000, {'MITGCM':da_T}, time_step=time_step)
   plt.savefig(rootdir+'PLOTS/'+time_step+'_timeseries_500yrs_z'+str(point[0])+'y'+str(point[1])+'x'+str(point[2]), bbox_inches = 'tight', pad_inches = 0.1)

#-----------------------------------------
# Plot min and max temp over whole domain
#-----------------------------------------

fig = plt.figure(figsize=(20,3))
plt.plot(nc_mon.variables['dynstat_theta_min'][0:])
plt.title('Minimum Temperature')
plt.ylabel('Temperature (degrees C)')
if time_step == '24hrs':
   plt.xlabel('Time (months)')
else:
   plt.xlabel('Time (decades)')
fig.savefig(rootdir+'PLOTS/'+time_step+'_minT.png', bbox_inches = 'tight', pad_inches = 0.1)
plt.close()

fig = plt.figure(figsize=(20,3))
plt.plot(nc_mon.variables['dynstat_theta_max'][0:])
plt.title('Maximum Temperature')
plt.ylabel('Temperature (degrees C)')
if time_step == '24hrs':
   plt.xlabel('Time (months)')
else:
   plt.xlabel('Time (decades)')
fig.savefig(rootdir+'PLOTS/'+time_step+'_maxT.png', bbox_inches = 'tight', pad_inches = 0.1)
plt.close()
