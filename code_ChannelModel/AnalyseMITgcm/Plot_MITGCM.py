#!/usr/bin/env python
# coding: utf-8

print('import packages')
import sys
sys.path.append('../Tools')
import Channel_Model_Plotting as ChnlPlt

import numpy as np
import matplotlib.pyplot as plt
#from sklearn import linear_model
#from sklearn.preprocessing import PolynomialFeatures
#from skimage.util import view_as_windows
import os
import xarray as xr
import pickle
from netCDF4 import Dataset

#------------------------
# Set plotting variables
#------------------------
# Note shape is 38, 108, 240 (z,y,x) and 36000 time steps (100 years)
time = int(.5*36000)+1 
point = [ 2, 50, 100]

time_step = '12hrs'

#----------------------

datadir  = '/data/hpcdata/users/racfur/MITgcm/verification/MundayChannelConfig10km_LandSpits/runs/50yr_Cntrl/'
data_filename=datadir + '12hrly_data.nc'
#data_filename=datadir + '12hrly_small_set.nc'
grid_filename=datadir + 'grid.nc'
mon_file = datadir + 'monitor.nc'
stats_file = datadir + 'Spits_stats.nc'
rootdir = '../../../MITGCM_Analysis_Channel/'+time_step+'/'


#-----------------------------------------------
# Read in netcdf file and get shape
#-----------------------------------------------
print('reading in ds')
ds = xr.open_dataset(data_filename)
da_T = ds['THETA'][:,:,:,:]
da_X = ds['X']
da_Y = ds['Y'][:]
ds_grid = xr.open_dataset(grid_filename)
da_Z = ds_grid['RC']

z_size = da_T.shape[1]
y_size = da_T.shape[2]
x_size = da_T.shape[3]

HFacC = ds_grid['HFacC'].values
HFacW = ds_grid['HFacW'].values
HFacS = ds_grid['HFacS'].values

print('da_T.shape')
print(da_T.shape)

#----------------------
# Read in monitor file
#----------------------
ds_mon = xr.open_dataset(mon_file)
ds_mon = Dataset(mon_file)

#--------------------
# Read in stats file
#--------------------
ds_stats = xr.open_dataset(stats_file)

#------------
# Plot Means
#------------
fig, ax, im = ChnlPlt.plot_depth_fld(np.where(HFacC[0,:,:]>0., ds_stats['MeanEta'][0,:,:], np.nan), 'Mean Sea Surface Height (m)', 0,
                                     da_X.values, da_Y.values, da_Z.values,
                                     title=None, cmap='PRGn',
                                     min_value= -max( abs(np.amin(ds_stats['MeanEta'][0,:,:])), np.amax(ds_stats['MeanEta'][0,:,:]) ),
                                     max_value=  max( abs(np.amin(ds_stats['MeanEta'][0,:,:])), np.amax(ds_stats['MeanEta'][0,:,:]) ) )
plt.savefig(rootdir+'PLOTS/MeanEta', bbox_inches = 'tight', pad_inches = 0.1)
plt.close()
   
for level in range(38):
   fig, ax, im = ChnlPlt.plot_depth_fld(np.where(HFacC[level,:,:]>0., ds_stats['MeanTemp'][level,:,:], np.nan), 'Mean Temperature ('+u'\xb0'+'C)', level,
                                        da_X.values, da_Y.values, da_Z.values,
                                        title=None, min_value=None, max_value=None)
                                        #title=None, min_value=0.0, max_value=6.5, extend='both', cmap='bwr')
   plt.savefig(rootdir+'PLOTS/MeanTemp_z'+str(level), bbox_inches = 'tight', pad_inches = 0.1)
   plt.close()
   
   fig, ax, im = ChnlPlt.plot_depth_fld(np.where(HFacW[level,:,:]>0., ds_stats['MeanUVel'][level,:,:], np.nan), 'Mean East-West Velocity (m/s)', level,
                                        da_X.values, da_Y.values, da_Z.values,
                                        title=None, cmap='PRGn',
                                        min_value=-0.3, max_value=0.3, extend='both')
                                        #min_value= -max( abs(np.amin(ds_stats['MeanUVel'][level,:,:])), np.amax(ds_stats['MeanUVel'][level,:,:]) ),
                                        #max_value=  max( abs(np.amin(ds_stats['MeanUVel'][level,:,:])), np.amax(ds_stats['MeanUVel'][level,:,:]) )   )
   plt.savefig(rootdir+'PLOTS/MeanUVel_z'+str(level), bbox_inches = 'tight', pad_inches = 0.1)
   plt.close()
   
   fig, ax, im = ChnlPlt.plot_depth_fld(np.where(HFacS[level,:,:]>0., ds_stats['MeanVVel'][level,:,:], np.nan), 'Mean North-South Velocity (m/s)', level,
                                        da_X.values, da_Y.values, da_Z.values,
                                        title=None, cmap='PRGn',
                                        min_value=-0.1, max_value=0.1, extend='both')
                                        #min_value= -max( abs(np.amin(ds_stats['MeanVVel'][level,:,:])), np.amax(ds_stats['MeanVVel'][level,:,:]) ),
                                        #max_value=  max( abs(np.amin(ds_stats['MeanVVel'][level,:,:])), np.amax(ds_stats['MeanVVel'][level,:,:]) )   )
   plt.savefig(rootdir+'PLOTS/MeanVVel_z'+str(level), bbox_inches = 'tight', pad_inches = 0.1)
   plt.close()
   
#-----------
# Plot Stds
#-----------
fig, ax, im = ChnlPlt.plot_depth_fld(np.where(HFacC[0,:,:]>0., ds_stats['StdEta'][0,:,:], np.nan), 'Std Sea Surface Height (m)', 0,
                                     da_X.values, da_Y.values, da_Z.values,
                                     title=None, min_value=None, max_value=None)
plt.savefig(rootdir+'PLOTS/StdEta', bbox_inches = 'tight', pad_inches = 0.1)
plt.close()
   
for level in range(38):
   fig, ax, im = ChnlPlt.plot_depth_fld(np.where(HFacC[level,:,:]>0., ds_stats['StdTemp'][level,:,:], np.nan), 'Std Temperature ('+u'\xb0'+'C)', level,
                                      da_X.values, da_Y.values, da_Z.values,
                                      title=None, min_value=None, max_value=None)
                                      #title=None, min_value=0.0, max_value=0.8, extend='both')
   plt.savefig(rootdir+'PLOTS/StdTemp_z'+str(level), bbox_inches = 'tight', pad_inches = 0.1)
   plt.close()
   
   fig, ax, im = ChnlPlt.plot_depth_fld(np.where(HFacW[level,:,:]>0., ds_stats['StdUVel'][level,:,:], np.nan), 'Std East-West Velocity (m/s)', level,
                                      da_X.values, da_Y.values, da_Z.values,
                                      title=None, min_value=None, max_value=None)
                                      #title=None, min_value=0.0, max_value=0.3, extend='both')
   plt.savefig(rootdir+'PLOTS/StdUVel_z'+str(level), bbox_inches = 'tight', pad_inches = 0.1)
   plt.close()
   
   fig, ax, im = ChnlPlt.plot_depth_fld(np.where(HFacS[level,:,:]>0., ds_stats['StdVVel'][level,:,:], np.nan), 'Std North-South Velocity (m/s)', level,
                                      da_X.values, da_Y.values, da_Z.values,
                                      title=None, min_value=None, max_value=None)
                                      #title=None, min_value=0.0, max_value=0.32, extend='both')
   plt.savefig(rootdir+'PLOTS/StdVVel_z'+str(level), bbox_inches = 'tight', pad_inches = 0.1)
   plt.close()
   
#--------------------------
# Plot spatial depth plots
#--------------------------
level = point[0]
y_coord = point[1]
x_coord = point[2]

print('plot spatial depth plots')
fig, ax, im = ChnlPlt.plot_depth_fld(np.where(HFacC[level,:,:]>0., da_T[time,level,:,:], np.nan), 'Temperature ('+u'\xb0'+'C)', level,
                                     da_X.values, da_Y.values, da_Z.values,
                                     title=None, min_value=0.0, max_value=6.3)
plt.savefig(rootdir+'PLOTS/Temp_z'+str(level)+'_time'+str(time), bbox_inches = 'tight', pad_inches = 0.1)
plt.close()

fig, ax, im = ChnlPlt.plot_depth_fld(np.where(HFacW[level,:,:]>0., ds['UVEL'][time,level,:,:], np.nan), 'East-West Velocity (m/s)', level,
                                     da_X.values, da_Y.values, da_Z.values,
                                     title=None, cmap='PRGn',
                                     min_value = -0.5, max_value = 0.5, extend='both')
                                     #min_value= -max( abs(np.amin(ds['UVEL'][level,:,:])), np.amax(ds['UVEL'][level,:,:]) ),
                                     #max_value=  max( abs(np.amin(ds['UVEL'][level,:,:])), np.amax(ds['UVEL'][level,:,:]) )   )
plt.savefig(rootdir+'PLOTS/UVel_z'+str(level)+'_time'+str(time), bbox_inches = 'tight', pad_inches = 0.1)
plt.close()

fig, ax, im = ChnlPlt.plot_depth_fld(np.where(HFacS[level,:,:]>0., ds['VVEL'][time,level,:,:], np.nan), 'North-South Velocity (m/s)', level,
                                     da_X.values, da_Y.values, da_Z.values,
                                     title=None, cmap='PRGn',
                                     min_value = -0.5, max_value = 0.5, extend='both')
                                     #min_value= -max( abs(np.amin(ds['VVEL'][level,:,:])), np.amax(ds['VVEL'][level,:,:]) ),
                                     #max_value=  max( abs(np.amin(ds['VVEL'][level,:,:])), np.amax(ds['VVEL'][level,:,:]) )   )
plt.savefig(rootdir+'PLOTS/VVel_z'+str(level)+'_time'+str(time), bbox_inches = 'tight', pad_inches = 0.1)
plt.close()

fig, ax, im = ChnlPlt.plot_depth_fld(np.where(HFacC[0,:,:]>0., ds['ETAN'][time,0,:,:], np.nan), 'Sea Surface Height (m)', 0,
                                     da_X.values, da_Y.values, da_Z.values,
                                     title=None, cmap='PRGn',
                                     min_value = -0.8, max_value = 0.8)
plt.savefig(rootdir+'PLOTS/Eta_time'+str(time), bbox_inches = 'tight', pad_inches = 0.1)
plt.close()

#-----------------------------
# Plot spatial tendancy plots
#-----------------------------
level = 2
print('plot tendancy plots')
fig, ax, im = ChnlPlt.plot_depth_fld(np.where(HFacC[level,:,:]>0., da_T[time+1,level,:,:]-da_T[time,level,:,:], np.nan),
                                     'Temperature Increment ('+u'\xb0'+'C)', level,
                                     da_X.values, da_Y.values, da_Z.values,
                                     title=None, min_value=-0.6, max_value=0.6, cmap='PRGn', extend='both')
plt.savefig(rootdir+'PLOTS/TempTend_z'+str(level)+'_time'+str(time), bbox_inches = 'tight', pad_inches = 0.1)
plt.close()

fig, ax, im = ChnlPlt.plot_depth_fld(np.where(HFacW[level,:,:]>0., ds['UVEL'][time+1,level,:,:]-ds['UVEL'][time,level,:,:], np.nan), 
                                     'East-West Velocity Increment (m/s)', level,
                                     da_X.values, da_Y.values, da_Z.values,
                                     title=None, cmap='PRGn',
                                     min_value=-0.2, max_value=0.2, extend='both')
plt.savefig(rootdir+'PLOTS/UVelTend_z'+str(level)+'_time'+str(time), bbox_inches = 'tight', pad_inches = 0.1)
plt.close()

fig, ax, im = ChnlPlt.plot_depth_fld(np.where(HFacS[level,:,:]>0., ds['VVEL'][time+1,level,:,:]-ds['VVEL'][time,level,:,:], np.nan),
                                     'North-South Velocity Increment (m/s)', level,
                                     da_X.values, da_Y.values, da_Z.values,
                                     title=None, cmap='PRGn',
                                     min_value=-0.3, max_value=0.3, extend='both')
plt.savefig(rootdir+'PLOTS/VVelTend_z'+str(level)+'_time'+str(time), bbox_inches = 'tight', pad_inches = 0.1)
plt.close()

fig, ax, im = ChnlPlt.plot_depth_fld(np.where(HFacC[0,:,:]>0., ds['ETAN'][time+1,0,:,:]-ds['ETAN'][time,0,:,:], np.nan),
                                     'Sea Surface Height Increment (m)', 0,
                                     da_X.values, da_Y.values, da_Z.values,
                                     title=None, cmap='PRGn',
                                     min_value=-0.08, max_value=0.08, extend='both')
plt.savefig(rootdir+'PLOTS/EtaTend_time'+str(time), bbox_inches = 'tight', pad_inches = 0.1)
plt.close()

   
#----------------------------------
# Plot diffs between time t and t+1
#----------------------------------
print('plot diffs between t and t+1')
fig = ChnlPlt.plot_depth_fld_diff(da_T[time,level,:,:], 'Temp at time t', da_T[time+1,level,:,:], 'Temp at time t+1', level,
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

#-----------------------------------------
# Plot min and max temp over whole domain
#-----------------------------------------
print('plot min and max temp plots')
fig = plt.figure(figsize=(15,3))
plt.plot(ds_mon.variables['dynstat_theta_min'][0:])
plt.title('Minimum Temperature')
plt.ylabel('Temperature (degrees C)')
plt.xlabel('Time (years)')
fig.savefig(rootdir+time_step+'_minT.png', bbox_inches = 'tight', pad_inches = 0.1)
plt.close()

fig = plt.figure(figsize=(15,3))
plt.plot(ds_mon.variables['dynstat_theta_max'][0:])
plt.title('Maximum Temperature')
plt.ylabel('Temperature (degrees C)')
plt.xlabel('Time (years)')
fig.savefig(rootdir+time_step+'_maxT.png', bbox_inches = 'tight', pad_inches = 0.1)
plt.close()

#-------------------------------------
# Plot time series of max and mean KE
#-------------------------------------
print('plot max and mean KE plots')
fig = plt.figure(figsize=(15,3))
plt.plot(ds_mon.variables['ke_max'][0:])
plt.title('Maximum KE over domain')
plt.ylabel('KE')
plt.xlabel('Time (years)')
fig.savefig(rootdir+time_step+'_maxKE.png', bbox_inches = 'tight', pad_inches = 0.1)
plt.close()

fig = plt.figure(figsize=(15,3))
plt.plot(ds_mon.variables['ke_mean'][0:])
plt.title('Mean KE over domain')
plt.ylabel('KE')
plt.xlabel('Time (years)')
fig.savefig(rootdir+time_step+'_meanKE.png', bbox_inches = 'tight', pad_inches = 0.1)
plt.close()

##---------------------------------------------------------------------------
## Plot timeseries
##---------------------------------------------------------------------------
#print('plot timeseries')
#fig = ChnlPlt.plt_timeseries(point, 360, {'MITGCM':da_T.values}, time_step=time_step)
#plt.savefig(rootdir+'PLOTS/'+time_step+'_timeseries_1yr_z'+str(point[0])+'y'+str(point[1])+'x'+str(point[2]), bbox_inches = 'tight', pad_inches = 0.1)
#fig = ChnlPlt.plt_2_timeseries(point, 360, 18000, {'MITGCM':da_T.values}, time_step=time_step)
#plt.savefig(rootdir+'PLOTS/'+time_step+'_timeseries_50yrs_z'+str(point[0])+'y'+str(point[1])+'x'+str(point[2]), bbox_inches = 'tight', pad_inches = 0.1)

