#!/usr/bin/env python
# coding: utf-8

print('import packages')
import sys
sys.path.append('../Tools')

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import netCDF4 as nc4

#------------------------
datadir  = '/data/hpcdata/users/racfur/MITgcm/verification/MundayChannelConfig10km_LandSpits/runs/50yr_Cntrl/'
#data_filename=datadir + '12hrly_data.nc'
data_filename=datadir + '12hrly_small_set.nc'
grid_filename=datadir + 'grid.nc'
mon_file = datadir + 'monitor.nc'
rootdir = '../../../MITGCM_Analysis_Channel/'

#-----------------------------------------------
# Read in netcdf file and get shape
#-----------------------------------------------
print('reading in ds')
ds = xr.open_dataset(data_filename)
# ignore last 8 y -points as these are land
da_T = ds['THETA']
da_U = ds['UVEL']
da_V = ds['VVEL']
da_E = ds['ETAN']
da_X = ds['X']
da_Xp1 = ds['Xp1']
da_Y = ds['Y']
da_Yp1 = ds['Yp1']
da_Z = ds['Zmd000038']

print('da_T.shape')
print(da_T.shape)

#----------------------
# Read in monitor file
#----------------------
ds_mon = xr.open_dataset(mon_file)
nc_mon = nc4.Dataset(mon_file)

#---------------------------------
# Calculate mean values per point
#---------------------------------
T_mean = np.nanmean(da_T, 0)
U_mean = np.nanmean(da_U, 0)
V_mean = np.nanmean(da_V, 0)
E_mean = np.nanmean(da_E, 0)

T_std = np.nanstd(da_T, 0)
U_std = np.nanstd(da_U, 0)
V_std = np.nanstd(da_V, 0)
E_std = np.nanstd(da_E, 0)

#-----------------------
# Output to Netcdf file 
#-----------------------
out_filename = datadir+'/stats.nc'
out_file = nc4.Dataset(out_filename,'w', format='NETCDF4') #'w' stands for write

# Create Dimensions
out_file.createDimension('Z', da_Z.shape[0])
out_file.createDimension('Y', da_Y.shape[0])
out_file.createDimension('Yp1', da_Yp1.shape[0])
out_file.createDimension('X', da_X.shape[0])
out_file.createDimension('Xp1', da_Xp1.shape[0])

# Create variables
nc_Z = out_file.createVariable('Z', 'i4', 'Z')
nc_Y = out_file.createVariable('Y', 'i4', 'Y')  
nc_X = out_file.createVariable('X', 'i4', 'X')

nc_MeanTemp = out_file.createVariable( 'MeanTemp'  , 'f4', ('Z', 'Y', 'X') )
nc_MeanUVel = out_file.createVariable( 'MeanUVel'  , 'f4', ('Z', 'Y', 'Xp1') )
nc_MeanVVel = out_file.createVariable( 'MeanVVel'  , 'f4', ('Z', 'Yp1', 'X') )
nc_MeanEta  = out_file.createVariable( 'MeanEta'   , 'f4', ('Y', 'X') )

nc_StdTemp = out_file.createVariable( 'StdTemp'  , 'f4', ('Z', 'Y', 'X') )
nc_StdUVel = out_file.createVariable( 'StdUVel'  , 'f4', ('Z', 'Y', 'Xp1') )
nc_StdVVel = out_file.createVariable( 'StdVVel'  , 'f4', ('Z', 'Yp1', 'X') )
nc_StdEta  = out_file.createVariable( 'StdEta'   , 'f4', ('Y', 'X') )

# Fill variables
nc_MeanTemp[:,:,:] = T_mean
nc_MeanUVel[:,:,:] = U_mean
nc_MeanVVel[:,:,:] = V_mean
nc_MeanEta[:,:]  = E_mean

nc_StdTemp[:,:,:] = T_std
nc_StdUVel[:,:,:] = U_std
nc_StdVVel[:,:,:] = V_std
nc_StdEta[:,:]  = E_std

