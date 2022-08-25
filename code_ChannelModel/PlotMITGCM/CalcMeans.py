#!/usr/bin/env python
# coding: utf-8

print('import packages')
import sys
sys.path.append('../Tools')

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import netCDF4 as nc4
import gc as gc

#------------------------
datadir  = '/data/hpcdata/users/racfur/MITgcm/verification/MundayChannelConfig10km_LandSpits/runs/50yr_Cntrl/'
data_filename=datadir + '12hrly_data.nc'
out_filename = datadir+'/stats.nc'
#data_filename=datadir + '12hrly_small_set.nc'
#out_filename = datadir+'/stats_small.nc'

print('reading in ds')
ds = xr.open_dataset(data_filename)

out_file = nc4.Dataset(out_filename,'w', format='NETCDF4') #'w' stands for write

# Create Dimensions
da_Z = ds['Zmd000038']
da_Y = ds['Y']
da_Yp1 = ds['Yp1']
da_X = ds['X']
da_Xp1 = ds['Xp1']
out_file.createDimension('Z', da_Z.shape[0])
out_file.createDimension('Y', da_Y.shape[0])
out_file.createDimension('Yp1', da_Yp1.shape[0])
out_file.createDimension('X', da_X.shape[0])
out_file.createDimension('Xp1', da_Xp1.shape[0])
del da_Z
del da_Y
del da_Yp1
del da_X
del da_Xp1
gc.collect()

# Create variables
nc_Z = out_file.createVariable('Z', 'i4', 'Z')
nc_Y = out_file.createVariable('Y', 'i4', 'Y')  
nc_X = out_file.createVariable('X', 'i4', 'X')

nc_MeanTemp = out_file.createVariable( 'MeanTemp'  , 'f4', ('Z', 'Y', 'X') )
nc_MeanUVel = out_file.createVariable( 'MeanUVel'  , 'f4', ('Z', 'Y', 'Xp1') )

# Fill variables
da_T = ds['THETA']
nc_MeanTemp[:,:,:] = np.nanmean(da_T, 0)
del da_T
gc.collect()
print('Temp done')

da_U = ds['UVEL']
nc_MeanUVel[:,:,:] = np.nanmean(da_U, 0)
del da_U
gc.collect()
print('U done')
