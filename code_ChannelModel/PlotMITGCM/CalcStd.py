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

out_file = nc4.Dataset(out_filename,'r+', format='NETCDF4') #'w' stands for write

# Create variables
nc_StdTemp = out_file.createVariable( 'StdTemp'  , 'f4', ('Z', 'Y', 'X') )
nc_StdUVel = out_file.createVariable( 'StdUVel'  , 'f4', ('Z', 'Y', 'Xp1') )

# Fill variables
da_T = ds['THETA']
nc_StdTemp[:,:,:] = np.nanstd(da_T, 0)
del da_T
gc.collect()
print('Temp done')

da_U = ds['UVEL']
nc_StdUVel[:,:,:] = np.nanstd(da_U, 0)
del da_U
gc.collect()
print('U done')
