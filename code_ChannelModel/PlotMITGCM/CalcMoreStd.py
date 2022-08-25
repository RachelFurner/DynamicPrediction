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
nc_StdVVel = out_file.createVariable( 'StdVVel'  , 'f4', ('Z', 'Yp1', 'X') )
nc_StdEta  = out_file.createVariable( 'StdEta'   , 'f4', ('Y', 'X') )

# Fill variables
da_V = ds['VVEL']
nc_StdVVel[:,:,:] = np.nanstd(da_V, 0)
del da_V
gc.collect()
print('V done')

da_E = ds['ETAN']
nc_StdEta[:,:] = np.nanstd(da_E, 0)
del da_E
gc.collect()
print('Eta done')


