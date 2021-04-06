# Script written by Rachel Furner
# Calculates average trends from various processes across 
# a dataset of MITgcm output - the trends are outputted by 
# MITgcm, and the script here reads them in and averages them

import numpy as np
import xarray as xr
import netCDF4 as nc4

#------------------
# Set up variables 
#------------------
no_points = 500  # in months/days
skip_rate = 14   # Take every skip_rate point in time, so as to avoid looking at heavily correlated points

DIR  = '/data/hpcdata/users/racfur/MITgcm/verification/MundaySectorConfig_2degree/runs/100yrs/'
data_filename=DIR+'mnc_test_0002/cat_tave.nc'

#---------------------------------------------
# Read in netcdf file for shape and variables
#---------------------------------------------
print('reading in ds')
ds = xr.open_dataset(data_filename)

#-------------------------
# Set up new netcdf array
#-------------------------
print('set up netcdf file')

nc_file = nc4.Dataset('../../../MITGCM_Analysis_Sector/AveragedMITgcmData.nc','w', format='NETCDF4') 
# Create Dimensions
nc_file.createDimension('T', None)
nc_file.createDimension('Z', ds['Z'].shape[0])
nc_file.createDimension('Y', ds['Y'].shape[0])
nc_file.createDimension('X', ds['X'].shape[0])
nc_file.createDimension('Yp1', ds['Yp1'].shape[0])
nc_file.createDimension('Xp1', ds['Xp1'].shape[0])

# Create variables
nc_T = nc_file.createVariable('T', 'i4', 'T')
nc_Z = nc_file.createVariable('Z', 'i4', 'Z')
nc_Y = nc_file.createVariable('Y', 'i4', 'Y')  
nc_X = nc_file.createVariable('X', 'i4', 'X')
nc_Yp1 = nc_file.createVariable('Yp1', 'i4', 'Yp1')  
nc_Xp1 = nc_file.createVariable('Xp1', 'i4', 'Xp1')

nc_Av_ADVr_TH = nc_file.createVariable('Av_ADVr_TH', 'f4', ('Z', 'Y', 'X'))
nc_Av_ADVx_TH = nc_file.createVariable('Av_ADVx_TH', 'f4', ('Z', 'Y', 'Xp1'))
nc_Av_ADVy_TH = nc_file.createVariable('Av_ADVy_TH', 'f4', ('Z', 'Yp1', 'X'))
nc_Av_DFrE_TH = nc_file.createVariable('Av_DFrE_TH', 'f4', ('Z', 'Y', 'X'))
nc_Av_DFrI_TH = nc_file.createVariable('Av_DFrI_TH', 'f4', ('Z', 'Y', 'X'))
nc_Av_DFxE_TH = nc_file.createVariable('Av_DFxE_TH', 'f4', ('Z', 'Y', 'Xp1'))
nc_Av_DFyE_TH = nc_file.createVariable('Av_DFyE_TH', 'f4', ('Z', 'Yp1', 'X'))
nc_Av_TOTTTEND= nc_file.createVariable('Av_TOTTTEND','f4', ('Z', 'Y', 'X'))
nc_Av_UVELTH  = nc_file.createVariable('Av_UVELTH' , 'f4', ('Z', 'Y', 'Xp1'))
nc_Av_VVELTH  = nc_file.createVariable('Av_VVELTH' , 'f4', ('Z', 'Yp1', 'X'))
nc_Av_WVELTH  = nc_file.createVariable('Av_WVELTH' , 'f4', ('Z', 'Y', 'X'))

print('fill netcdf file')   
# Calc averages and add data to netcdf file
nc_Z[:] = ds['Z'].data
nc_Y[:] = ds['Y'].data
nc_X[:] = ds['X'].data
nc_Yp1[:] = ds['Yp1'].data
nc_Xp1[:] = ds['Xp1'].data
nc_T[:] = ds['T'].data[1:no_points*skip_rate+1:skip_rate]

nc_Av_ADVr_TH[:,:,:]  = np.nanmean(np.abs(ds['ADVr_TH'].data[:no_points*skip_rate+1:skip_rate,:,:,:]), axis=0)
nc_Av_ADVx_TH[:,:,:]  = np.nanmean(np.abs(ds['ADVx_TH'].data[:no_points*skip_rate+1:skip_rate,:,:,:]), axis=0)
nc_Av_ADVy_TH[:,:,:]  = np.nanmean(np.abs(ds['ADVy_TH'].data[:no_points*skip_rate+1:skip_rate,:,:,:]), axis=0)
nc_Av_DFrE_TH[:,:,:]  = np.nanmean(np.abs(ds['DFrE_TH'].data[:no_points*skip_rate+1:skip_rate,:,:,:]), axis=0)
nc_Av_DFrI_TH[:,:,:]  = np.nanmean(np.abs(ds['DFrI_TH'].data[:no_points*skip_rate+1:skip_rate,:,:,:]), axis=0)
nc_Av_DFxE_TH[:,:,:]  = np.nanmean(np.abs(ds['DFxE_TH'].data[:no_points*skip_rate+1:skip_rate,:,:,:]), axis=0)
nc_Av_DFyE_TH[:,:,:]  = np.nanmean(np.abs(ds['DFyE_TH'].data[:no_points*skip_rate+1:skip_rate,:,:,:]), axis=0)
nc_Av_TOTTTEND[:,:,:] = np.nanmean(np.abs(ds['TOTTTEND'].data[:no_points*skip_rate+1:skip_rate,:,:,:]), axis=0)
nc_Av_UVELTH[:,:,:]   = np.nanmean(np.abs(ds['UVELTH'].data[:no_points*skip_rate+1:skip_rate,:,:,:]), axis=0)
nc_Av_VVELTH[:,:,:]   = np.nanmean(np.abs(ds['VVELTH'].data[:no_points*skip_rate+1:skip_rate,:,:,:]), axis=0)
nc_Av_WVELTH[:,:,:]   = np.nanmean(np.abs(ds['WVELTH'].data[:no_points*skip_rate+1:skip_rate,:,:,:]), axis=0)

nc_file.close()

