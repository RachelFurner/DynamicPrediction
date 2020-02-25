#!/usr/bin/env python
# coding: utf-8

print('import packages')
import sys
sys.path.append('/data/hpcdata/users/racfur/DynamicPrediction/code_git/')
from Tools import CreateDataName as cn
from Tools import Iterator as it

import numpy as np
import xarray as xr
import pickle

#from sklearn.preprocessing import PolynomialFeatures
#from skimage.util import view_as_windows

#----------------------------
# Set variables for this run
#----------------------------
run_vars={'dimension':3, 'lat':True , 'lon':True , 'dep':True , 'current':True , 'sal':True , 'eta':True , 'poly_degree':2}
model_type = 'nn'

exp_prefix = ''

for_len_yrs = 100    # forecast length in years

no_chunks = 100

#----------------------------
for_len = int(for_len_yrs * 12)

data_name = cn.create_dataname(run_vars)
exp_name = exp_prefix+data_name

rootdir = '/data/hpcdata/users/racfur/DynamicPrediction/'+model_type+'_Outputs/'
#-------------------------------------------
# Read in netcdf file for shape and 'truth'
#-------------------------------------------
print('reading in ds')
DIR  = '/data/hpcdata/users/racfur/MITGCM_OUTPUT/20000yr_Windx1.00_mm_diag/'
if for_len_yrs < 500:
   data_filename=DIR+'cat_tave_500yrs_SelectedVars_masked.nc'
else:
   data_filename=DIR+'cat_tave_5000yrs_SelectedVars_masked.nc'
ds = xr.open_dataset(data_filename)
da_T=ds['Ttave'][:for_len_yrs*12+1]   

z_size = da_T.shape[1]
y_size = da_T.shape[2]
x_size = da_T.shape[3]

print('da_T.shape')
print(da_T.shape)

#---------------------
# Set up netcdf files
#---------------------

import netCDF4 as nc4

nc_file = nc4.Dataset(rootdir+'ITERATED_PREDICTION_ARRAYS/'+exp_name+'_IterativePredictions.nc','w', format='NETCDF4') #'w' stands for write
# Create Dimensions
nc_file.createDimension('T', None)
nc_file.createDimension('Z', ds['Z'].shape[0])
nc_file.createDimension('Y', ds['Y'].shape[0])
nc_file.createDimension('X', ds['X'].shape[0])

# Create variables
nc_T = nc_file.createVariable('T', 'i4', 'T')
nc_Z = nc_file.createVariable('Z', 'i4', 'Z')
nc_Y = nc_file.createVariable('Y', 'i4', 'Y')  
nc_X = nc_file.createVariable('X', 'i4', 'X')
nc_Ttave = nc_file.createVariable('Temperature', 'f4', ('T', 'Z', 'Y', 'X'))
nc_Errors = nc_file.createVariable('Prediction-Truth', 'f4', ('T', 'Z', 'Y', 'X'))

# Fill some variables - rest done during iteration steps
nc_T[:] = ds['T'].data
nc_Z[:] = ds['Z'].data
nc_Y[:] = ds['Y'].data
nc_X[:] = ds['X'].data

#----------------------
# Make the predictions
#----------------------
# Read in the model
pkl_filename = rootdir+'MODELS/pickle_'+model_type+'_'+exp_name+'.pkl'
with open(pkl_filename, 'rb') as file:
   print('opening '+pkl_filename)
   model = pickle.load(file)

# Call iterator for chunks of data at a time, saving interim results in arrays, so if the
# predictions become unstable and lead to Nan's etc we still have some of the trajectory saved
print('Call iterator and make the predictions')
predictions = np.zeros((for_len+1, z_size, y_size, x_size))
predictions[:,:,:,:]=np.nan
pred_filename = rootdir+'ITERATED_PREDICTION_ARRAYS/'+model_type+'_'+exp_name+'_IterativePredictions.npy'
size_chunk = int(for_len/no_chunks)
print(size_chunk)
init = da_T[0,:,:,:]
for chunk in range(no_chunks):
   print('')
   print(chunk)
   chunk_start = size_chunk*chunk
   chunk_end = size_chunk*(chunk+1)+1
   predictions[chunk_start:chunk_end,:,:,:] = it.interator(exp_name, run_vars, model, size_chunk, ds.isel(T=slice(chunk_start,chunk_end)), model_type=model_type, init=init)
   init = predictions[size_chunk*(chunk+1),:,:,:]
   # Save to array
   np.save(pred_filename, np.array(predictions))
   # Save to netcdf file
   nc_Ttave[:,:,:,:] = predictions
   errors = predictions-da_T.data
   nc_Errors[:,:,:,:] = errors 
predictions = np.load(pred_filename)
print('predictions.shape')
print(predictions.shape)
