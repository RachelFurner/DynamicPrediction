#!/usr/bin/env python
# coding: utf-8

print('import packages')
import sys
sys.path.append('/data/hpcdata/users/racfur/DynamicPrediction/code_git/')
from Tools import CreateDataName as cn
from Tools import Iterator as it
from Tools import Model_Plotting as rfplt
from Tools import AssessModel as am
import matplotlib.pyplot as plt

import numpy as np
import xarray as xr
import pickle

#from sklearn.preprocessing import PolynomialFeatures
#from skimage.util import view_as_windows

#----------------------------
# Set variables for this run
#----------------------------
run_vars={'dimension':3, 'lat':True , 'lon':True , 'dep':True , 'current':True , 'sal':True , 'eta':True , 'poly_degree':2}
model_type = 'lr'

exp_prefix = ''

for_len_yrs = 1   # forecast length in years

no_chunks = 1

#----------------------------
for_len = int(for_len_yrs * 12)

data_name = cn.create_dataname(run_vars)
#data_name = 'Steps1to10_'+data_name

exp_name = exp_prefix+data_name

rootdir = '/data/hpcdata/users/racfur/DynamicPrediction/'+model_type+'_Outputs/'
#-------------------------------------------
# Read in netcdf file for shape and 'truth'
#-------------------------------------------
print('reading in ds')
DIR  = '/data/hpcdata/users/racfur/MITGCM_OUTPUT/20000yr_Windx1.00_mm_diag/'
data_filename=DIR+'cat_tave_2000yrs_SelectedVars_masked.nc'
ds = xr.open_dataset(data_filename)
da_T=ds['Ttave'][:for_len+1]   

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
nc_Temp = nc_file.createVariable('Pred_Temp', 'f4', ('T', 'Z', 'Y', 'X'))
nc_DelT = nc_file.createVariable('Pred_DelT', 'f4', ('T', 'Z', 'Y', 'X'))
nc_Errors = nc_file.createVariable('Errors', 'f4', ('T', 'Z', 'Y', 'X'))
nc_Mask = nc_file.createVariable('Mask', 'i4', ('Z', 'Y', 'X'))
nc_SpongeMask = nc_file.createVariable('sponge_mask', 'i4', ('Z', 'Y', 'X'))

# Fill some variables - rest done during iteration steps
#nc_T[:] = ds['T'].data
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

pred_filename = rootdir+'ITERATED_PREDICTION_ARRAYS/'+exp_name+'_IterativePredictions.npz'
size_chunk = int(for_len/no_chunks)
print(size_chunk)
init = da_T[0,:,:,:]

for chunk in range(no_chunks):
   print('')
   print(chunk)
   chunk_start = size_chunk*chunk
   chunk_end = size_chunk*(chunk+1)+1
   tmp_pred, tmp_out, mask, sponge_mask = it.interator(data_name, run_vars, model, size_chunk, ds.isel(T=slice(chunk_start,chunk_end)), init=init, model_type=model_type)
   if chunk == 0:
      Pred_Temp = tmp_pred  # initialise these arrays, and keep both initial state and prediction for this first step
      Pred_DelT = tmp_out       # initialise these arrays, and keep both initial state and prediction for this first step            
   else:
      Pred_Temp = np.concatenate( (Pred_Temp, tmp_pred[-1:,:,:,:]), axis=0 )  # concatenate on time axis. take prediction only - ignore istate
      Pred_DelT = np.concatenate( (Pred_DelT,  tmp_out[-1:,:,:,:]), axis=0 )  # concatenate on time axis. take prediction only - ignore istate

   init = Pred_Temp[-1,:,:,:]
   # Save to array
   np.savez(pred_filename, np.array(Pred_Temp), np.array(Pred_DelT))
   # Save to netcdf file
   length = Pred_Temp.shape[0]
   nc_T[:] = ds['T'].data[:length]
   nc_Temp[:,:,:,:] = Pred_Temp
   nc_DelT[:,:,:,:] = Pred_DelT
   errors = Pred_Temp-da_T.data[1,:,:,:]
   nc_Errors[:,:,:,:] = errors 
   nc_Mask[:,:,:] = mask
   nc_SpongeMask[:,:,:] = sponge_mask
Pred_Temp, Pred_DelT = np.load(pred_filename).values()
print('Pred_Temp.shape')
print(Pred_Temp.shape)


fig = rfplt.Plot_Histogram(errors[1:,1:-1,1:-3,1:-2], 100)
plt.savefig('../../'+model_type+'_Outputs/PLOTS/'+exp_name+'_Errors_IterationSCRIPT_hist', bbox_inches = 'tight', pad_inches = 0.1)

print('Calc stats and print to file')
am.get_stats(model_type, exp_name, name1='Iter_DeltaT', truth1=da_T.data[1:,1:-1,1:-3,1:-2]-da_T.data[:-1,1:-1,1:-3,1:-2], exp1=Pred_DelT[1:,1:-1,1:-3,1:-2], name='Iter_DeltaT_denorm')


point = [5,15,4]
fig = rfplt.plt_timeseries(point, 120, {model_type: Pred_Temp})
plt.savefig(rootdir+'PLOTS/'+model_type+'_'+exp_name+'_multimodel_forecast_10yrs_z'+str(point[0])+'y'+str(point[1])+'x'+str(point[2]), bbox_inches = 'tight', pad_inches = 0.1)


