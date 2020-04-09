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

import netCDF4 as nc4

#from sklearn.preprocessing import PolynomialFeatures
#from skimage.util import view_as_windows

#----------------------------
# Set variables for this run
#----------------------------
run_vars={'dimension':3, 'lat':True , 'lon':True , 'dep':True , 'current':True , 'sal':True , 'eta':True , 'density':True , 'poly_degree':2}
model_type = 'lr'
iterate_method = 'AB1'

data_prefix='WithThroughFlow_'
model_prefix=''
exp_prefix = iterate_method+'_'

for_len_yrs = 1/2   # forecast length in years
no_chunks = 1
start = 0        # Can't iterate, as the code crashes when it reaches NaN's, so just have to manually do one at a time.

run_iterations = True 

#----------------------------
for_len = int(for_len_yrs * 12)

data_name = cn.create_dataname(run_vars)
data_name = data_prefix+data_name
model_name = model_prefix+data_name
exp_name = exp_prefix+model_name

rootdir = '/data/hpcdata/users/racfur/DynamicPrediction/'+model_type+'_Outputs/'
#-------------------------------------------
# Read in netcdf file for shape and 'truth'
#-------------------------------------------
print('reading in ds')
DIR  = '/data/hpcdata/users/racfur/MITGCM_OUTPUT/20000yr_Windx1.00_mm_diag/'
data_filename=DIR+'cat_tave_2000yrs_SelectedVars_masked.nc'
ds = xr.open_dataset(data_filename)
ds = ds.isel(T=slice(start,start+for_len+1))
#da_T=ds['Ttave'][start:start+for_len+1]   
da_T=ds['Ttave']

z_size = da_T.shape[1]
y_size = da_T.shape[2]
x_size = da_T.shape[3]

print('da_T.shape')
print(da_T.shape)

#-------------------
# Read in the model
#-------------------
pkl_filename = rootdir+'MODELS/pickle_'+model_name+'.pkl'
with open(pkl_filename, 'rb') as file:
   print('opening '+pkl_filename)
   model = pickle.load(file)

#---------------------
# Set up netcdf files
#---------------------

nc_file = nc4.Dataset(rootdir+'ITERATED_PREDICTION_ARRAYS/'+exp_name+'_IterativePredictions_'+str(start)+'.nc','w', format='NETCDF4') #'w' stands for write
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
nc_SpongeMask = nc_file.createVariable('sponge_mask', 'i4', ('Z', 'Y', 'X'))
nc_Mask = nc_file.createVariable('Mask', 'i4', ('Z', 'Y', 'X'))

# Fill some variables - rest done during iteration steps
nc_Z[:] = ds['Z'].data
nc_Y[:] = ds['Y'].data
nc_X[:] = ds['X'].data

#----------------------
# Make the predictions
#----------------------

# Call iterator for chunks of data at a time, saving interim results in arrays, so if the
# predictions become unstable and lead to Nan's etc we still have some of the trajectory saved
print('Call iterator and make the predictions')

pred_filename = rootdir+'ITERATED_PREDICTION_ARRAYS/'+exp_name+'_IterativePredictions_'+str(start)+'.npz'
size_chunk = int(for_len/no_chunks)
print(size_chunk)
init = da_T[0,:,:,:]

if run_iterations:
   for chunk in range(no_chunks):
      print('')
      print(chunk)
      chunk_start = size_chunk*chunk
      chunk_end = size_chunk*(chunk+1)+1
      tmp_pred, tmp_out, sponge_mask, mask = it.iterator(data_name, run_vars, model, size_chunk, ds.isel(T=slice(chunk_start,chunk_end)),
                                   init=init, model_type=model_type, method=iterate_method)
      if chunk == 0:
         Pred_Temp = tmp_pred  # initialise these arrays, and keep both initial state and prediction for this first step
         Pred_DelT = tmp_out       # initialise these arrays, and keep both initial state and prediction for this first step            
      else:
         Pred_Temp = np.concatenate( (Pred_Temp, tmp_pred[1:,:,:,:]), axis=0 )  # concatenate on time axis. take prediction only - ignore istate
         Pred_DelT = np.concatenate( (Pred_DelT,  tmp_out[1:,:,:,:]), axis=0 )  # concatenate on time axis. take prediction only - ignore istate
   
      init = Pred_Temp[-1,:,:,:]
      # Save to array
      np.savez(pred_filename, np.array(Pred_Temp), np.array(Pred_DelT))
      # Save to netcdf file
      length = Pred_Temp.shape[0]
      nc_T[:] = ds['T'].data[:length]
      nc_Temp[:,:,:,:] = Pred_Temp
      nc_DelT[:,:,:,:] = Pred_DelT
      errors = Pred_Temp-da_T.data[:length,:,:,:]
      nc_Errors[:,:,:,:] = errors 
      nc_SpongeMask[:,:,:] = sponge_mask
      nc_Mask[:,:,:] = mask
Pred_Temp, Pred_DelT = np.load(pred_filename).values()
length = Pred_Temp.shape[0]
errors = Pred_Temp-da_T.data[:length,:,:,:]
print('Pred_Temp.shape')
print(Pred_Temp.shape)

print(errors.shape)
print(mask.shape)
mask_4d=np.broadcast_to(mask,errors.shape)
print(mask_4d.shape)
fig = rfplt.Plot_Histogram(errors[mask_4d==1], 100)
plt.savefig('../../'+model_type+'_Outputs/PLOTS/'+exp_name+'_Errors_IterationSCRIPT_hist_'+str(start), bbox_inches = 'tight', pad_inches = 0.1)

print('Calc stats and print to file')
print(da_T.data.shape)
print(da_T.data[mask_4d==1].shape)
if np.isnan(Pred_DelT[1:,:,:,:][mask_4d[1:,:,:,:]==1]).any():
   print( 'Pred_DelT array contains a NaN at ' + str( np.argwhere(np.isnan(Pred_DelT[1:,:,:,:][mask_4d[1:,:,:,:]==1])) ) )
if np.isnan(da_T.data[1:,:,:,:]-da_T.data[:-1,:,:,:])[mask_4d[1:,:,:,:]==1].any():
   print( 'true_DelT array contains a NaN at ' + str( np.argwhere(np.isnan(da_T.data[1:,:,:,:]-da_T.data[:-1,:,:,:])[mask_4d[1:,:,:,:]==1]) ) )

am.get_stats(model_type, exp_name, name1='Iter_DeltaT', truth1=(da_T.data[1:,:,:,:]-da_T.data[:-1,:,:,:])[mask_4d[1:,:,:,:]==1],
                                                          exp1=(Pred_DelT[1:,:,:,:])[mask_4d[1:,:,:,:]==1], name='Iter_DeltaT_denorm_'+str(start))

for point in [ [5,15,4] ]:
   fig = rfplt.plt_timeseries(point, 120, {model_type: Pred_Temp})
   plt.savefig(rootdir+'PLOTS/'+model_type+'_'+exp_name+'_multimodel_forecast_10yrs_z'+str(point[0])+'y'+str(point[1])+'x'+str(point[2])+'_'+str(start), bbox_inches = 'tight', pad_inches = 0.1)
   

