#!/usr/bin/env python
# coding: utf-8

print('import packages')
import sys
sys.path.append('../')
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
run_vars={'dimension':2, 'lat':True , 'lon':True , 'dep':True , 'current':True , 'bolus_vel':True , 'sal':True , 'eta':True , 'density':True , 'poly_degree':2}
model_type = 'lr'
iterate_method = 'AB3'

time_step = '24hrs'
data_prefix=''
model_prefix=''
exp_prefix = ''

for_len = 30*5
no_chunks = 5

start = 5        # Allow for a variety of start points, from different MITGCM init states

run_iterations = True  

#----------------------------
DIR  = '/data/hpcdata/users/racfur/MITGCM_OUTPUT/100yr_Windx1.00_FrequentOutput/'
data_filename=DIR+'cat_tave_50yr_SelectedVars_masked_withBolus.nc'
density_file = DIR+'DensityData.npy'

data_name = data_prefix+cn.create_dataname(run_vars)+'_'+time_step
model_name = model_prefix+data_name
exp_name = exp_prefix+model_name+'_'+iterate_method

rootdir = '../../'+model_type+'_Outputs/'

nc_filename = rootdir+'ITERATED_PREDICTION_ARRAYS/'+exp_name+'_IterativePredictions_'+str(start)+'.nc'
pred_filename = rootdir+'ITERATED_PREDICTION_ARRAYS/'+exp_name+'_IterativePredictions_'+str(start)+'.npz'

#-------------------------------------------------------------------
# Read in netcdf file for shape, 'truth', and other variable inputs
#-------------------------------------------------------------------
print('reading in ds')
ds_orig = xr.open_dataset(data_filename)
ds = ds_orig.isel(T=slice(start,start+for_len+1))

da_T=ds['Ttave']
print('da_T.shape')
print(da_T.shape)

#Read in density from separate array - it was outputted using MITGCM levels code, so not in ncfile
density = np.load( density_file, mmap_mode='r' ) 
density = density[start:start+for_len+1,:,:,:]
print('density.shape')
print(density.shape) 

#-------------------
# Read in the model
#-------------------
pkl_filename = rootdir+'MODELS/'+model_name+'_pickle.pkl'
with open(pkl_filename, 'rb') as file:
   print('opening '+pkl_filename)
   model = pickle.load(file)

if run_iterations:
   #---------------------
   # Set up netcdf files
   #---------------------
   nc_file = nc4.Dataset(nc_filename,'w', format='NETCDF4') #'w' stands for write
   
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
   
   size_chunk = int(for_len/no_chunks)
   print(size_chunk)
   init = da_T[0,:,:,:]
   
   ## Set up initial out_tm1, out_tm2 etc for AB methods
   if iterate_method == 'AB1':
      outs = {}
   elif iterate_method == 'AB2':
      ds_outs = ds_orig.isel(T=slice(start-1,start+1))
      da_T_outs=ds_outs['Ttave']
      outs = {'tm1':(da_T_outs[-1,:,:,:]-da_T_outs[-2,:,:,:])}
   elif iterate_method == 'AB3':
      ds_outs = ds_orig.isel(T=slice(start-2,start+1))
      da_T_outs=ds_outs['Ttave']
      outs = {'tm1':(da_T_outs[-1,:,:,:]-da_T_outs[-2,:,:,:]),
              'tm2':(da_T_outs[-2,:,:,:]-da_T_outs[-3,:,:,:])}
   elif iterate_method == 'AB5':
      ds_outs = ds_orig.isel(T=slice(start-4,start+1))
      da_T_outs=ds_outs['Ttave']
      outs = {'tm1':(da_T_outs[-1,:,:,:]-da_T_outs[-2,:,:,:]),
              'tm2':(da_T_outs[-2,:,:,:]-da_T_outs[-3,:,:,:]),
              'tm3':(da_T_outs[-3,:,:,:]-da_T_outs[-4,:,:,:]),
              'tm4':(da_T_outs[-4,:,:,:]-da_T_outs[-5,:,:,:])}

   # Actually make the predictions!   
   for chunk in range(no_chunks):
      print('')
      print(chunk)
      chunk_start = size_chunk*chunk
      chunk_end = size_chunk*(chunk+1)+1
      tmp_pred, tmp_out, sponge_mask, mask, outs = it.iterator(data_name, run_vars, model, size_chunk, ds.isel(T=slice(chunk_start,chunk_end)), 
                                   density[chunk_start:chunk_end,:,:,:], init=init, model_type=model_type, method=iterate_method, outs=outs)
      if chunk == 0:
         # initialise these arrays, and keep both initial state and prediction for this first step
         Pred_Temp = tmp_pred 
         Pred_DelT = tmp_out 
      else:
         # concatenate on time axis. take prediction only - ignore istate
         Pred_Temp = np.concatenate( (Pred_Temp, tmp_pred[1:,:,:,:]), axis=0 )  
         Pred_DelT = np.concatenate( (Pred_DelT,  tmp_out[1:,:,:,:]), axis=0 ) 
   
      init = Pred_Temp[-1,:,:,:]
      # Save to array
      np.savez( pred_filename, np.array(Pred_Temp), np.array(Pred_DelT), np.array(mask) )
      # Save to netcdf file
      length = Pred_Temp.shape[0]
      nc_T[:] = ds['T'].data[:length]
      nc_Temp[:,:,:,:] = Pred_Temp
      nc_DelT[:,:,:,:] = Pred_DelT
      errors = Pred_Temp-da_T.data[:length,:,:,:]
      nc_Errors[:,:,:,:] = errors 
      nc_SpongeMask[:,:,:] = sponge_mask
      nc_Mask[:,:,:] = mask

#Pred_Temp, Pred_DelT, mask = np.load(pred_filename).values()
pred_data = np.load(pred_filename)
Pred_Temp = pred_data['arr_0']
Pred_DelT = pred_data['arr_1']
mask      = pred_data['arr_2']
length = Pred_Temp.shape[0]
da_T=da_T.data[:length,:,:,:]
errors = Pred_Temp-da_T[:,:,:,:]
print('Pred_Temp.shape')
print(Pred_Temp.shape)
mask_4d=np.broadcast_to( mask, errors.shape )

if np.isnan(Pred_DelT[1:,:,:,:][mask_4d[1:,:,:,:]==1]).any():
   print( 'Pred_DelT array contains a NaN at ' + str( np.argwhere(np.isnan(Pred_DelT[1:,:,:,:][mask_4d[1:,:,:,:]==1])) ) )
if np.isnan(da_T[1:,:,:,:]-da_T[:-1,:,:,:])[mask_4d[1:,:,:,:]==1].any():
   print( 'true_DelT array contains a NaN at ' + str( np.argwhere(np.isnan(da_T[1:,:,:,:]-da_T[:-1,:,:,:])[mask_4d[1:,:,:,:]==1]) ) )
