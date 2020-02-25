# Script to calculate errors from a model, and plot these
# as a histogram, and spatially

import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('/data/hpcdata/users/racfur/DynamicPrediction/code_git/')
from Tools import CreateDataName as cn
from Tools import Iterator as it
from Tools import Model_Plotting as rfplt
import xarray as xr
import pickle

#----------------------------
# Set variables for this run
#----------------------------
run_vars={'dimension':3, 'lat':True , 'lon':True, 'dep':True , 'current':True , 'sal':True , 'eta':True , 'poly_degree':2}
model_type = 'lr'
exp_prefix = ''

iter_length = 12000  # in months

calc_predictions = False

#-----------
data_name = cn.create_dataname(model_type, run_vars)

exp_name = exp_prefix+data_name 

#---------------------------------------------------
# Read in netcdf file for 'truth' and get shape
#---------------------------------------------------
print('reading in ds')
DIR  = '/data/hpcdata/users/racfur/MITGCM_OUTPUT/20000yr_Windx1.00_mm_diag/'
data_filename=DIR+'cat_tave_5000yrs_SelectedVars_masked.nc'
ds = xr.open_dataset(data_filename)
truth = ds['Ttave'][:iter_length+1,:,:,:].data

print('truth.shape')
print(truth.shape)

#-------------------
# Read in the model
#-------------------
pkl_filename = '/data/hpcdata/users/racfur/DynamicPrediction/'+model_type+'_Outputs/MODELS/pickle_'+model_type+'_'+exp_name+'.pkl'
with open(pkl_filename, 'rb') as file:
   print('opening '+pkl_filename)
   model = pickle.load(file)

#---------------------
# Set up netcdf array
#---------------------
import netCDF4 as nc4

nc_file = nc4.Dataset('/data/hpcdata/users/racfur/DynamicPrediction/'+model_type+'_Outputs/ITERATED_PREDICTION_ARRAYS/'+exp_name+'_SinglePredictions.nc','w', format='NETCDF4') #'w' stands for write
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
nc_Predictions = nc_file.createVariable('Predictions', 'f4', ('T', 'Z', 'Y', 'X'))
nc_Errors = nc_file.createVariable('Prediction-Truth', 'f4', ('T', 'Z', 'Y', 'X'))
nc_Tav_Errors = nc_file.createVariable('Averaged Prediction-Truth', 'f4', ('Z', 'Y', 'X'))

# Fill some variables - rest done during iteration steps
nc_T[:] = ds['T'].data[1:]
nc_Z[:] = ds['Z'].data
nc_Y[:] = ds['Y'].data
nc_X[:] = ds['X'].data

#-----------------------------------------------------------------------------------
# Call iterator to get prediction for entire grid for one time step ahead at a time
# i.e. each entry is the result of a single prediction using the 'truth' as inputs,
# rather than iteratively predicting through time
#-----------------------------------------------------------------------------------
pred_filename = '/data/hpcdata/users/racfur/DynamicPrediction/'+model_type+'_Outputs/ITERATED_PREDICTION_ARRAYS/'+model_type+'_'+exp_name+'_SinglePredictions.npy'
if calc_predictions:
    predictions = np.zeros((truth.shape))
    predictions[:,:,:,:] = np.nan  # ensure any 'unforecastable' points display as NaNs 
    for t in range(1,iter_length+1): # ignore first value, as we can't calcualte this - we'd need ds at t-1    
        print(t)
        # Note iterator returns the initial condition plus the number of iterations, so skip time slice 0
        predictions[t,:,:,:] = it.interator( exp_name, run_vars, model, 1, ds.isel(T=slice(t-1,t+1)) )[1:,:,:,:]
    #Save as array
    np.save(pred_filename, np.array(predictions))
predictions = np.load(pred_filename)    

#----------------------------
# Calculate and plot things!
#----------------------------

errors = (predictions[1:,:,:,:] - truth[1:,:,:,:])

# Average temporally and spatially 
t_av_errors = np.nanmean(errors, axis=0)
s_av_errors = np.nanmean(errors[:,1:-1,1:-3,1:-2], axis=(1,2,3))  # Remove points not being predicted (boundaries) from this

# Plot a histogram of the predictions minus the truth 
fig = rfplt.Plot_Histogram(errors[:,1:-1,1:-3,1:-2], 100)  # Remove points not being predicted (boundaries) from this
plt.savefig('../../'+model_type+'_Outputs/PLOTS/'+model_type+'_'+exp_name+'_y_ybar_hist', bbox_inches = 'tight', pad_inches = 0.1)

# Save to netcdf
nc_Predictions[:,:,:,:] = predictions[1:]
nc_Errors[:,:,:,:] = errors
nc_Tav_Errors[:,:,:] = t_av_errors 

# Plot s_av_errors as a timeseries
fig = rfplt.plt_timeseries([], errors.shape[0], {'spatially averaged errors':s_av_errors}, ylim=None)
plt.savefig('../../'+model_type+'_Outputs/PLOTS/'+model_type+'_'+exp_name+'_ErrorTimeSeries', bbox_inches = 'tight', pad_inches = 0.1)
