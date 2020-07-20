# Script to calculate errors from a model, and plot these
# as a histogram, and spatially

import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('/data/hpcdata/users/racfur/DynamicPrediction/code_git/')
from Tools import CreateDataName as cn
from Tools import Iterator as it
from Tools import Model_Plotting as rfplt
from Tools import AssessModel as am
import xarray as xr
import pickle
import netCDF4 as nc4

plt.rcParams.update({'font.size': 14})

#----------------------------
# Set variables for this run
#----------------------------
run_vars={'dimension':3, 'lat':True , 'lon':True, 'dep':True , 'current':True , 'bolus_vel':True , 'sal':True , 'eta':True , 'density':True , 'poly_degree':2}
model_type = 'lr'

time_step = '24hrs'
data_prefix='DensLayers_'
model_prefix = ''
exp_prefix = ''

calc_predictions = True 
iter_length = 500  # in months/days

if time_step == '1mnth':
   DIR  = '/data/hpcdata/users/racfur/MITGCM_OUTPUT/20000yr_Windx1.00_mm_diag/'
   data_filename=DIR+'cat_tave_2000yrs_SelectedVars_masked_withBolus.nc'
   density_file = DIR+'DensityData.npy'
elif time_step == '24hrs':
   DIR  = '/data/hpcdata/users/racfur/MITGCM_OUTPUT/100yr_Windx1.00_FrequentOutput/'
   data_filename=DIR+'cat_tave_50yr_SelectedVars_masked_withBolus.nc'
   density_file = DIR+'DensityData.npy'
else:
   print('ERROR!!! No suitable time step given!!')

#-----------
data_name = data_prefix+cn.create_dataname(run_vars)+'_'+time_step
model_name = model_prefix+data_name
exp_name = exp_prefix+model_name

#-----------------------------------------------------------------------
# Read in netcdf file for 'truth', inputs for other variables and shape
#-----------------------------------------------------------------------
print('reading in ds')
ds = xr.open_dataset(data_filename)

Temp_truth = ds['Ttave'][:iter_length+1,:,:,:].data
print('Temp_truth.shape')
print(Temp_truth.shape)

#Read in density from separate array - it was outputted using MITGCM levels code, so not in ncfile
density = np.load( density_file, mmap_mode='r' ) 
print('density.shape')
print(density.shape) 

mask = ds['Mask'].values

#-------------------
# Read in the model
#-------------------
pkl_filename = '/data/hpcdata/users/racfur/DynamicPrediction/'+model_type+'_Outputs/MODELS/'+model_name+'_pickle.pkl'
print(pkl_filename)
with open(pkl_filename, 'rb') as file:
   print('opening '+pkl_filename)
   model = pickle.load(file)

#---------------------
# Set up netcdf array
#---------------------
print('set up netcdf file')

nc_file = nc4.Dataset('/data/hpcdata/users/racfur/DynamicPrediction/'+model_type+'_Outputs/ITERATED_PREDICTION_ARRAYS/'+exp_name+'_AveragedSinglePredictions.nc','w', format='NETCDF4') #'w' stands for write
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
nc_PredictedTemp = nc_file.createVariable('PredictedTemp', 'f4', ('T', 'Z', 'Y', 'X'))
nc_PredictedDeltaT = nc_file.createVariable('PredictedDeltaT', 'f4', ('T', 'Z', 'Y', 'X'))
nc_Errors = nc_file.createVariable('Errors', 'f4', ('T', 'Z', 'Y', 'X'))
nc_av_Errors = nc_file.createVariable('Av_Errors', 'f4', ('Z', 'Y', 'X'))
nc_wtd_av_Errors = nc_file.createVariable('Weighted_Av_Errors', 'f4', ('Z', 'Y', 'X'))
nc_AbsErrors = nc_file.createVariable('AbsErrors', 'f4', ('T', 'Z', 'Y', 'X'))
nc_av_AbsErrors = nc_file.createVariable('Av_AbsErrors', 'f4', ('Z', 'Y', 'X'))
nc_wtd_av_AbsErrors = nc_file.createVariable('Weighted_Av_AbsErrors', 'f4', ('Z', 'Y', 'X'))

# Fill some variables - rest done during iteration steps
nc_T[:] = ds['T'].data[1:]
nc_Z[:] = ds['Z'].data
nc_Y[:] = ds['Y'].data
nc_X[:] = ds['X'].data

#--------------------------------------------------------------------------------------
# Call iterator multiple times to get many predictions for entire grid for one time 
# step ahead, which can then be averaged to get a spatial pattern of temporally 
# averaged errors i.e. each entry here is the result of a single prediction using the 
# 'truth' as inputs, rather than iteratively predicting through time
#--------------------------------------------------------------------------------------
print('get predictions')
#pred_filename = '/data/hpcdata/users/racfur/DynamicPrediction/'+model_type+'_Outputs/ITERATED_PREDICTION_ARRAYS/'+exp_name+'_AveragedSinglePredictions.npz'

if calc_predictions:

    # Array to hold predicted values of Temperature
    predictedTemp = np.zeros((Temp_truth.shape))
    predictedTemp[:,:,:,:] = np.nan  # ensure any 'unforecastable' points display as NaNs 

    # Array to hold de-normalised outputs from model, i.e. DeltaTemp (before applying any AB-timestepping methods)
    predictedDelT = np.zeros((Temp_truth.shape))
    predictedDelT[:,:,:,:] = np.nan  # ensure any 'unforecastable' points display as NaNs 

    for t in range(1,iter_length+1): # ignore first value, as we can't calculate this - we'd need ds at t-1, leave as NaN so index's match with Temp_truth.
        print(t)
        # Note iterator returns the initial condition plus the number of iterations, so skip time slice 0
        predT_temp, predDelT_temp, sponge_mask, mask, outs = it.iterator( data_name, run_vars, model, 1, ds.isel(T=slice(t-1,t+1)),
                                                                          density[t-1:t+1,:,:,:], method='AB1' )
        predictedTemp[t,:,:,:] = predT_temp[1,:,:,:]
        predictedDelT[t,:,:,:] = predDelT_temp[1,:,:,:]

    #Save as arrays
    #np.savez(pred_filename, np.array(predictedTemp), np.array(predictedDelT))
#predictedTemp, predictedDelT = np.load(pred_filename).values()    

#------------------
# Calculate errors
#------------------
print('calc errors')
DelT_truth = Temp_truth[1:,:,:,:]-Temp_truth[:-1,:,:,:]
Errors = (predictedDelT[1:,:,:,:] - DelT_truth[:,:,:,:])
AbsErrors = np.abs(predictedDelT[1:,:,:,:] - DelT_truth[:,:,:,:])

# Average temporally and spatially 
##t_av_Temp_errors = np.nanmean(Temp_errors, axis=0)
##s_av_Temp_errors = np.nanmean(Temp_errors[:,1:-1,1:-3,1:-2], axis=(1,2,3))  # Remove points not being predicted (boundaries) from this
Time_Av_DelT_Truth  = np.nanmean(DelT_truth , axis=0)

Time_Av_Errors = np.nanmean(Errors, axis=0)
Spacial_Av_Errors = np.nanmean(Errors[:,1:-1,1:-3,1:-2], axis=(1,2,3))  # Remove points not being predicted (boundaries) from this
Weighted_Av_Error = np.where( Time_Av_DelT_Truth==0., np.nan, Time_Av_Errors/Time_Av_DelT_Truth )

Time_Av_AbsErrors = np.nanmean(AbsErrors, axis=0)
Spacial_Av_AbsErrors = np.nanmean(AbsErrors[:,1:-1,1:-3,1:-2], axis=(1,2,3))  # Remove points not being predicted (boundaries) from this
Weighted_Av_AbsError = np.where( Time_Av_DelT_Truth==0., np.nan, Time_Av_AbsErrors/Time_Av_DelT_Truth )


# Save to netcdf
print('save to netcdf')
nc_PredictedTemp[:,:,:,:] = predictedTemp[1:]
nc_PredictedDeltaT[:,:,:,:] = predictedDelT[1:]
nc_Errors[:,:,:,:] = Errors
nc_av_Errors[:,:,:] = Time_Av_Errors 
nc_wtd_av_Errors[:,:,:] = Weighted_Av_Error
nc_AbsErrors[:,:,:,:] = AbsErrors
nc_av_AbsErrors[:,:,:] = Time_Av_AbsErrors 
nc_wtd_av_AbsErrors[:,:,:] = Weighted_Av_AbsError
nc_file.close()

#-----------------
# Plot timeseries 
#-----------------
print('plot timeseries of the error')
fig = plt.figure(figsize=(14 ,3))

ax1=plt.subplot(121)
ax1.plot(Spacial_Av_Errors)
ax1.set_ylabel('Errors')
ax1.set_xlabel('No of months')

ax2=plt.subplot(122)
ax2.plot(Spacial_Av_AbsErrors)
ax2.set_ylabel('Absolute Errors')
ax2.set_xlabel('No of months')

plt.tight_layout()
plt.subplots_adjust(hspace = 0.5, left=0.05, right=0.95, bottom=0.15, top=0.90)
plt.savefig('../../'+model_type+'_Outputs/PLOTS/'+model_name+'/'+exp_name+'_timeseries_Av_Error', bbox_inches = 'tight', pad_inches = 0.1)
