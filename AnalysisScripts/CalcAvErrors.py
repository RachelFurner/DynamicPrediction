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
run_vars={'dimension':2, 'lat':True , 'lon':True, 'dep':True , 'current':True , 'bolus_vel':False, 'sal':True , 'eta':True , 'density':False, 'poly_degree':2}
model_type = 'lr'

time_step = '24hrs'
data_prefix=''
model_prefix = ''
exp_prefix = ''

calc_predictions = True 
iter_length = 500  # in months/days

if time_step == '1mnth':
   DIR  = '/data/hpcdata/users/racfur/MITGCM_OUTPUT/20000yr_Windx1.00_mm_diag/'
   data_filename=DIR+'cat_tave_2000yrs_SelectedVars_masked_withBolus.nc'
elif time_step == '24hrs':
   DIR  = '/data/hpcdata/users/racfur/MITGCM_OUTPUT/100yr_Windx1.00_FrequentOutput/'
   data_filename=DIR+'cat_tave_50yr_SelectedVars_masked_withBolus.nc'
else:
   print('ERROR!!! No suitable time step given!!')

#-----------
data_name = cn.create_dataname(run_vars)
data_name = time_step+'_'+data_prefix+data_name
model_name = model_prefix+data_name
exp_name = exp_prefix+model_name

#---------------------------------------------------
# Read in netcdf file for 'truth' and get shape
#---------------------------------------------------
print('reading in ds')
ds = xr.open_dataset(data_filename)
Temp_truth = ds['Ttave'][:iter_length+1,:,:,:].data
mask = ds['Mask'].values

print('Temp_truth.shape')
print(Temp_truth.shape)

#-------------------
# Read in the model
#-------------------
pkl_filename = '/data/hpcdata/users/racfur/DynamicPrediction/'+model_type+'_Outputs/MODELS/pickle_'+model_name+'.pkl'
print(pkl_filename)
with open(pkl_filename, 'rb') as file:
   print('opening '+pkl_filename)
   model = pickle.load(file)

## run a few simple tests to check against iterator:
#in1 = np.zeros((1, 7260))
#in2 = np.ones((1, 7260))
#
#out1 = model.predict(in1)
#out2 = model.predict(in2)
#
#print('some quick test values:')
#print(out1)
#print(out2)

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
nc_TempErrors = nc_file.createVariable('Errors_Temp', 'f4', ('T', 'Z', 'Y', 'X'))
nc_Tav_Errors = nc_file.createVariable('Av_Errors_Temp', 'f4', ('Z', 'Y', 'X'))
nc_PredictedDeltaT = nc_file.createVariable('PredictedDeltaT', 'f4', ('T', 'Z', 'Y', 'X'))
nc_DeltaTErrors = nc_file.createVariable('Errors_DelT', 'f4', ('T', 'Z', 'Y', 'X'))
nc_DelTav_Errors = nc_file.createVariable('Av_Errors_DelT', 'f4', ('Z', 'Y', 'X'))
nc_wtd_DelT_av_Errors = nc_file.createVariable('Weighted_Av_Errors_DelT', 'f4', ('Z', 'Y', 'X'))

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
pred_filename = '/data/hpcdata/users/racfur/DynamicPrediction/'+model_type+'_Outputs/ITERATED_PREDICTION_ARRAYS/'+exp_name+'_AveragedSinglePredictions.npz'

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
        predT_temp, predDelT_temp, sponge_mask, mask, outs = it.iterator( data_name, run_vars, model, 1, ds.isel(T=slice(t-1,t+1)), method='AB1' )
        predictedTemp[t,:,:,:] = predT_temp[1,:,:,:]
        predictedDelT[t,:,:,:] = predDelT_temp[1,:,:,:]

    #Save as arrays
    np.savez(pred_filename, np.array(predictedTemp), np.array(predictedDelT))
predictedTemp, predictedDelT = np.load(pred_filename).values()    

#------------------
# Calculate errors
#------------------
print('calc errors')
Temp_errors = (predictedTemp[1:,:,:,:] - Temp_truth[1:,:,:,:])
DelT_truth = Temp_truth[1:,:,:,:]-Temp_truth[:-1,:,:,:]
DelT_errors = (predictedDelT[1:,:,:,:] - DelT_truth[:,:,:,:])

# Average temporally and spatially 
t_av_Temp_errors = np.nanmean(Temp_errors, axis=0)
s_av_Temp_errors = np.nanmean(Temp_errors[:,1:-1,1:-3,1:-2], axis=(1,2,3))  # Remove points not being predicted (boundaries) from this
t_av_DelT_errors = np.nanmean(DelT_errors, axis=0)
s_av_DelT_errors = np.nanmean(DelT_errors[:,1:-1,1:-3,1:-2], axis=(1,2,3))  # Remove points not being predicted (boundaries) from this
t_av_DelT_truth  = np.nanmean(DelT_truth , axis=0)


weighted_av_DelT_error = np.where( t_av_DelT_truth==0., np.nan, t_av_DelT_errors/t_av_DelT_truth )

# Save to netcdf
print('save to netcdf')
nc_PredictedTemp[:,:,:,:] = predictedTemp[1:]
nc_TempErrors[:,:,:,:] = Temp_errors
nc_Tav_Errors[:,:,:] = t_av_Temp_errors 
nc_PredictedDeltaT[:,:,:,:] = predictedDelT[1:]
nc_DeltaTErrors[:,:,:,:] = DelT_errors
nc_DelTav_Errors[:,:,:] = t_av_DelT_errors 
nc_wtd_DelT_av_Errors[:,:,:] = weighted_av_DelT_error
nc_file.close()

#-----------------
# Plot timeseries 
#-----------------
print('plot timeseries of the error')
fig = plt.figure(figsize=(14 ,3))

ax=plt.subplot(111)
ax.plot(s_av_DelT_errors)
ax.set_ylabel('Errors')
ax.set_xlabel('No of months')
plt.tight_layout()
plt.subplots_adjust(hspace = 0.5, left=0.05, right=0.95, bottom=0.15, top=0.90)
plt.savefig('../../'+model_type+'_Outputs/PLOTS/'+exp_name+'_timeseries_Av_DelT_Error', bbox_inches = 'tight', pad_inches = 0.1)

##----------------------------------------------------------
## Redo stats and scatter plots with this large dataset....
##----------------------------------------------------------
#print('Plot scatter plots')
#am.plot_results(model_type, exp_name, DelT_truth[:,1:-1,1:-3,1:-2], predictedDelT[1:,1:-1,1:-3,1:-2], name='denorm_DeltaT')
#
#print('Calc stats and print to file')
#am.get_stats(model_type, exp_name, name1='DeltaT', truth1=DelT_truth[:,1:-1,1:-3,1:-2], exp1=predictedDelT[1:,1:-1,1:-3,1:-2], name='DeltaT_denorm')
