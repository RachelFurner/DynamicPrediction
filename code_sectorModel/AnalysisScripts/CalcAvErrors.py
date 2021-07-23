# Script written by Rachel Furner
# Makes a number of one-step predictions for the entire grid, 
# these are then temporally averaged and saved in a netcdf file,
# and spatially averaged and a time series of errors evolving is
# plotted

import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('../Tools')
import CreateDataName as cn
import Iterator as it
import Model_Plotting as rfplt
import AssessModel as am
import xarray as xr
import pickle
import netCDF4 as nc4

plt.rcParams.update({'font.size': 14})


## WARNING: this code is not set up to work with StepSize other than 1

#----------------------------
# Set variables for this run
#----------------------------
run_vars={'dimension':3, 'lat':True , 'lon':True, 'dep':True , 'current':True , 'bolus_vel':True , 'sal':True , 'eta':True , 'density':True , 'poly_degree':2, 'StepSize':1}

data_prefix=''
exp_prefix = ''

model_prefix = 'alpha.001_'

calc_predictions = True 
no_points = 500  # no days of predictions being calculated
skip_rate = 14   # Take every skip_rate point in time, so as to avoid looking at heavily correlated points

DIR  = '/data/hpcdata/users/racfur/MITgcm/verification/MundaySectorConfig_2degree/runs/100yrs/'
data_filename=DIR+'mnc_test_0002/cat_tave.nc'
density_file = DIR+'DensityData.npy'

#-----------------------------------------
# Calc other variables - shouldn't change
#-----------------------------------------
data_name = data_prefix+cn.create_dataname(run_vars)
model_name = model_prefix+data_name
exp_name = exp_prefix+model_name

#-----------------------------------------------------------------------
# Read in netcdf file for 'truth', inputs for other variables and shape
#-----------------------------------------------------------------------
print('reading in ds')
ds = xr.open_dataset(data_filename)
Temp_truth = ds['Ttave'].data
Length_Truth = Temp_truth.shape[0]
if (no_points*skip_rate) > (Length_Truth*.7):
    print('Be aware some data is coming from both Training and Val portions of run')
    print('Training set ends at time = '+str(Length_Truth*.7) )
    print('We are using data up to time = '+str(no_points*skip_rate) )
else:
    print('All data from Training portion of run')
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
pkl_filename = '../../../lr_Outputs/MODELS/'+model_name+'_pickle.pkl'
print(pkl_filename)
with open(pkl_filename, 'rb') as file:
    print('opening '+pkl_filename)
    model = pickle.load(file)
   
#---------------------
# Set up netcdf array
#---------------------
print('set up netcdf file')

nc_file = nc4.Dataset('../../../lr_Outputs/ITERATED_PREDICTION_ARRAYS/'+exp_name+'_AveragedSinglePredictions.nc','w', format='NETCDF4') #'w' stands for write
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
nc_Cor_Coef = nc_file.createVariable('Cor_Coef', 'f4', ('Z', 'Y', 'X'))

# Fill some variables - rest done during iteration steps
nc_Z[:] = ds['Z'].data
nc_Y[:] = ds['Y'].data
nc_X[:] = ds['X'].data
nc_T[:] = ds['T'].data[1:no_points*skip_rate+1:skip_rate]

#--------------------------------------------------------------------------------------
# Call iterator multiple times to get many predictions for entire grid for one time 
# step ahead, which can then be averaged to get a spatial pattern of temporally 
# averaged errors i.e. each entry here is the result of a single prediction using the 
# 'truth' as inputs, rather than iteratively predicting through time
#--------------------------------------------------------------------------------------
print('get predictions')
pred_filename = '../../../lr_Outputs/ITERATED_PREDICTION_ARRAYS/'+exp_name+'_AveragedSinglePredictions.npz'

if calc_predictions:

    # Array to hold predicted values of Temperature
    predictedTemp = np.zeros((no_points,Temp_truth.shape[1],Temp_truth.shape[2],Temp_truth.shape[3]))
    predictedTemp[:,:,:,:] = np.nan  # ensure any 'unforecastable' points display as NaNs 

    # Array to hold de-normalised outputs from model, i.e. DeltaTemp (before applying any AB-timestepping methods)
    predictedDelT = np.zeros((no_points,Temp_truth.shape[1],Temp_truth.shape[2],Temp_truth.shape[3]))
    predictedDelT[:,:,:,:] = np.nan  # ensure any 'unforecastable' points display as NaNs 

    for t in range(no_points): 
        print(t)
        predT_temp, predDelT_temp, dummy, dummy, dummy = it.iterator( data_name, run_vars, model, 1, ds.isel(T=slice(t*skip_rate,t*skip_rate+2)),
                                                                          density[t*skip_rate:t*skip_rate+2,:,:,:], method='AB1' )
        # Note iterator returns the initial condition plus the number of iterations, so skip time slice 0
        predictedTemp[t,:,:,:] = predT_temp[1,:,:,:]
        predictedDelT[t,:,:,:] = predDelT_temp[1,:,:,:]

    #Save as arrays
    np.savez(pred_filename, np.array(predictedTemp), np.array(predictedDelT))

# Load in arrays of predictions
predicted_data = np.load(pred_filename)
predictedTemp = predicted_data['arr_0']
predictedDelT = predicted_data['arr_1']

#------------------
# Calculate errors
#------------------
print('calc errors')
DelT_truth = ( Temp_truth[1:no_points*skip_rate+1:skip_rate,:,:,:] - 
               Temp_truth[0:no_points*skip_rate:skip_rate,:,:,:] )
Errors = (predictedDelT[:,:,:,:] - DelT_truth[:,:,:,:])
AbsErrors = np.abs(predictedDelT[:,:,:,:] - DelT_truth[:,:,:,:])

# Average temporally and spatially 
Time_Av_DelT_Truth  = np.nanmean(DelT_truth , axis=0)

Time_Av_Errors = np.nanmean(Errors, axis=0)
Spacial_Av_Errors = np.nanmean(Errors[:,1:-1,1:-3,1:-2], axis=(1,2,3))  # Remove points not being predicted (boundaries) from this
Weighted_Av_Error = np.where( Time_Av_DelT_Truth==0., np.nan, Time_Av_Errors/Time_Av_DelT_Truth )

Time_Av_AbsErrors = np.nanmean(AbsErrors, axis=0)
Spacial_Av_AbsErrors = np.nanmean(AbsErrors[:,1:-1,1:-3,1:-2], axis=(1,2,3))  # Remove points not being predicted (boundaries) from this
Weighted_Av_AbsError = np.where( Time_Av_DelT_Truth==0., np.nan, Time_Av_AbsErrors/Time_Av_DelT_Truth )

# Calculate Spatial Correlation Coefficients
DelT_cor_coef = np.zeros((DelT_truth.shape[1], DelT_truth.shape[2], DelT_truth.shape[3]))
for i in range(DelT_truth.shape[1]):
   for j in range(DelT_truth.shape[2]):
      for k in range(DelT_truth.shape[3]):
         DelT_cor_coef[i,j,k] = np.corrcoef(predictedDelT[:,i,j,k], DelT_truth[:,i,j,k])[0,1]
          
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
nc_Cor_Coef[:,:,:] = DelT_cor_coef 
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
plt.savefig('../../../lr_Outputs/PLOTS/'+model_name+'/'+exp_name+'_timeseries_Av_Error.png', bbox_inches = 'tight', pad_inches = 0.1)

