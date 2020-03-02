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
import netCDF4 as nc4

#----------------------------
# Set variables for this run
#----------------------------
run_vars={'dimension':3, 'lat':True , 'lon':True, 'dep':True , 'current':True , 'sal':True , 'eta':True , 'poly_degree':2}
model_type = 'lr'
exp_prefix = ''

calc_predictions = False
iter_length = 12000  # in months

#-----------
data_name = cn.create_dataname(run_vars)
exp_name = exp_prefix+data_name 

#---------------------------------------------------
# Read in netcdf file for 'truth' and get shape
#---------------------------------------------------
print('reading in ds')
DIR  = '/data/hpcdata/users/racfur/MITGCM_OUTPUT/20000yr_Windx1.00_mm_diag/'
data_filename=DIR+'cat_tave_2000yrs_SelectedVars_masked.nc'
ds = xr.open_dataset(data_filename)
Temp_truth = ds['Ttave'][:iter_length+1,:,:,:].data

print('Temp_truth.shape')
print(Temp_truth.shape)

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
nc_TempErrors = nc_file.createVariable('PredTemp-Truth', 'f4', ('T', 'Z', 'Y', 'X'))
nc_Tav_Errors = nc_file.createVariable('Averaged_PredTemp-Truth', 'f4', ('Z', 'Y', 'X'))
nc_PredictedDeltaT = nc_file.createVariable('PredictedDeltaT', 'f4', ('T', 'Z', 'Y', 'X'))
nc_DeltaTErrors = nc_file.createVariable('PredDelT-Truth', 'f4', ('T', 'Z', 'Y', 'X'))
nc_DelTav_Errors = nc_file.createVariable('Averaged_PredDelT-Truth', 'f4', ('Z', 'Y', 'X'))

# Fill some variables - rest done during iteration steps
nc_T[:] = ds['T'].data[1:]
nc_Z[:] = ds['Z'].data
nc_Y[:] = ds['Y'].data
nc_X[:] = ds['X'].data

#-------------------------------------------------------------------------------------
# Call iterator multiple time to get many predictions for entire grid for one time 
# step ahead, which can then be averaged to get a spatial pattern of temporally 
# averaged errors i.e. each entry here is the result of a single prediction using the 
# 'truth' as inputs, rather than iteratively predicting through time
#--------------------------------------------------------------------------------------
print('get predictions')
pred_filename = '/data/hpcdata/users/racfur/DynamicPrediction/'+model_type+'_Outputs/ITERATED_PREDICTION_ARRAYS/'+model_type+'_'+exp_name+'_AveragedSinglePredictions.npz'

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
        predT_temp, predDelT_temp = it.interator( exp_name, run_vars, model, 1, ds.isel(T=slice(t-1,t+1)) )
        predictedTemp[t,:,:,:] = predT_temp[1:,:,:,:]
        predictedDelT[t,:,:,:] = predDelT_temp[1:,:,:,:]
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

# Save to netcdf
print('save to netcdf')
nc_PredictedTemp[:,:,:,:] = predictedTemp[1:]
nc_TempErrors[:,:,:,:] = Temp_errors
nc_Tav_Errors[:,:,:] = t_av_Temp_errors 
nc_PredictedDeltaT[:,:,:,:] = predictedDelT[1:]
nc_DeltaTErrors[:,:,:,:] = DelT_errors
nc_DelTav_Errors[:,:,:] = t_av_DelT_errors 
nc_file.close()

#-------------
# Plot errors
#-------------
print('plot errors etc')
# Plot a histogram of Temperature predictions minus the truth 
fig = rfplt.Plot_Histogram(Temp_errors[:,1:-1,1:-3,1:-2], 100)  # Remove points not being predicted (boundaries) from this
plt.savefig('../../'+model_type+'_Outputs/PLOTS/'+model_type+'_'+exp_name+'_Temp_yminusybar_hist', bbox_inches = 'tight', pad_inches = 0.1)

# Plot a histogram of deltaT predictions minus the true deltaT
fig = rfplt.Plot_Histogram(DelT_errors[:,1:-1,1:-3,1:-2], 100)  # Remove points not being predicted (boundaries) from this
plt.savefig('../../'+model_type+'_Outputs/PLOTS/'+model_type+'_'+exp_name+'_DeltaT_yminusybar_hist', bbox_inches = 'tight', pad_inches = 0.1)

# Plot spatially averaged errors as a timeseries
fig = rfplt.plt_timeseries([], Temp_errors.shape[0], {'spatially averaged Temp errors':s_av_Temp_errors}, ylim=None)
plt.savefig('../../'+model_type+'_Outputs/PLOTS/'+model_type+'_'+exp_name+'_TempErrorTimeSeries', bbox_inches = 'tight', pad_inches = 0.1)

fig = rfplt.plt_timeseries([], Temp_errors.shape[0], {'spatially averaged DelT errors':s_av_DelT_errors}, ylim=None)
plt.savefig('../../'+model_type+'_Outputs/PLOTS/'+model_type+'_'+exp_name+'_DelTErrorTimeSeries', bbox_inches = 'tight', pad_inches = 0.1)

# Plot histograms of deltaT from truth and predictions
fig = rfplt.Plot_Histogram(DelT_truth[1:,1:-1,1:-3,1:-2], 100)  # Remove points not being predicted (boundaries) from this
plt.savefig('../../'+model_type+'_Outputs/PLOTS/'+model_type+'_'+exp_name+'_deltaTtruth_hist', bbox_inches = 'tight', pad_inches = 0.1)

fig = rfplt.Plot_Histogram(predictedDelT[1:,1:-1,1:-3,1:-2], 100)  # Remove points not being predicted (boundaries) from this
plt.savefig('../../'+model_type+'_Outputs/PLOTS/'+model_type+'_'+exp_name+'_deltaTpredicted_hist', bbox_inches = 'tight', pad_inches = 0.1)

##########
# Redo training script stats and plots with this large dataset....
print('plot some truth vs prediction plots')
fig = plt.figure(figsize=(20,9.4))
   
ax1 = fig.add_subplot(121)
ax1.scatter(Temp_truth[1:,:,:,:], predictedTemp[1:,:,:,:], edgecolors=(0, 0, 0))
#ax1.plot([bottom, top], [bottom, top], 'k--', lw=1)
ax1.set_xlabel('Truth')
ax1.set_ylabel('Predicted')
ax1.set_title('Temp')
#ax1.set_xlim(bottom, top)
#ax1.set_ylim(bottom, top)

ax2 = fig.add_subplot(122)
ax2.scatter(DelT_truth, predictedDelT[1:,:,:,:], edgecolors=(0, 0, 0))
#ax2.plot([bottom, top], [bottom, top], 'k--', lw=1)
ax2.set_xlabel('Truth')
ax2.set_ylabel('Predicted')
ax2.set_title('DeltaT')
#ax2.set_xlim(bottom, top)
#ax2.set_ylim(bottom, top)

outdir = '/data/hpcdata/users/racfur/DynamicPrediction/'+model_type+'_Outputs/'
plt.savefig(outdir+'/PLOTS/'+model_type+'_'+exp_name+'_predictedVtruth_FullDataset.png', bbox_inches = 'tight', pad_inches = 0.1)

print('calc stats and print to file')
# calculate stats
Temp_mse = metrics.mean_squared_error(Temp_truth[1:,:,:,:], predictedTemp[1:,:,:,:])
DelT_mse = metrics.mean_squared_error(DelT_truth, predictedDelT[1:,:,:,:])

# Print to file

stats_filename = outdir+'STATS/'+model_type+'_'+exp_name+'_FullDataset.txt'
stats_file=open(stats_filename,"w")

stats_file.write('\n')
stats_file.write('Temperature: \n')
stats_file.write('%30s %.10f; \n' % (' '+exp_name+' rms score', np.sqrt(Temp_mse)))
stats_file.write('\n')
stats_file.write('--------------------------------------------------------')
stats_file.write('Delta Temperature: \n')
stats_file.write('%30s %.10f; \n' % (' '+exp_name+' rms score', np.sqrt(DelT_mse)))
stats_file.write('\n')
stats_file.close()
