#!/usr/bin/env python
# coding: utf-8

print('import packages')
import sys
sys.path.append('/data/hpcdata/users/racfur/DynamicPrediction/code_git/Tools/')
import RF_routines as RF

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
import os
import xarray as xr
import pickle

#----------------------------
# Set variables for this run
#----------------------------
run_vars={'dimension':3, 'lat':True, 'dep':True, 'current':True, 'sal':True, 'eta':True, 'poly_degree':2}
model_type = 'lr'

for_len_yrs = 10 # forecast length in years

no_chunks = 50

#x_points=[5,5,5]
#y_points=[5,10,15]
#z_points=[3,3,3]
x_points=[5]
y_points=[5]
z_points=[5]

start = 200
halo_size = 1
predict_step = 1
halo_list = (range(-halo_size, halo_size+1))
for_len = int(for_len_yrs/predict_step * 12)

exp_name = RF.create_expname(model_type, run_vars)

#-----------------
# Define iterator
#-----------------
print('define iterator')
def interator(exp_name, model, init, ch_start, num_steps, ds):
    x_size = init.shape[2]
    y_size = init.shape[1]
    z_size = init.shape[0]
    
    print('read in other variables') 
    da_T=ds['Ttave'][ch_start:ch_start+num_steps,:,:,:]
    da_S=ds['Stave'][ch_start:ch_start+num_steps,:,:,:]
    da_U=ds['uVeltave'][ch_start:ch_start+num_steps,:,:,:]
    da_V=ds['vVeltave'][ch_start:ch_start+num_steps,:,:,:]
    da_eta=ds['ETAtave'][ch_start:ch_start+num_steps,:,:]
    da_lat=ds['Y'][:]
    da_depth=ds['Z'][:]

    #Read in mean and std to normalise inputs - should move this outside of the loop!
    print('read in info to normalise data')
    norm_file=open('/data/hpcdata/users/racfur/DynamicPrediction/RegressionOutputs/STATS/normalising_parameters_'+exp_name+'.txt',"r")
    count = len(norm_file.readlines(  ))
    input_mean=[]
    input_std =[]
    norm_file.seek(0)
    for i in range( int( (count-4)/4) ):
       a_str = norm_file.readline()
       a_str = norm_file.readline() ;  input_mean.append(a_str.split())
       a_str = norm_file.readline()
       a_str = norm_file.readline() ;  input_std.append(a_str.split())
    a_str = norm_file.readline() 
    a_str = norm_file.readline() ;  output_mean = float(a_str.split()[0])
    a_str = norm_file.readline() 
    a_str = norm_file.readline() ;  output_std = float(a_str.split()[0])
    norm_file.close()
    input_mean = np.array(input_mean).astype(float)
    input_std  = np.array(input_std).astype(float)
    input_mean = input_mean.reshape(1,input_mean.shape[0])
    input_std  = input_std.reshape(1,input_std.shape[0])
    
    out_t   = np.zeros((z_size, y_size, x_size))
    out_tm1 = np.zeros((z_size, y_size, x_size))
    out_tm2 = np.zeros((z_size, y_size, x_size))

    predictions = np.zeros((num_steps, z_size, y_size, x_size))
    print('predictions.shape ; ' + str(predictions.shape))
    # Set initial conditions to match init
    predictions[0,:,:,:] = init[:,:,:]
    # Set all boundary and near land data to match 'da_T'
    predictions[1:,:,:,0:3]             = da_T[1:num_steps,:,:,0:3]
    predictions[1:,:,:,x_size-4:x_size] = da_T[1:num_steps,:,:,x_size-4:x_size]
    predictions[1:,:,0:3,:]             = da_T[1:num_steps,:,0:3,:]
    predictions[1:,:,y_size-4:y_size,:] = da_T[1:num_steps,:,y_size-4:y_size,:]
    predictions[1:,0:3,:,:]             = da_T[1:num_steps,0:3,:,:]
    predictions[1:,z_size-4:z_size,:,:] = da_T[1:num_steps,z_size-4:z_size,:,:]

    for t in range(1,num_steps):
        out_tm2[:,:,:] = out_tm1[:,:,:]
        out_tm1[:,:,:] = out_t[:,:,:]
        print(t)
        for x in range(3,x_size-4):
            for y in range(3,y_size-4):
                for z in range(3,z_size-4):
                   #print(t)
                   #print(x,y,z)
                   input_temp = []
                   if run_vars['dimension'] == 2:
                      [input_temp.append(predictions[t-1,z,y+y_offset,x+x_offset]) for x_offset in halo_list for y_offset in halo_list]
                      if run_vars['sal']:
                         [input_temp.append(da_S[t-1,z,y+y_offset,x+x_offset]) for x_offset in halo_list for y_offset in halo_list]
                      if run_vars['current']:
                         [input_temp.append(da_U[t-1,z,y+y_offset,x+x_offset]) for x_offset in halo_list for y_offset in halo_list]
                         [input_temp.append(da_V[t-1,z,y+y_offset,x+x_offset]) for x_offset in halo_list for y_offset in halo_list]
                   elif run_vars['dimension'] == 3:
                      [input_temp.append(predictions[t-1,z+z_offset,y+y_offset,x+x_offset]) for x_offset in halo_list for y_offset in halo_list for z_offset in halo_list]
                      if run_vars['sal']:
                         [input_temp.append(da_S[t-1,z+z_offset,y+y_offset,x+x_offset]) for x_offset in halo_list for y_offset in halo_list for z_offset in halo_list]
                      if run_vars['current']:
                         [input_temp.append(da_U[t-1,z+z_offset,y+y_offset,x+x_offset]) for x_offset in halo_list for y_offset in halo_list for z_offset in halo_list]
                         [input_temp.append(da_V[t-1,z+z_offset,y+y_offset,x+x_offset]) for x_offset in halo_list for y_offset in halo_list for z_offset in halo_list]
                   if run_vars['eta']:
                      [input_temp.append(da_eta[t-1,y+y_offset,x+x_offset]) for x_offset in halo_list for y_offset in halo_list]
                   if run_vars['lat']:
                      input_temp.append(da_lat[y])
                   if run_vars['dep']:
                      input_temp.append(da_depth[z])
  
                   # convert to array and make 2-d
                   input_temp = np.array(input_temp)
                   input_temp = input_temp.astype(float)
                   input_temp = input_temp.reshape(1, -1)

                   if run_vars['poly_degree'] > 1: 
                      # Add polynomial combinations of the features
                      polynomial_features = PolynomialFeatures(degree=run_vars['poly_degree'], interaction_only=True, include_bias=False)
                      input_temp = polynomial_features.fit_transform(input_temp)

                   # normalise inputs
                   inputs = np.divide( np.subtract(input_temp, input_mean), input_std)
                   
                   #predict and then de-normalise outputs
                   out_t[z,y,x] = model.predict(inputs) * output_std + output_mean
        if t==0: 
           deltaT = out_t
        if t==1: 
           deltaT = 1.5*out_t-0.5*out_tm1
        if t>1: 
           deltaT = (23.0/12.0)*out_t-(4.0/3.0)*out_tm1+(5.0/12.0)*out_tm2
        
        predictions[t,:,:,:] = predictions[t-1,:,:,:] + ( deltaT )

    return predictions

#-----------------------------------------------
# Read in netcdf file for 'truth' and get shape
#-----------------------------------------------
print('reading in ds')
DIR  = '/data/hpcdata/users/racfur/MITGCM_OUTPUT/20000yr_Windx1.00_mm_diag/'
data_filename=DIR+'cat_tave_5000yrs_SelectedVars.nc'
ds   = xr.open_dataset(data_filename)
da_T=ds['Ttave'][start:start+for_len_yrs*12]   

z_size = da_T.shape[1]
y_size = da_T.shape[2]
x_size = da_T.shape[3]

print('da_T.shape')
print(da_T.shape)

#---------------------------
# Set persistence forecasts
#---------------------------
print('Set perstistence forecast')
persistence = np.zeros((for_len_yrs*12,len(x_points)))  # assume 1 month forecast period - doesn't matter if this is actually too long
for point in range(len(x_points)):
   persistence[:,point] = da_T[0,z_points[point],y_points[point],x_points[point]]

#----------------------
# Make the predictions
#----------------------
# Read in the model
pkl_filename = '/data/hpcdata/users/racfur/DynamicPrediction/RegressionOutputs/MODELS/pickle_'+model_type+'_'+exp_name+'.pkl'
with open(pkl_filename, 'rb') as file:
   print('opening '+pkl_filename)
   model = pickle.load(file)

# Call iterator for chunks of data at a time, saving interim results in arrays, so if the
# predictions become unstable and lead to Nan's etc we still have some of the trajectory saved
print('Call iterator and make the predictions')
predictions = np.zeros((for_len+1, z_size, y_size, x_size))
pred_filename = '/data/hpcdata/users/racfur/DynamicPrediction/DATASETS/'+model_type+'_'+exp_name+'_predictions.npy'
size_chunk = int(for_len/no_chunks)
print(size_chunk)
init = da_T[0,:,:,:]
for chunk in range(no_chunks):
    print('')
    print(chunk)
    print('')
    print(predictions[size_chunk*chunk:size_chunk*(chunk+1)+1,:,:,:].shape)
    chunk_start = start + size_chunk*chunk
    predictions[size_chunk*chunk:size_chunk*(chunk+1)+1,:,:,:] = interator(exp_name, model, init, chunk_start, size_chunk+1, ds)
    print('')
    print('')
    print('')
    print('saving to file...!')
    print('')
    print('')
    print('')
    init = predictions[size_chunk*(chunk+1),:,:,:]
    np.save(pred_filename, np.array(predictions))
predictions = np.load(pred_filename)
print('predictions.shape')
print(predictions.shape)

#---------------------------------------------------------------------------
# Plot the predictions against da_T and persistence for a variety of points
#---------------------------------------------------------------------------
for point in range(len(x_points)):
   x = x_points[point]
   y = y_points[point]
   z = z_points[point]

   fig = plt.figure(figsize=(15 ,8))

   ax1=plt.subplot(311)
   #ax1=plt.subplot(211)
   plt_len=int(10/predict_step * 12)
   ax1.plot(da_T[:plt_len,z,y,x])
   ax1.plot(predictions[:plt_len,z,y,x])
   ax1.plot(persistence[:plt_len,point])
   ax1.set_ylabel('Temperature')
   ax1.set_xlabel('No of months')
   ax1.set_title('10 years')
   ax1.legend(['da_T (truth)', 'lr model', 'persistence' ])

   ax2=plt.subplot(312)
   plt_len=int(100/predict_step * 12)
   ax2.plot(da_T[:plt_len,z,y,x])
   ax2.plot(predictions[:plt_len,z,y,x])
   ax2.plot(persistence[:plt_len,point])
   ax2.set_ylabel('Temperature')
   ax2.set_xlabel('No of forecast steps')
   ax2.set_title('100 years')
   ax2.legend(['da_T (truth)', 'lr model', 'persistence' ])

   ax3=plt.subplot(313)
   #ax3=plt.subplot(212)
   plt_len=for_len
   ax3.plot(da_T[:plt_len,z,y,x])
   ax3.plot(predictions[:plt_len,z,y,x])
   ax3.plot(persistence[:plt_len,point])
   ax3.set_ylabel('Temperature')
   ax3.set_xlabel('No of months')
   ax3.set_title(str(for_len_yrs)+' years')
   ax3.legend(['da_T (truth)', 'lr model', 'persistence' ])

   #plt.subplots_adjust(hspace=0.4, top=0.9, bottom=0.12, left=0.08, right=0.85)
   plt.subplots_adjust(hspace=0.4)
   plt.suptitle('Timeseries at point '+str(x)+','+str(y)+','+str(z)+' from the simulator (da_T),\n and as predicted by the regression model, and persistence', fontsize=15, x=0.47, y=0.98)

   plt.savefig('/data/hpcdata/users/racfur/DynamicPrediction/RegressionOutputs/PLOTS/'+model_type+'_'+exp_name+'_'+str(x)+'.'+str(y)+'.'+str(z)+'_TruthvsPredict_Timeseries.png', bbox_inches = 'tight', pad_inches = 0.1)
   plt.close()

