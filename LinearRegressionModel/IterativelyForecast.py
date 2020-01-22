#!/usr/bin/env python
# coding: utf-8

print('import packages')

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import os
import xarray as xr
import pickle

for_len_yrs = 100 # forecast length in years
#exp_names = ['T_3dLatDepVarsINT_halo1_pred1']
exp_names = ['T_3dLatDepVarsINT_halo1_pred1']
start = 200
#xy = 4  # 2d halo
xy = 13  # 3d halo 

#-----------------
# Define iterator
#-----------------
print('define iterator')
def interator(exp_name, model, init, start, num_steps, ds):
    x_size = init.shape[2]
    y_size = init.shape[1]
    z_size = init.shape[0]
    print('z_size, y_size, x_size ; '+str(z_size)+', '+str(y_size)+', '+str(x_size))
    
    print('read in other variables') 
    da_T=ds['Ttave'][start:start+num_steps,:,:,:]
    da_S=ds['Stave'][start:start+num_steps,:,:,:]
    da_U=ds['uVeltave'][start:start+num_steps,:,:,:]
    da_V=ds['vVeltave'][start:start+num_steps,:,:,:]
    da_eta=ds['ETAtave'][start:start+num_steps,:,:]
    da_lat=ds['Y'][:]
    da_depth=ds['Z'][:]
    print('da_T.shape ; '+str(da_T.shape))
    print('da_S.shape ; '+str(da_S.shape))
    print('da_U.shape ; '+str(da_U.shape))
    print('da_V.shape ; '+str(da_V.shape))
    print('da_eta.shape ; '+str(da_eta.shape))
    print('da_lat.shape ; '+str(da_lat.shape))
    print('da_depth.shape ; '+str(da_depth.shape))

    #Read in mean and std to normalise inputs
    print('read in info to normalise data')
    norm_file=open('/data/hpcdata/users/racfur/DynamicPrediction/RegressionOutputs/STATS/normalising_parameters_'+exp_name+'.txt',"r")
    count = len(norm_file.readlines(  ))
    print('count ; '+str(count))
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

    predictions = np.zeros((num_steps, z_size, y_size, x_size))
    print('predictions.shape ; ' + str(predictions.shape))
    print('input_mean.shape ; ' + str(input_mean.shape))
    print('input_std.shape ; ' + str(input_std.shape))
    print('output_mean ; ' + str(output_mean))
    print('output_std ; ' + str(output_std))
    predictions[0,:,:,:] = init
    # Set all boundary and near land data to match 'da_T'
    predictions[1:,:,:,0:2]             = da_T[1:num_steps,:,:,0:2]
    predictions[1:,:,:,x_size-2:x_size] = da_T[1:num_steps,:,:,x_size-2:x_size]
    predictions[1:,:,0:2,:]             = da_T[1:num_steps,:,0:2,:]
    predictions[1:,:,y_size-3:y_size,:] = da_T[1:num_steps,:,y_size-3:y_size,:]
    predictions[1:,0:2,:,:]             = da_T[1:num_steps,0:2,:,:]
    predictions[1:,z_size-2:z_size,:,:] = da_T[1:num_steps,z_size-2:z_size,:,:]
    for t in range(1,num_steps):
        #print(t)
        for x in range(2,x_size-2):
            for y in range(2,y_size-3):
                for z in range(2,z_size-2):

                    input_temp=[]
                    #[input_temp.append(predictions[t-1,z,y+y_offset,x+x_offset]) for x_offset in halo_list for y_offset in halo_list]
                    [input_temp.append(predictions[t-1,z+z_offset,y+y_offset,x+x_offset]) for x_offset in halo_list for y_offset in halo_list for z_offset in halo_list]
                    [input_temp.append(da_S[t-1,z+z_offset,y+y_offset,x+x_offset]) for x_offset in halo_list for y_offset in halo_list for z_offset in halo_list]
                    [input_temp.append(da_U[t-1,z+z_offset,y+y_offset,x+x_offset]) for x_offset in halo_list for y_offset in halo_list for z_offset in halo_list]
                    [input_temp.append(da_V[t-1,z+z_offset,y+y_offset,x+x_offset]) for x_offset in halo_list for y_offset in halo_list for z_offset in halo_list]
                    [input_temp.append(da_eta[t-1,y+y_offset,x+x_offset]) for x_offset in halo_list for y_offset in halo_list]
                    input_temp.append(da_lat[y])
                    input_temp.append(da_depth[z])

                    input_temp = np.array(input_temp)
                    input_temp = input_temp.astype(float)
                    # normalise inputs
                    inputs = np.divide( np.subtract(input_temp, input_mean), input_std)
                    inputs = inputs.reshape(1, -1)
                    predictions[t,z,y,x] = model.predict(inputs)
                    #predictions[t,z,y,x] = inputs[:,xy]
                    # de-normalise inputs
                    predictions[t,z,y,x] = predictions[t,z,y,x] * input_std[0,xy] + input_mean[0,xy]

    return predictions
#----------------------------------
# Read in netcdf file and get shape etc
#----------------------------------

print('reading in ds')
DIR  = '/data/hpcdata/users/racfur/MITGCM_OUTPUT/20000yr_Windx1.00_mm_diag/'
data_filename=DIR+'cat_tave_500yrs_SelectedVars.nc'
ds   = xr.open_dataset(data_filename)
da_T=ds['Ttave'][start:start+for_len_yrs*12]   

z_size = da_T.shape[1]
y_size = da_T.shape[2]
x_size = da_T.shape[3]


print('da_T.shape')
print(da_T.shape)

#----------------------------------
# Predict for a variety of points
#----------------------------------
x_points=[5,5,5]
y_points=[5,10,15]
z_points=[3,3,3]
#x_points=[5]
#y_points=[5]
#z_points=[5]

#---------------------------
# Set persistence forecasts
#---------------------------
print('Set perstistence forecast')
persistence = np.zeros((for_len_yrs*12,len(x_points)))  # assume 1 month forecast period - doesn't matter if this is actually too long
for point in range(len(x_points)):
   persistence[:,point] = da_T[0,z_points[point],y_points[point],x_points[point]]

#------------------------------------------------------------------------------------------------------------
# Loop over different experiments and models (different training/forecast set ups, and different model types)
#------------------------------------------------------------------------------------------------------------

for exp in exp_names:
   ## Set variables for this specific forecast
   #halo_size = int(exp[10])
   #predict_step = int(exp[16])
   halo_size = 1
   predict_step = 1

   halo_list = (range(-halo_size, halo_size+1))
   for_len = int(for_len_yrs/predict_step * 12)

   pkl_filename = '/data/hpcdata/users/racfur/DynamicPrediction/RegressionOutputs/MODELS/pickle_lr_'+exp+'.pkl'
   with open(pkl_filename, 'rb') as file:
      print('opening '+pkl_filename)
      model = pickle.load(file)

   #----------------------
   # Make the predictions
   #----------------------
   print('Call iterator and make the predictions')
   pred_filename = '/data/hpcdata/users/racfur/DynamicPrediction/RegressionOutputs/DATASETS/'+exp+'_predictions.npy'
   predictions = interator(exp, model, da_T[0,:,:,:], start, int(for_len), ds)
   np.save(pred_filename, np.array(predictions))
   predictions = np.load(pred_filename)
   print('predictions.shape')
   print(predictions.shape)
   #for i in range(34):
   #   fig = plt.figure(figsize=(8,20))
   #   plt.pcolormesh(predictions[i,15,:,:])
   #   plt.xlabel('longitude')
   #   plt.ylabel('latitude')
   #   plt.colorbar()
   #   plt.show()
   #   fig.savefig('forecast'+str(i)+'.png')
   #   plt.close()


   #----------------------------------
   # Plot for a variety of points
   #----------------------------------

   for point in range(len(x_points)):
      x = x_points[point]
      y = y_points[point]
      z = z_points[point]
   
      #------------------------------------------------
      # Plot predictions against da_T and persistence
      #------------------------------------------------
      print('Plot it')
      fig = plt.figure(figsize=(15 ,8))

      #ax1=plt.subplot(311)
      ax1=plt.subplot(211)
      plt_len=int(10/predict_step * 12)
      ax1.plot(da_T[:plt_len,z,y,x])
      ax1.plot(predictions[:plt_len,z,y,x])
      ax1.plot(persistence[:plt_len,point])
      ax1.set_ylabel('Temperature')
      ax1.set_xlabel('No of months')
      ax1.set_title('10 years')
      ax1.legend(['da_T (truth)', 'lr model', 'persistence' ])

      #ax2=plt.subplot(312)
      #plt_len=int(100/predict_step * 12)
      #ax2.plot(da_T[:plt_len,z,y,x])
      #ax2.plot(predictions[:plt_len,z,y,x])
      #ax2.plot(persistence[:plt_len,point])
      #ax2.set_ylabel('Temperature')
      #ax2.set_xlabel('No of forecast steps')
      #ax2.set_title('100 years')
      #ax2.legend(['da_T (truth)', 'lr model', 'persistence' ])

      #ax3=plt.subplot(313)
      ax3=plt.subplot(212)
      plt_len=for_len
      ax3.plot(da_T[:plt_len,z,y,x])
      ax3.plot(predictions[:plt_len,z,y,x])
      ax3.plot(persistence[:plt_len,point])
      ax3.set_ylabel('Temperature')
      ax3.set_xlabel('No of months')
      ax3.set_title(str(for_len_yrs)+' years')
      ax3.legend(['da_T (truth)', 'lr model', 'persistence' ])

      #plt.subplots_adjust(hspace=0.4, top=0.9, bottom=0.12, left=0.08, right=0.85)
      plt.subplots_adjust(hspace=0.5)
      plt.suptitle('Timeseries at point '+str(x)+','+str(y)+','+str(z)+' from the simulator (da_T),\n and as predicted by the regression model, and persistence', fontsize=15, x=0.47, y=0.98)

      plt.savefig('/data/hpcdata/users/racfur/DynamicPrediction/RegressionOutputs/PLOTS/lr_'+exp+'_'+str(x)+'.'+str(y)+'.'+str(z)+'_TruthvsPredict_Timeseries.png', bbox_inches = 'tight', pad_inches = 0.1)
      plt.close()

