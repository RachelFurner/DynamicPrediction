#!/usr/bin/env python
# coding: utf-8

print('import packages')

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import os
import xarray as xr
import pickle


#-----------------
# Define iterator
#-----------------
print('define iterator')
def interator(model, init, num_steps, truth):
    x_size = init.shape[2]
    y_size = init.shape[1]
    z_size = init.shape[0]
    predictions = np.zeros((num_steps, z_size, y_size, x_size))
    predictions[0,:,:,:] = init
    # Set all boundary and near land data to match 'truth' for now
    predictions[:,:,:,0] = truth[:num_steps,:,:,0]
    predictions[:,:,:,x_size-1] = truth[:num_steps,:,:,x_size-1]
    predictions[:,:,:,x_size-2] = truth[:num_steps,:,:,x_size-2]
    predictions[:,:,0,:] = truth[:num_steps,:,0,:]
    predictions[:,:,y_size-1,:] = truth[:num_steps,:,y_size-1,:]
    predictions[:,:,y_size-2,:] = truth[:num_steps,:,y_size-2,:]
    predictions[:,:,y_size-3,:] = truth[:num_steps,:,y_size-3,:]
    for t in range(1,num_steps):
        for x in range(1,x_size-2):
            for y in range(1,y_size-3):
                for z in range(z_size):
                    inputs=[]
                    inputs.append([predictions[t-1,z,y+y_offset,x+x_offset] for x_offset in halo_list for y_offset in halo_list])
                    predictions[t,z,y,x] = model.predict(inputs)
    return predictions

#----------------------------------
# Read in 'truth' from netcdf file
#----------------------------------
print('reading in truth')
DIR = '/data/hpcdata/users/racfur/MITGCM_OUTPUT/'
exp = '20000yr_Windx1.00_mm_diag/'
filename=DIR+exp+'cat_tave_selectvars_5000yrs.nc'
ds   = xr.open_dataset(filename)
truth=ds['Ttave']

z_size = truth.shape[0]
y_size = truth.shape[1]
x_size = truth.shape[2]

print(x_size, y_size, z_size)

#----------------------------------
# Predict for a variety of points
#----------------------------------
x_points=[5,5,5]
y_points=[5,10,15]
z_points=[3,3,3]
#x_points=[5]
#y_points=[5]
#z_points=[3]

#---------------------------
# Set persistence forecasts
#---------------------------
print('Set perstistence forecast')
for_len_yrs = 1000 # forecast length in years
persistence = np.zeros((len(x_points),for_len_yrs*12))  # assume 1 month forecast period - doesn't matter if this is actually too long
for point in range(len(x_points)):
   persistence[point,:] = truth[2400,z_points[point],y_points[point],x_points[point]]

#------------------------------------------------------------------------------------------------------------
# Loop over different experiments and models (different training/forecast set ups, and different model types)
#------------------------------------------------------------------------------------------------------------
exp_names = ['tr200_halo1_pred1', 'tr200_halo2_pred1', 'tr200_halo1_pred3', 'tr400_halo1_pred1']
#exp_names = ['tr200_halo1_pred1']
model_types = ['lr', 'rfr', 'gbr', 'mlp']
#model_types = ['lr']

for exp in exp_names:
   ## Set variables for this specific forecast
   halo_size = int(exp[10])
   halo_list = (range(-halo_size, halo_size+1))
   predict_step = int(exp[16])
   for_len = int(for_len_yrs/predict_step * 12)

   for model_type in model_types:
      pkl_filename = 'PICKLED_MODELS/pickle_'+model_type+'_Ttave_'+exp+'.pkl'
      with open(pkl_filename, 'rb') as file:
         model = pickle.load(file)

      #----------------------
      # Make the predictions
      #----------------------
      print('Call iterator and make the predictions')
      predictions = interator(model, truth[2400,:,:,:], int(for_len), truth)
      #predictions = truth[2400:2400+for_len,:,:,:]

      #----------------------------------
      # Predict for a variety of points
      #----------------------------------

      for point in range(len(x_points)):
         x = x_points[point]
         y = y_points[point]
         z = z_points[point]
   
         #------------------------------------------------
         # Plot predictions against truth and persistence
         #------------------------------------------------
         print('Plot it')
         fig = plt.figure(figsize=(20,10))

         ax1=plt.subplot(311)
         plt_len=for_len
         plt.plot(truth[2400:plt_len+2400,z,y,x])
         plt.plot(predictions[:plt_len,z,y,x])
         plt.plot(persistence[point,:plt_len])
         plt.ylabel('Temperature')
         plt.xlabel('No of forecast steps')
         plt.title(str(for_len_yrs)+' years')
         plt.legend(['truth', model_type+' model', 'persistence' ])
         #plt.legend(['truth', model_type+' model'])

         ax2=plt.subplot(312)
         plt_len=int(100/predict_step * 12)
         plt.plot(truth[2400:plt_len+2400,z,y,x])
         plt.plot(predictions[:plt_len,z,y,x])
         plt.plot(persistence[point,:plt_len])
         plt.ylabel('Temperature')
         plt.xlabel('No of forecast steps')
         plt.title('100 years')
         plt.legend(['truth', model_type+' model', 'persistence' ])
         #plt.legend(['truth', model_type+' model'])

         ax2=plt.subplot(313)
         plt_len=int(10/predict_step * 12)
         plt.plot(truth[2400:plt_len+2400,z,y,x])
         plt.plot(predictions[:plt_len,z,y,x])
         plt.plot(persistence[point,:plt_len])
         plt.ylabel('Temperature')
         plt.xlabel('No of forecast steps')
         plt.title('10 years')
         plt.legend(['truth', model_type+' model', 'persistence' ])
         #plt.legend(['truth', model_type+' model'])

         plt.subplots_adjust(hspace=0.4, top=0.9, bottom=0.12, left=0.08, right=0.85)
         plt.suptitle('Timeseries at point '+str(x)+','+str(y)+','+str(z)+' from the simulator (truth),\n and as predicted by the '+model_type+' model, and persistence', fontsize=15, x=0.47, y=0.98)
         #plt.suptitle('Timeseries at point '+str(x)+','+str(y)+','+str(z)+' from the simulator (truth),\n and as predicted by the '+model_type+' model', fontsize=15, x=0.47, y=0.98)

         plt.savefig('/data/hpcdata/users/racfur/DynamicPrediction/regression_plots/'+model_type+'_'+exp+'_'+str(x)+'.'+str(y)+'.'+str(z)+'_TruthvsPredict_Timeseries.png')
         plt.close()

