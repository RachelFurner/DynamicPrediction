#!/usr/bin/env python
# coding: utf-8


import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import os
import xarray as xr
import pickle

## Read in 'truth' from netcdf file
DIR = '/data/hpcdata/users/racfur/MITGCM_OUTPUT/'
exp = '20000yr_Windx1.00_mm_diag/'
filename=DIR+exp+'cat_tave_selectvars_5000yrs.nc'
print(filename)
ds   = xr.open_dataset(filename)

truth=ds['Ttave']

print(truth.shape)

## Load in linear regression model 

pkl_filename = "pickle_LRmodelTtave_tr200_halo1_pred1.pkl"
with open(pkl_filename, 'rb') as file:
    lr = pickle.load(file)

## Run iteratively to forecast
halo_size = 1
halo_list = (range(-halo_size, halo_size+1))

z_size = 42
y_size = 78
x_size = 11

for_len = 12*1000 # 1000 years

def lr_interator(init, num_steps):
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
                    predictions[t,z,y,x] = lr.predict(inputs)
    return predictions

predictions = lr_interator(truth[2400,:,:,:], for_len)

# Set persistence forecast
persistence = np.zeros((for_len))
persistence[:] = truth[2400,3,10,5]

## Plot it..
plt.plot(truth[2400:for_len+2400,3,10,5])
plt.plot(predictions[:,3,10,5])
plt.plot(persistence[:])
plt.legend(['truth', 'linear regression', 'persistence' ]);
plt.title('Timeseries at point 5,10,3, from the simulator (truth), and as predicted by the linear regressor and persistence');
plt.savefig('/data/hpcdata/users/racfur/DynamicPrediction/regression_plots/TruthvsPredict_Timeseries.png')
plt.show()
plt.close()

## Plot first 100 years..
end = 12*100
plt.plot(truth[2400:end+2400,3,10,5])
plt.plot(predictions[:end,3,10,5])
plt.plot(persistence[:end])
plt.legend(['truth', 'linear regression', 'persistence' ]);
plt.title('Timeseries at point 5,10,3, from the simulator (truth), and as predicted by the linear regressor and persistence');
plt.savefig('/data/hpcdata/users/racfur/DynamicPrediction/regression_plots/TruthvsPredict_Timeseries_100yrs.png')
plt.show()
plt.close()

## Plot first 10 years..
end = 12*10
plt.plot(truth[2400:end+2400,3,10,5])
plt.plot(predictions[:end,3,10,5])
plt.plot(persistence[:end])
plt.legend(['truth', 'linear regression', 'persistence' ]);
plt.title('Timeseries at point 5,10,3, from the simulator (truth), and as predicted by the linear regressor and persistence');
plt.savefig('/data/hpcdata/users/racfur/DynamicPrediction/regression_plots/TruthvsPredict_Timeseries_10yrs.png')
plt.show()
plt.close()

