#!/usr/bin/env python
# coding: utf-8

print('import packages')
import sys
sys.path.append('../')
from Tools import Channel_Model_Plotting as ChnPlt

import numpy as np
import matplotlib.pyplot as plt
import os
import xarray as xr
from netCDF4 import Dataset

plt.rcParams.update({'font.size': 14})

#----------------------------
# Set variables for this run
#----------------------------
point = [ 2, 8, 6]

model_dir_name = 'MultiModel_IncLand_ksize3_UNet2dtransp'
epochs = '199'
for_len = '180'

#-----------

model_name = model_dir_name+'_'+epochs+'epochs'

rootdir = '../../../Channel_nn_Outputs/'+model_dir_name

#------------------------
#------------------------
iter_data_filename=rootdir+'/ITERATED_FORECAST/'+model_name+'_Forecast'+str(for_len)+'.nc'
iter_ds = xr.open_dataset(iter_data_filename)
da_True_Eta  = iter_ds['True_Eta']
da_True_U    = iter_ds['True_U']
da_True_V    = iter_ds['True_V']
da_Pred_Eta  = iter_ds['Pred_Eta']
da_Pred_U    = iter_ds['Pred_U']
da_Pred_V    = iter_ds['Pred_V']
da_X = iter_ds['X']
da_Y = iter_ds['Y']
da_Z = iter_ds['Z']

#--------------------------------------------------
# Plot timeseries of total ETA, summed over domain
#--------------------------------------------------
fig = plt.figure(figsize=(15 ,2))
ax = plt.subplot(111)
ax.plot(np.nansum(da_True_Eta.values, axis=(1,2)), label='Truth')
ax.plot(np.nansum(da_Pred_Eta.values, axis=(1,2)), label='Predicted')
ax.set_xlabel('No days')
ax.set_ylabel('SSH change')
ax.set_title('Change in total SSH over entire domain')
ax.legend()
plt.savefig(rootdir+'/ITERATED_FORECAST/'+model_name+'_conservation_Eta.png', bbox_inches = 'tight', pad_inches = 0.1)


fig = plt.figure(figsize=(15 ,2))
ax = plt.subplot(111)
ax.plot(np.nansum( np.add(np.square(da_True_U.values), np.square(da_True_V.values)), axis=(1,2,3) ), label='Truth')
ax.plot(np.nansum( np.add(np.square(da_Pred_U.values), np.square(da_Pred_V.values)), axis=(1,2,3) ), label='Predicted')
ax.set_xlabel('No days')
ax.set_ylabel('TKE change')
ax.set_title('Change in total kinetic energy over entire domain')
ax.legend()
plt.savefig(rootdir+'/ITERATED_FORECAST/'+model_name+'_conservation_TKE.png', bbox_inches = 'tight', pad_inches = 0.1)


