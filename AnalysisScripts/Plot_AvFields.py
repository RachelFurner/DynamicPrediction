#!/usr/bin/env python
# coding: utf-8

print('import packages')
import sys
sys.path.append('/data/hpcdata/users/racfur/DynamicPrediction/code_git/')
from Tools import CreateDataName as cn
from Tools import Model_Plotting as rfplt

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from skimage.util import view_as_windows
import os
import xarray as xr
import pickle
from netCDF4 import Dataset

plt.rcParams.update({'font.size': 14})

#----------------------------
# Set variables for this run
#----------------------------
point = [ 5,10,5]

run_vars={'dimension':3, 'lat':True , 'lon':True, 'dep':True , 'current':True , 'bolus_vel':True , 'sal':True , 'eta':True , 'density':True , 'poly_degree':2}
model_type = 'lr'

time_step = '1mnth'
data_prefix=''
model_prefix = ''
exp_prefix = ''

if time_step == '1mnth':
   DIR  = '/data/hpcdata/users/racfur/MITGCM_OUTPUT/20000yr_Windx1.00_mm_diag/'
   MITGCM_filename=DIR+'cat_tave_2000yrs_SelectedVars_masked_withBolus.nc'
elif time_step == '24hrs':
   DIR  = '/data/hpcdata/users/racfur/MITGCM_OUTPUT/100yr_Windx1.00_FrequentOutput/'
   MITGCM_filename=DIR+'cat_tave_50yr_SelectedVars_masked_withBolus.nc'
else:
   print('ERROR!!! No suitable time step chosen!!')

#-----------
data_name = cn.create_dataname(run_vars)
data_name = time_step+'_'+data_prefix+data_name
model_name = model_prefix+data_name
exp_name = exp_prefix+model_name

rootdir = '/data/hpcdata/users/racfur/DynamicPrediction/'+model_type+'_Outputs/'

level = point[0]
y_coord = point[1]
x_coord = point[2]

#-------------------
# Read in land mask 
#-------------------
MITGCM_ds = xr.open_dataset(MITGCM_filename)
land_mask=MITGCM_ds['Mask'].values

#------------------------
print('reading in data')
#------------------------
data_filename=rootdir+'ITERATED_PREDICTION_ARRAYS/'+exp_name+'_AveragedSinglePredictions.nc'
ds = xr.open_dataset(data_filename)
da_AvTempError=ds['Av_Errors_Temp'].values

# mask data
AvTempError = np.where(land_mask==1, da_AvTempError, np.nan)

z_size = AvTempError.shape[0]
y_size = AvTempError.shape[1]
x_size = AvTempError.shape[2]

print('AvTempError.shape')
print(AvTempError.shape)

#--------------------------
# Plot spatial depth plots
#--------------------------
fig, ax, im = rfplt.plot_depth_fld(AvTempError[:,:,:], 'Temperature from Model', level, title=None, min_value=None, max_value=None, diff=True)
plt.savefig(rootdir+'PLOTS/'+exp_name+'_AvTempErrors_z'+str(level), bbox_inches = 'tight', pad_inches = 0.1)

#-----------------------
# Plot y-cross sections
#-----------------------
fig, ax, im = rfplt.plot_yconst_crss_sec(AvTempError[:,:,:], 'Temperature Predictions', y_coord, title=None, min_value=None, max_value=None, diff=True)
plt.savefig(rootdir+'PLOTS/'+exp_name+'_AvTempErrors_y'+str(y_coord), bbox_inches = 'tight', pad_inches = 0.1)

#-----------------------
# Plot x-cross sections
#-----------------------
fig, ax, im = rfplt.plot_xconst_crss_sec(AvTempError[:,:,:], 'Temperature Predictions', x_coord, title=None, min_value=None, max_value=None, diff=True)
plt.savefig(rootdir+'PLOTS/'+exp_name+'_AvTempErrors_x'+str(x_coord), bbox_inches = 'tight', pad_inches = 0.1)

