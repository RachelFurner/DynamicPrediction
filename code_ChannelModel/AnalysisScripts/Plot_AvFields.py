#!/usr/bin/env python
# coding: utf-8

print('import packages')
import sys
sys.path.append('../')
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
point = [ 2, 8, 6]

run_vars={'dimension':3, 'lat':True , 'lon':True, 'dep':True , 'current':True , 'bolus_vel':True , 'sal':True , 'eta':True , 'density':True , 'poly_degree':2}
model_type = 'lr'

time_step = '24hrs'
data_prefix=''
exp_prefix = ''

if model_type == 'nn':
   inputDim = 26106
   outputDim = 1
   no_layers = 1
   no_nodes = 100
   model_prefix = 'MSE_'+str(no_layers)+'layers_'+str(no_nodes)+'nodes_lr0.001_batch4092_'
else:
   model_prefix = 'alpha.0001_'

DIR  = '/data/hpcdata/users/racfur/MITGCM_OUTPUT/100yr_Windx1.00_FrequentOutput/'
MITGCM_filename=DIR+'cat_tave_50yr_SelectedVars_masked_withBolus.nc'

#-----------
data_name = cn.create_dataname(run_vars)
data_name = data_prefix+data_name+'_'+time_step 
model_name = model_prefix+data_name
exp_name = exp_prefix+model_name

rootdir = '../../'+model_type+'_Outputs/'

level = point[0]
y_coord = point[1]
x_coord = point[2]

#-------------------
# Read in land mask 
#-------------------
MITGCM_ds = xr.open_dataset(MITGCM_filename)
land_mask = MITGCM_ds['Mask'].values
da_T = MITGCM_ds['Ttave'].values

#------------------------
print('reading in data')
#------------------------
data_filename=rootdir+'ITERATED_PREDICTION_ARRAYS/'+exp_name+'_AveragedSinglePredictions.nc'
ds = xr.open_dataset(data_filename)
da_Av_Error=ds['Av_Errors'].values
da_Av_AbsError=ds['Av_AbsErrors'].values
da_wtd_Av_Error=ds['Weighted_Av_Errors'].values

# mask data
Av_Error = np.where(land_mask==1, da_Av_Error, np.nan)
Av_AbsError = np.where(land_mask==1, da_Av_AbsError, np.nan)
wtd_Av_Error = np.where(land_mask==1, da_wtd_Av_Error, np.nan)

z_size = Av_Error.shape[0]
y_size = Av_Error.shape[1]
x_size = Av_Error.shape[2]

print('Av_Error.shape')
print(Av_Error.shape)

#--------------------------
# Plot spatial depth plots
#--------------------------
fig, ax, im = rfplt.plot_depth_fld(Av_Error[:,:,:], 'Averaged Errors', level,
                                   MITGCM_ds['X'].values, MITGCM_ds['Y'].values, MITGCM_ds['Z'].values,
                                   title=None, min_value=None, max_value=None, diff=True)
plt.savefig(rootdir+'PLOTS/'+model_name+'/'+exp_name+'_AvErrors_z'+str(level)+'.png', bbox_inches = 'tight', pad_inches = 0.1)

fig, ax, im = rfplt.plot_depth_fld(Av_AbsError[:,:,:], 'Averaged Absolute Errors', level,
                                   MITGCM_ds['X'].values, MITGCM_ds['Y'].values, MITGCM_ds['Z'].values,
                                   title=None, min_value=None, max_value=None, cmap='Reds')
plt.savefig(rootdir+'PLOTS/'+model_name+'/'+exp_name+'_AvAbsErrors_z'+str(level)+'.png', bbox_inches = 'tight', pad_inches = 0.1)

fig, ax, im = rfplt.plot_depth_fld(wtd_Av_Error[:,:,:], 'Weighted Averaged Errors', level,
                                   MITGCM_ds['X'].values, MITGCM_ds['Y'].values, MITGCM_ds['Z'].values,
                                   title=None, min_value=None, max_value=None, diff=True)
plt.savefig(rootdir+'PLOTS/'+model_name+'/'+exp_name+'_WtdAvErrors_z'+str(level)+'.png', bbox_inches = 'tight', pad_inches = 0.1)

#-----------------------
# Plot y-cross sections
#-----------------------
fig, ax, im = rfplt.plot_yconst_crss_sec(Av_Error[:,:,:], 'Averaged Errors', y_coord, 
                                         MITGCM_ds['X'].values, MITGCM_ds['Y'].values, MITGCM_ds['Z'].values,
                                         title=None, min_value=None, max_value=None, diff=True)
plt.savefig(rootdir+'PLOTS/'+model_name+'/'+exp_name+'_AvErrors_y'+str(y_coord)+'.png', bbox_inches = 'tight', pad_inches = 0.1)

fig, ax, im = rfplt.plot_yconst_crss_sec(Av_AbsError[:,:,:], 'Averaged Absolute Errors', y_coord, 
                                         MITGCM_ds['X'].values, MITGCM_ds['Y'].values, MITGCM_ds['Z'].values,
                                         title=None, min_value=None, max_value=None, cmap='Reds')
plt.savefig(rootdir+'PLOTS/'+model_name+'/'+exp_name+'_AvAbsErrors_y'+str(y_coord)+'.png', bbox_inches = 'tight', pad_inches = 0.1)

fig, ax, im = rfplt.plot_yconst_crss_sec(wtd_Av_Error[:,:,:], 'Weighted Averaged Errors', y_coord, 
                                         MITGCM_ds['X'].values, MITGCM_ds['Y'].values, MITGCM_ds['Z'].values,
                                         title=None, min_value=None, max_value=None, diff=True)
plt.savefig(rootdir+'PLOTS/'+model_name+'/'+exp_name+'_WtdAvErrors_y'+str(y_coord)+'.png', bbox_inches = 'tight', pad_inches = 0.1)

#-----------------------
# Plot x-cross sections
#-----------------------
fig, ax, im = rfplt.plot_xconst_crss_sec(Av_Error[:,:,:], 'Averaged Errors', x_coord,
                                         MITGCM_ds['X'].values, MITGCM_ds['Y'].values, MITGCM_ds['Z'].values,
                                         title=None, min_value=None, max_value=None, diff=True)
plt.savefig(rootdir+'PLOTS/'+model_name+'/'+exp_name+'_AvErrors_x'+str(x_coord)+'.png', bbox_inches = 'tight', pad_inches = 0.1)

fig, ax, im = rfplt.plot_xconst_crss_sec(Av_AbsError[:,:,:], 'Averaged Absolute Errors', x_coord,
                                         MITGCM_ds['X'].values, MITGCM_ds['Y'].values, MITGCM_ds['Z'].values,
                                         title=None, min_value=None, max_value=None, cmap='Reds')
plt.savefig(rootdir+'PLOTS/'+model_name+'/'+exp_name+'_AvAbsErrors_x'+str(x_coord)+'.png', bbox_inches = 'tight', pad_inches = 0.1)

fig, ax, im = rfplt.plot_xconst_crss_sec(wtd_Av_Error[:,:,:], 'Weighted Averaged Errors', x_coord, 
                                         MITGCM_ds['X'].values, MITGCM_ds['Y'].values, MITGCM_ds['Z'].values,
                                         title=None, min_value=None, max_value=None, diff=True)
plt.savefig(rootdir+'PLOTS/'+model_name+'/'+exp_name+'_WtdAvErrors_x'+str(x_coord)+'.png', bbox_inches = 'tight', pad_inches = 0.1)

