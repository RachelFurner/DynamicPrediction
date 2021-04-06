#!/usr/bin/env python
# coding: utf-8
# 
# Script written by Rachel Furner
# Plots the fields created by CalcAvErrors.py, giving cross sections
# which show spatial distribution of average errors from linear regressor 
# predictions

print('import packages')
import sys
sys.path.append('../Tools')
import CreateDataName as cn
import Model_Plotting as rfplt

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

plt.rcParams.update({'font.size': 10})
plt.rc('font', family='sans serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')

#----------------------------
# Set variables for this run
#----------------------------
point = [ 2, 8, 6]

run_vars={'dimension':2, 'lat':True , 'lon':True, 'dep':True , 'current':True , 'bolus_vel':True , 'sal':True , 'eta':True , 'density':True , 'poly_degree':2}

data_prefix=''
exp_prefix = ''
model_prefix = 'alpha.001_'

DIR  = '/data/hpcdata/users/racfur/MITgcm/verification/MundaySectorConfig_2degree/runs/100yrs/mnc_test_0002/'
MITGCM_filename=DIR+'cat_tave.nc'

#-----------------------------------------------
# Calc other variables - these shouldn't change
#-----------------------------------------------
data_name = data_prefix + cn.create_dataname(run_vars)
model_name = model_prefix+data_name
exp_name = exp_prefix+model_name
cntrl_name = exp_prefix+model_prefix+data_prefix+'3dLatLonDepUVBolSalEtaDnsPolyDeg2'

rootdir = '../../../lr_Outputs/'

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
da_CC=ds['Cor_Coef'].values

cntrl_filename = rootdir+'ITERATED_PREDICTION_ARRAYS/'+cntrl_name+'_AveragedSinglePredictions.nc'
cntrl_ds = xr.open_dataset(cntrl_filename)
da_cntrl_Av_AbsError=cntrl_ds['Av_AbsErrors'].values
da_cntrl_CC=cntrl_ds['Cor_Coef'].values
 
# mask data
Av_Error = np.where(land_mask==1, da_Av_Error, np.nan)
Av_AbsError = np.where(land_mask==1, da_Av_AbsError, np.nan)
wtd_Av_Error = np.where(land_mask==1, da_wtd_Av_Error, np.nan)
Cntrl_Av_AbsError = np.where(land_mask==1, da_cntrl_Av_AbsError, np.nan)
Cor_Coef = np.where(land_mask==1, da_CC, np.nan)
Cntrl_Cor_Coef = np.where(land_mask==1, da_cntrl_CC, np.nan)

z_size = Av_Error.shape[0]
y_size = Av_Error.shape[1]
x_size = Av_Error.shape[2]

print('Av_Error.shape')
print(Av_Error.shape)

cbar_label = 'Error ('+u'\xb0'+'C)'
#--------------------------
# Plot spatial depth plots
#--------------------------
fig, ax, im = rfplt.plot_depth_fld(Av_Error[:,:,:], 'Averaged Errors', level,
                                   MITGCM_ds['X'].values, MITGCM_ds['Y'].values, MITGCM_ds['Z'].values,
                                   title=None, min_value=None, max_value=None, diff=True,
                                   cbar_label=cbar_label, Sci=True)
plt.savefig(rootdir+'PLOTS/'+model_name+'/'+exp_name+'_AvErrors_z'+str(level)+'.eps', bbox_inches = 'tight', pad_inches = 0.1, format='eps')

fig, ax, im = rfplt.plot_depth_fld(Av_AbsError[:,:,:], 'Averaged Absolute Errors', level,
                                   MITGCM_ds['X'].values, MITGCM_ds['Y'].values, MITGCM_ds['Z'].values,
                                   text='(a)', title=None, min_value=0.0, max_value=0.000306, cmap='Reds',
                                   cbar_label=cbar_label, Sci=True)
plt.savefig(rootdir+'PLOTS/'+model_name+'/'+exp_name+'_AvAbsErrors_z'+str(level)+'.eps', bbox_inches = 'tight', pad_inches = 0.1, format='eps')

fig, ax, im = rfplt.plot_depth_fld(wtd_Av_Error[:,:,:], 'Weighted Averaged Errors', level,
                                   MITGCM_ds['X'].values, MITGCM_ds['Y'].values, MITGCM_ds['Z'].values,
                                   title=None, min_value=None, max_value=None, diff=True,
                                   cbar_label=cbar_label, Sci=True)
plt.savefig(rootdir+'PLOTS/'+model_name+'/'+exp_name+'_WtdAvErrors_z'+str(level)+'.eps', bbox_inches = 'tight', pad_inches = 0.1, format='eps')

fig, ax, im = rfplt.plot_depth_fld(Cor_Coef[:,:,:], 'Correlation Coeffieicents', level,
                                   MITGCM_ds['X'].values, MITGCM_ds['Y'].values, MITGCM_ds['Z'].values,
                                   text='(a)', title=None, min_value=-1., max_value=1., cmap='bwr',
                                   cbar_label=cbar_label, Sci=False)
plt.savefig(rootdir+'PLOTS/'+model_name+'/'+exp_name+'_CorCoef_z'+str(level)+'.eps', bbox_inches = 'tight', pad_inches = 0.1, format='eps')

#-----------------------
# Plot x-cross sections
#-----------------------
fig, ax, im = rfplt.plot_xconst_crss_sec(Av_Error[:,:,:], 'Averaged Errors', x_coord,
                                         MITGCM_ds['X'].values, MITGCM_ds['Y'].values, MITGCM_ds['Z'].values,
                                         title=None, min_value=None, max_value=None, diff=True,
                                         cbar_label=cbar_label, Sci=True)
plt.savefig(rootdir+'PLOTS/'+model_name+'/'+exp_name+'_AvErrors_x'+str(x_coord)+'.eps', bbox_inches = 'tight', pad_inches = 0.1, format='eps')

fig, ax, im = rfplt.plot_xconst_crss_sec(Av_AbsError[:,:,:], 'Averaged Absolute Errors', x_coord,
                                         MITGCM_ds['X'].values, MITGCM_ds['Y'].values, MITGCM_ds['Z'].values,
                                         text='(b)', title=None, min_value=0.0, max_value=0.000306, cmap='Reds',
                                         #text='(a)', title=None, min_value=0.0, max_value=0.000306, cmap='Reds',
                                         cbar_label=cbar_label, Sci=True)
plt.savefig(rootdir+'PLOTS/'+model_name+'/'+exp_name+'_AvAbsErrors_x'+str(x_coord)+'.eps', bbox_inches = 'tight', pad_inches = 0.1, format='eps')

fig, ax, im = rfplt.plot_xconst_crss_sec(wtd_Av_Error[:,:,:], 'Weighted Averaged Errors', x_coord, 
                                         MITGCM_ds['X'].values, MITGCM_ds['Y'].values, MITGCM_ds['Z'].values,
                                         title=None, min_value=None, max_value=None, diff=True,
                                         cbar_label=cbar_label, Sci=True)
plt.savefig(rootdir+'PLOTS/'+model_name+'/'+exp_name+'_WtdAvErrors_x'+str(x_coord)+'.eps', bbox_inches = 'tight', pad_inches = 0.1, format='eps')

fig, ax, im = rfplt.plot_xconst_crss_sec(Cor_Coef[:,:,:], 'Corelation Coeficients', x_coord,
                                         MITGCM_ds['X'].values, MITGCM_ds['Y'].values, MITGCM_ds['Z'].values,
                                         #text='(b)', title=None, min_value=None, max_value=None, cmap='Reds',
                                         text='(a)', title=None, min_value=-1., max_value=1., cmap='bwr',
                                         cbar_label=cbar_label, Sci=False)
plt.savefig(rootdir+'PLOTS/'+model_name+'/'+exp_name+'_CorCoef_x'+str(x_coord)+'.eps', bbox_inches = 'tight', pad_inches = 0.1, format='eps')

#Plot x cross section of difference between cntrl and exp
fig, ax, im = rfplt.plot_xconst_crss_sec(Av_AbsError[:,:,:]-Cntrl_Av_AbsError, 'Diff in Averaged Absolute Errors with Control Run', x_coord,
                                         MITGCM_ds['X'].values, MITGCM_ds['Y'].values, MITGCM_ds['Z'].values,
                                         text='(b)', title=None, min_value=None, max_value=None, diff=True,
                                         cbar_label=cbar_label, Sci=True)
plt.savefig(rootdir+'PLOTS/'+model_name+'/'+exp_name+'_AvAbsErrors_DiffwCntrl_x'+str(x_coord)+'.eps', bbox_inches = 'tight', pad_inches = 0.1, format='eps')

fig, ax, im = rfplt.plot_xconst_crss_sec(Cor_Coef[:,:,:]-Cntrl_Cor_Coef, 'Diff in Corelation Coeficients with Control Run', x_coord,
                                         MITGCM_ds['X'].values, MITGCM_ds['Y'].values, MITGCM_ds['Z'].values,
                                         text='(b)', title=None, min_value=None, max_value=None, diff=True,
                                         cbar_label=cbar_label, Sci=True)
plt.savefig(rootdir+'PLOTS/'+model_name+'/'+exp_name+'_CoerCoef_DiffwCntrl_x'+str(x_coord)+'.eps', bbox_inches = 'tight', pad_inches = 0.1, format='eps')

#-----------------------
# Plot y-cross sections
#-----------------------
fig, ax, im = rfplt.plot_yconst_crss_sec(Av_Error[:,:,:], 'Averaged Errors', y_coord, 
                                         MITGCM_ds['X'].values, MITGCM_ds['Y'].values, MITGCM_ds['Z'].values,
                                         title=None, min_value=None, max_value=None, diff=True,
                                         cbar_label=cbar_label, Sci=True)
plt.savefig(rootdir+'PLOTS/'+model_name+'/'+exp_name+'_AvErrors_y'+str(y_coord)+'.eps', bbox_inches = 'tight', pad_inches = 0.1, format='eps')

fig, ax, im = rfplt.plot_yconst_crss_sec(Av_AbsError[:,:,:], 'Averaged Absolute Errors', y_coord, 
                                         MITGCM_ds['X'].values, MITGCM_ds['Y'].values, MITGCM_ds['Z'].values,
                                         text='(c)', title=None, min_value=0.0, max_value=0.000306, cmap='Reds',
                                         cbar_label=cbar_label, Sci=True)
plt.savefig(rootdir+'PLOTS/'+model_name+'/'+exp_name+'_AvAbsErrors_y'+str(y_coord)+'.eps', bbox_inches = 'tight', pad_inches = 0.1, format='eps')

fig, ax, im = rfplt.plot_yconst_crss_sec(wtd_Av_Error[:,:,:], 'Weighted Averaged Errors', y_coord, 
                                         MITGCM_ds['X'].values, MITGCM_ds['Y'].values, MITGCM_ds['Z'].values,
                                         title=None, min_value=None, max_value=None, diff=True,
                                         cbar_label=cbar_label, Sci=True)
plt.savefig(rootdir+'PLOTS/'+model_name+'/'+exp_name+'_WtdAvErrors_y'+str(y_coord)+'.eps', bbox_inches = 'tight', pad_inches = 0.1, format='eps')

fig, ax, im = rfplt.plot_yconst_crss_sec(Cor_Coef[:,:,:], 'Corelation Coefiecients', y_coord, 
                                         MITGCM_ds['X'].values, MITGCM_ds['Y'].values, MITGCM_ds['Z'].values,
                                         text='(c)', title=None, min_value=-1., max_value=1., cmap='bwr',
                                         cbar_label=cbar_label, Sci=False)
plt.savefig(rootdir+'PLOTS/'+model_name+'/'+exp_name+'_CorCoef_y'+str(y_coord)+'.eps', bbox_inches = 'tight', pad_inches = 0.1, format='eps')


