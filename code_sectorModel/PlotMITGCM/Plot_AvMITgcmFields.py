#!/usr/bin/env python
# coding: utf-8

print('import packages')
import sys
sys.path.append('../Tools')
import CreateDataName as cn
import Model_Plotting as rfplt

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

DIR  = '/data/hpcdata/users/racfur/MITgcm/verification/MundaySectorConfig_2degree/runs/100yrs/mnc_test_0002/'
MITGCM_filename=DIR+'cat_tave.nc'


variable_names = ('Av_DFrE_TH', 'Av_DFrI_TH', 'Av_DFyE_TH', 'Av_DFxE_TH', 'Av_ADVr_TH', 'Av_ADVy_TH', 'Av_ADVx_TH')

plotnames = {'Av_DFrE_TH':'Explicit Vertical Diffusive Flux of Pot.Temperature\n',
             'Av_DFrI_TH':'Implicit Vertical Diffusive Flux of Pot.Temperature\n',
             'Av_DFyE_TH':'Meridional Diffusive Flux of Pot.Temperature\n',
             'Av_DFxE_TH':'Zonal Diffusive Flux of Pot.Temperature\n',
             'Av_ADVr_TH':'Vertical Advective Flux of Pot.Temperature\n',
             'Av_ADVy_TH':'Meridional Advective Flux of Pot.Temperature\n',
             'Av_ADVx_TH':'Zonal Advective Flux of Pot.Temperature\n'}


filenames = {'Av_DFrE_TH':'VertDiffExp',
             'Av_DFrI_TH':'VertDiffImp',
             'Av_DFyE_TH':'MerDiff',
             'Av_DFxE_TH':'ZonalDiff',
             'Av_ADVr_TH':'VertAdv',
             'Av_ADVy_TH':'MerAdv',
             'Av_ADVx_TH':'ZonalAdv'}


cbar_label = 'Flux ($'+u'\xb0'+'Cm^3/s)$'
#-----------
level = point[0]
y_coord = point[1]
x_coord = point[2]

rootdir = '../../../MITGCM_Analysis_Sector/'

#------------------------
print('reading in data')
#------------------------
data_filename=rootdir+'AveragedMITgcmData.nc'
ds = xr.open_dataset(data_filename)

for variable in variable_names: 
   Av_field=ds[variable].values
   #Av_field=ds[variable_name].values+ds[variable2_name].values
   
   #-----------------------
   # Plot x-cross sections
   #-----------------------
   fig, ax, im = rfplt.plot_xconst_crss_sec(Av_field[:,:,:], plotnames[variable], x_coord,
                                            ds['X'].values, ds['Y'].values, ds['Z'].values,
                                            title=None, min_value=None, max_value=None, diff=False,
                                            cmap='Reds', cbar_label=cbar_label, Sci=True)
   plt.savefig(rootdir+'PLOTS/Av_'+filenames[variable]+'_x'+str(x_coord)+'.png', bbox_inches = 'tight', pad_inches = 0.1)
   
##--------------------------
## Plot spatial depth plots
##--------------------------
#fig, ax, im = rfplt.plot_depth_fld(Av_Error[:,:,:], 'Averaged Errors', level,
#                                   MITGCM_ds['X'].values, MITGCM_ds['Y'].values, MITGCM_ds['Z'].values,
#                                   title=None, min_value=None, max_value=None, diff=True)
#plt.savefig(rootdir+'PLOTS/'+model_name+'/'+exp_name+'_AvErrors_z'+str(level)+'.png', bbox_inches = 'tight', pad_inches = 0.1)
#
#
##-----------------------
## Plot y-cross sections
##-----------------------
#fig, ax, im = rfplt.plot_yconst_crss_sec(Av_Error[:,:,:], 'Averaged Errors', y_coord, 
#                                         MITGCM_ds['X'].values, MITGCM_ds['Y'].values, MITGCM_ds['Z'].values,
#                                         title=None, min_value=None, max_value=None, diff=True)
#plt.savefig(rootdir+'PLOTS/'+model_name+'/'+exp_name+'_AvErrors_y'+str(y_coord)+'.png', bbox_inches = 'tight', pad_inches = 0.1)
#
