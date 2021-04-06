#!/usr/bin/env python
# coding: utf-8
# Script written by Rachel Furner
# Plots avergaged fluxes outputted by MITgcm and averaged in CalcAvMITgcmFields.py
# creating spatial patterns of average trends.

print('import packages')
import sys
sys.path.append('../Tools')
import Model_Plotting as rfplt
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from netCDF4 import Dataset

#----------------------------
# Set variables for this run
#----------------------------
point = [ 2, 8, 6]
level = point[0]
y_coord = point[1]
x_coord = point[2]

DIR  = '/data/hpcdata/users/racfur/MITgcm/verification/MundaySectorConfig_2degree/runs/100yrs/mnc_test_0002/'
MITGCM_filename=DIR+'cat_tave.nc'

rootdir = '../../../MITGCM_Analysis_Sector/'

variable_names = ('Av_ADVr_TH', 'Av_ADVy_TH', 'Av_ADVx_TH', 'Av_DFrE_TH', 'Av_DFrI_TH', 'Av_DFyE_TH', 'Av_DFxE_TH')

plotnames = {'Av_ADVr_TH':'Vertical Advective Flux of Pot.Temperature\n',
             'Av_ADVy_TH':'Meridional Advective Flux of Pot.Temperature\n',
             'Av_ADVx_TH':'Zonal Advective Flux of Pot.Temperature\n',
             'Av_DFrE_TH':'Explicit Vertical Diffusive Flux of Pot.Temperature\n',
             'Av_DFrI_TH':'Implicit Vertical Diffusive Flux of Pot.Temperature\n',
             'Av_DFyE_TH':'Meridional Diffusive Flux of Pot.Temperature\n',
             'Av_DFxE_TH':'Zonal Diffusive Flux of Pot.Temperature\n'}

text = {'Av_ADVx_TH':'(a)',
        'Av_ADVy_TH':'(b)',
        'Av_ADVr_TH':'(c)',
        'Av_DFxE_TH':'(a)',
        'Av_DFyE_TH':'(b)',
        'Av_DFrE_TH':'(c)',
        'Av_DFrI_TH':'(d)'}

filenames = {'Av_ADVx_TH':'fig02a',
             'Av_ADVy_TH':'fig02b',
             'Av_ADVr_TH':'fig02c',
             'Av_DFxE_TH':'fig03a',
             'Av_DFyE_TH':'fig03b',
             'Av_DFrE_TH':'fig03c',
             'Av_DFrI_TH':'fig03d'}

cbar_label = 'Flux $(Cm^3 s^{-1})$'

plt.rcParams.update({'font.size': 10})
plt.rc('font', family='sans serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')    

#------------------------
print('reading in data')
#------------------------
data_filename=rootdir+'AveragedMITgcmData.nc'
ds = xr.open_dataset(data_filename)

for variable in variable_names: 
   Av_field=ds[variable].values
   
   #-----------------------
   # Plot x-cross sections
   #-----------------------
   fig, ax, im = rfplt.plot_xconst_crss_sec(Av_field[:,:,:], plotnames[variable], x_coord,
                                            ds['X'].values, ds['Y'].values, ds['Z'].values,
                                            text=text[variable], title=None, min_value=None, max_value=None, diff=False,
                                            cmap='Reds', cbar_label=cbar_label, Sci=True)
   plt.savefig(rootdir+'PLOTS/'+filenames[variable]+'.eps', bbox_inches = 'tight', pad_inches = 0.1, format='eps')
 
