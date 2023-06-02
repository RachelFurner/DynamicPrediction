#!/usr/bin/env python
# coding: utf-8

print('import packages')
import sys
sys.path.append('../')
from Tools import Channel_Model_Plotting as ChnPlt

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os
import xarray as xr
from netCDF4 import Dataset

plt.rcParams.update({'font.size': 14})

#----------------------------
# Set variables for this run
#----------------------------
point = [ 2, 8, 6]

epochs = '10'
dir_name = 'Spits12hrly_UNet2dtransp_histlen1_rolllen1_seed30475'
#dir_name = 'MultiModel_Spits12hrly_UNet2dtransp_histlen1_rolllen1'
model_name = dir_name+'_'+epochs+'epochs'
trainorval='test'

rootdir = '../../../Channel_nn_Outputs/'+dir_name+'/STATS/'

#------------------------
print('reading in data')
#------------------------
stats_data_filename=rootdir+model_name+'_StatsOutput_'+trainorval+'.nc'
stats_ds = xr.open_dataset(stats_data_filename)
da_Temp_RMS = stats_ds['Temp_RMS']
da_U_RMS    = stats_ds['U_RMS']
da_V_RMS    = stats_ds['V_RMS']
da_Eta_RMS  = stats_ds['Eta_RMS']
da_X = stats_ds['X']
da_Y = stats_ds['Y']
da_Z = stats_ds['Z']
da_Temp_mask= stats_ds['Temp_Mask']
da_U_mask   = stats_ds['U_Mask']
da_V_mask   = stats_ds['V_Mask']
da_Eta_mask = stats_ds['Eta_Mask']

masked_Temp_RMS = np.where( da_Temp_mask.values==0, np.nan, da_Temp_RMS.values )
masked_U_RMS    = np.where( da_U_mask.values==0, np.nan, da_U_RMS.values )
masked_V_RMS    = np.where( da_V_mask.values==0, np.nan, da_V_RMS.values )
masked_Eta_RMS  = np.where( da_Eta_mask.values==0, np.nan, da_Eta_RMS.values )

#--------------------------
# Plot spatial depth plots
#--------------------------
#for level in range(38):
for level in [2]:
   fig, ax, im = ChnPlt.plot_depth_fld(masked_Temp_RMS[level,:,:], 'Temperature RMS Errors ('+u'\xb0'+'C)', level,
                                       da_X.values, da_Y.values, da_Z.values, title=None, extend='max',
                                       norm='log', min_value=0.005, max_value=0.13)
                                       #min_value=0.0, max_value=0.025)
                                       #min_value=0.0, max_value=0.06)
                                       #min_value=0.0, max_value=0.13)
   plt.savefig(rootdir+'PLOTS/'+model_name+'_Temp_RMS_z'+str(level)+'_'+trainorval+'.png', bbox_inches = 'tight', pad_inches = 0.1)
   plt.close()
   
   fig, ax, im = ChnPlt.plot_depth_fld(masked_U_RMS[level,:,:], 'East-West Velocity RMS Errors (m/s)', level,
                                       da_X.values, da_Y.values, da_Z.values, title=None, extend='max',
                                       norm='log', min_value=0.0005, max_value=0.05)
                                       #min_value=0.0, max_value=0.005)
                                       #min_value=0.0, max_value=0.015)
                                       #min_value=0.0, max_value=0.05)
   plt.savefig(rootdir+'PLOTS/'+model_name+'_U_RMS_z'+str(level)+'_'+trainorval+'.png', bbox_inches = 'tight', pad_inches = 0.1)
   plt.close()
   
   fig, ax, im = ChnPlt.plot_depth_fld(masked_V_RMS[level,:,:], 'North-South Velocity RMS Errors (m/s)', level,
                                       da_X.values, da_Y.values, da_Z.values, title=None, extend='max',
                                       norm='log', min_value=0.0005, max_value=0.06)
                                       #min_value=0.0, max_value=0.006)
                                       #min_value=0.0, max_value=0.02)
                                       #min_value=0.0, max_value=0.06)
   plt.savefig(rootdir+'PLOTS/'+model_name+'_V_RMS_z'+str(level)+'_'+trainorval+'.png', bbox_inches = 'tight', pad_inches = 0.1)
   plt.close()
   
fig, ax, im = ChnPlt.plot_depth_fld(masked_Eta_RMS[:,:], 'Sea Surface Height RMS Errors (m)', 0,
                                    da_X.values, da_Y.values, da_Z.values, title=None, extend='max',
                                    norm='log', min_value=0.0001, max_value=0.019)
                                    #min_value=0.0, max_value=0.0012)
                                    #min_value=0.0, max_value=0.004)
                                    #min_value=0.0, max_value=0.019)
plt.savefig(rootdir+'PLOTS/'+model_name+'_Eta_RMS_'+trainorval+'.png', bbox_inches = 'tight', pad_inches = 0.1)
plt.close()
