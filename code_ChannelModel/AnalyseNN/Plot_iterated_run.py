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
point = [ 2, 48, 120]

dir_name = 'Spits12hrly_UNet2dtransp_histlen1_rolllen1_seed30475'
#dir_name = 'MultiModel_Spits_UNet2dtransp_histlen1'
epochs = '200'
iteration_len = 180
iteration_method = 'simple'
smooth_level = '0'
smooth_steps = '0'

plot_timeseries = True 
plot_rms_timeseries = True 
make_animation_plots = False
animation_end = 60 
ts_end = 180

pert_main_dir = '/data/hpcdata/users/racfur/MITgcm/verification/MundayChannelConfig10km_LandSpits/runs/'
pert_list = ['50yr_smooth100']
pert_filename = '/12hrly_data.nc'

model_name = dir_name+'_'+epochs+'epochs_'+iteration_method+'_smth'+smooth_level+'stps'+smooth_steps

rootdir = '../../../Channel_nn_Outputs/'+dir_name

level = point[0]
y_coord = point[1]
x_coord = point[2]

#-------------------------------------
print('reading in NN iteration data')
#-------------------------------------
iter_data_filename=rootdir+'/ITERATED_FORECAST/'+model_name+'_Forlen'+str(iteration_len)+'.nc'
iter_ds = xr.open_dataset(iter_data_filename)
da_true_Temp = iter_ds['True_Temp']
da_true_U    = iter_ds['True_U']
da_true_V    = iter_ds['True_V']
da_true_Eta  = iter_ds['True_Eta']
da_pred_Temp = iter_ds['Pred_Temp']
da_pred_U    = iter_ds['Pred_U']
da_pred_V    = iter_ds['Pred_V']
da_pred_Eta  = iter_ds['Pred_Eta']
da_Temp_errors = iter_ds['Temp_Errors']
da_U_errors    = iter_ds['U_Errors']
da_V_errors    = iter_ds['V_Errors']
da_Eta_errors  = iter_ds['Eta_Errors']
da_Temp_mask   = iter_ds['Temp_Mask']
da_U_mask      = iter_ds['U_Mask']
da_V_mask      = iter_ds['V_Mask']
da_Eta_mask    = iter_ds['Eta_Mask']
da_X = iter_ds['X']
da_Y = iter_ds['Y']
da_Z = iter_ds['Z']

print('da_true_Temp.shape')
print(da_true_Temp.shape)

masked_True_Temp = np.where( da_Temp_mask.values==0, np.nan, da_true_Temp.values )
masked_Pred_Temp = np.where( da_Temp_mask.values==0, np.nan, da_pred_Temp.values )
masked_True_U    = np.where( da_U_mask.values==0, np.nan, da_true_U.values )
masked_Pred_U    = np.where( da_U_mask.values==0, np.nan, da_pred_U.values )
masked_True_V    = np.where( da_V_mask.values==0, np.nan, da_true_V.values )
masked_Pred_V    = np.where( da_V_mask.values==0, np.nan, da_pred_V.values )
masked_True_Eta  = np.where( da_Eta_mask.values==0, np.nan, da_true_Eta.values )
masked_Pred_Eta  = np.where( da_Eta_mask.values==0, np.nan, da_pred_Eta.values )

#--------------------------------------------
# Read in data from perturbed runs if needed
#--------------------------------------------
if plot_timeseries or plot_rms_timeseries: 
   Temp_dict = {'True Temperature':masked_True_Temp[:ts_end], 'Predicted Temperature':masked_Pred_Temp[:ts_end] }
   UVel_dict = {'True East-West Velocity':masked_True_U[:ts_end], 'Predicted East-West Velocity':masked_Pred_U[:ts_end] }
   VVel_dict = {'True North-South Velocity':masked_True_V[:ts_end], 'Predicted North-South Velocity':masked_Pred_V[:ts_end] }
   Eta_dict  = {'True Sea Surface Height':masked_True_Eta[:ts_end], 'Predicted Sea Surface Height':masked_Pred_Eta[:ts_end] }

   colors = ['black', 'red']
   
   Temp_RMS = np.sqrt(np.nanmean( np.square(masked_Pred_Temp[:ts_end,:,:,:]-masked_True_Temp[:ts_end,:,:,:]), axis=(1,2,3) ))
   UVel_RMS = np.sqrt(np.nanmean( np.square(masked_Pred_U[:ts_end,:,:,:]-masked_True_U[:ts_end,:,:,:]), axis=(1,2,3) ))
   VVel_RMS = np.sqrt(np.nanmean( np.square(masked_Pred_V[:ts_end,:,:,:]-masked_True_V[:ts_end,:,:,:]), axis=(1,2,3) ))
   Eta_RMS  = np.sqrt(np.nanmean( np.square(masked_Pred_Eta[:ts_end,:,:]-masked_True_Eta[:ts_end,:,:]), axis=(1,2) ))

   RMS_Temp_dict = { 'Temperature RMS':Temp_RMS } 
   RMS_UVel_dict = { 'East-West Velocity RMS':UVel_RMS }
   RMS_VVel_dict = { 'North-South Velocity RMS':VVel_RMS }
   RMS_Eta_dict  = { 'Sea Surface Height RMS':Eta_RMS }
   RMS_dict = { 'Temperature RMS':Temp_RMS, 'East-West Velocity RMS':UVel_RMS, 'North-South Velocity RMS':VVel_RMS, 'Sea Surface Height RMS':Eta_RMS }

   RMS_colors = ['red']

   for pert in pert_list:
      print(pert)
      pert_file =  pert_main_dir+pert+pert_filename
      Pert_ds = xr.open_dataset(pert_file).isel( T=slice( 0, int(ts_end) ) )
   
      da_PertTemp = Pert_ds['THETA']
      masked_PertTemp = np.where( da_Temp_mask.values==0, np.nan, da_PertTemp.values )
      Temp_dict[pert] = masked_PertTemp[:ts_end]
      RMS_Temp_dict[pert] =  np.sqrt(np.nanmean( np.square(masked_PertTemp[:ts_end,:,:,:]-masked_True_Temp[:ts_end,:,:,:]), axis=(1,2,3) ))
   
      da_PertUvel = Pert_ds['UVEL']
      PertUvel = 0.5 * (da_PertUvel.values[:,:,:,:-1]+da_PertUvel.values[:,:,:,1:])   
      masked_PertUvel = np.where( da_U_mask.values==0, np.nan, PertUvel )
      UVel_dict[pert] = masked_PertUvel[:ts_end]
      RMS_UVel_dict[pert] =  np.sqrt(np.nanmean( np.square(masked_PertUvel[:ts_end,:,:,:]-masked_True_U[:ts_end,:,:,:]), axis=(1,2,3) ))
   
      da_MITgcmVvel = Pert_ds['VVEL']
      MITgcmVvel = 0.5 * (da_MITgcmVvel.values[:,:,:-1,:]+da_MITgcmVvel.values[:,:,1:,:])   
      masked_MITgcmVvel = np.where( da_V_mask.values==0, np.nan, MITgcmVvel )
      RMS_VVel_dict[pert] =  np.sqrt(np.nanmean( np.square(masked_MITgcmVvel[:ts_end,:,:,:]-masked_True_V[:ts_end,:,:,:]), axis=(1,2,3) ))
      VVel_dict[pert] = masked_MITgcmVvel[:ts_end]
   
      da_PertEta = Pert_ds['ETAN']
      masked_PertEta = np.where( da_Eta_mask.values==0, np.nan, da_PertEta.values[:,0,:,:] )
      Eta_dict[pert] = masked_PertEta[:ts_end]
      RMS_Eta_dict[pert] =  np.sqrt(np.nanmean( np.square(masked_PertEta[:ts_end,:,:]-masked_True_Eta[:ts_end,:,:]), axis=(1,2) ))

      colors.append('grey')
      RMS_colors.append('grey')

#----------------------------
# Plot timeseries at a point
#----------------------------
if plot_timeseries: 
   fig = ChnPlt.plt_timeseries( point, ts_end, Temp_dict, y_label='Temperature ('+u'\xb0'+'C)', colors=colors )
   plt.savefig(rootdir+'/ITERATED_FORECAST/'+model_name+'_Temp_timeseries_z'+str(point[0])+'y'+str(point[1])+'x'+str(point[2])+
               '_'+str(ts_end)+'.png', bbox_inches = 'tight', pad_inches = 0.1)

   fig = ChnPlt.plt_timeseries( point, ts_end, UVel_dict, y_label='East-West Velocity $(m s^{-1})$' ) 
   plt.savefig(rootdir+'/ITERATED_FORECAST/'+model_name+'_U_timeseries_z'+str(point[0])+'y'+str(point[1])+'x'+str(point[2])+
               '_'+str(ts_end)+'.png', bbox_inches = 'tight', pad_inches = 0.1)

   fig = ChnPlt.plt_timeseries( point, ts_end, VVel_dict, y_label='North-South Velocity $(m s^{-1})$') 
   plt.savefig(rootdir+'/ITERATED_FORECAST/'+model_name+'_V_timeseries_z'+str(point[0])+'y'+str(point[1])+'x'+str(point[2])+
               '_'+str(ts_end)+'.png', bbox_inches = 'tight', pad_inches = 0.1)
   
   fig = ChnPlt.plt_timeseries( point[1:], ts_end, Eta_dict, y_label='SSH $(m)$') 
   plt.savefig(rootdir+'/ITERATED_FORECAST/'+model_name+'_Eta_timeseries_z'+str(point[0])+'y'+str(point[1])+'x'+str(point[2])+
               '_'+str(ts_end)+'.png', bbox_inches = 'tight', pad_inches = 0.1)

#----------------------------------------
# Plot spatially averaged RMS timeseries
#----------------------------------------
if plot_rms_timeseries:
   fig = ChnPlt.plt_timeseries( [], ts_end, RMS_Temp_dict, y_label='Temperature RMS error ('+u'\xb0'+'C)', colors=RMS_colors )
   plt.savefig(rootdir+'/ITERATED_FORECAST/'+model_name+'_Temp_RMS_timeseries_'+str(ts_end)+'.png',
               bbox_inches = 'tight', pad_inches = 0.1)

   fig = ChnPlt.plt_timeseries( [], ts_end, RMS_UVel_dict, y_label='East-West Velocity RMS error $(m s^{-1})$' ) 
   plt.savefig(rootdir+'/ITERATED_FORECAST/'+model_name+'_U_RMS_timeseries_'+str(ts_end)+'.png',
               bbox_inches = 'tight', pad_inches = 0.1)

   fig = ChnPlt.plt_timeseries( [], ts_end, RMS_VVel_dict, y_label='North-South Velocity RMS error $(m s^{-1})$') 
   plt.savefig(rootdir+'/ITERATED_FORECAST/'+model_name+'_V_RMS_timeseries_'+str(ts_end)+'.png',
               bbox_inches = 'tight', pad_inches = 0.1)
   
   fig = ChnPlt.plt_timeseries( [], ts_end, RMS_Eta_dict, y_label='Sea Surface Height RMS error $(m)$') 
   plt.savefig(rootdir+'/ITERATED_FORECAST/'+model_name+'_Eta_RMS_timeseries_'+str(ts_end)+'.png',
               bbox_inches = 'tight', pad_inches = 0.1)

   fig = ChnPlt.plt_timeseries( [], ts_end, RMS_dict, y_label='RMS errors') 
   plt.savefig(rootdir+'/ITERATED_FORECAST/'+model_name+'_RMS_timeseries_'+str(ts_end)+'.png',
               bbox_inches = 'tight', pad_inches = 0.1)

#-------------------------------
# Plot spatial plots to animate
#-------------------------------
if make_animation_plots:
   for time in range(0,animation_end):
      fig = ChnPlt.plot_depth_fld_diff(masked_True_Temp[time,level,:,:], 'True Temperature',
                                       masked_Pred_Temp[time,level,:,:], 'Predicted Temperature',
                                       level, da_X.values, da_Y.values, da_Z.values, title=None,
                                       flds_min_value=0., flds_max_value=6.5, diff_min_value=-1, diff_max_value=1, extend='max' )
                                       #flds_min_value=0., flds_max_value=6.5, diff_min_value=-1, diff_max_value=1, extend='max' )
      plt.savefig(rootdir+'/ITERATED_FORECAST/PLOTS/'+model_name+'_Temp_level'+str(level)+'_time'+f'{time:03}'+'.png',
                  bbox_inches = 'tight', pad_inches = 0.1)
      plt.close()
      
      fig = ChnPlt.plot_depth_fld_diff(masked_True_U[time,level,:,:], 'True East-West Velocity',
                                       masked_Pred_U[time,level,:,:], 'Predicted East-West Velocity',
                                       level, da_X.values, da_Y.values, da_Z.values,
                                       flds_min_value=-1.2, flds_max_value=1.2, diff_min_value=-1. , diff_max_value=1. , extend='both' )
                                       #flds_min_value=-1.2, flds_max_value=1.2, diff_min_value=-0.5, diff_max_value=0.5, extend='both' )
      plt.savefig(rootdir+'/ITERATED_FORECAST/PLOTS/'+model_name+'_U_level'+str(level)+'_time'+f'{time:03}'+'.png',
                  bbox_inches = 'tight', pad_inches = 0.1)
      plt.close()
      
      fig = ChnPlt.plot_depth_fld_diff(masked_True_V[time,level,:,:], 'True North-South Velocity',
                                       masked_Pred_V[time,level,:,:], 'Predicted North-South Velocity',
                                       level, da_X.values, da_Y.values, da_Z.values, title=None,
                                       flds_min_value=-1.2, flds_max_value=1.2, diff_min_value=-1.0, diff_max_value=1.0, extend='both' )
                                       #flds_min_value=-1.2, flds_max_value=1.2, diff_min_value=-0.5, diff_max_value=0.5, extend='both' )
      plt.savefig(rootdir+'/ITERATED_FORECAST/PLOTS/'+model_name+'_V_level'+str(level)+'_time'+f'{time:03}'+'.png',
                  bbox_inches = 'tight', pad_inches = 0.1)
      plt.close()
      
      fig = ChnPlt.plot_depth_fld_diff(masked_True_Eta[time,:,:], 'True Sea Surface Height',
                                       masked_Pred_Eta[time,:,:], 'Predicted Sea Surface Height',
                                       0, da_X.values, da_Y.values, da_Z.values, title=None,
                                       flds_min_value=-1, flds_max_value=1, diff_min_value=-1.0, diff_max_value=1.0 )
                                       #flds_min_value=-1, flds_max_value=1, diff_min_value=-0.3, diff_max_value=0.3 )
      plt.savefig(rootdir+'/ITERATED_FORECAST/PLOTS/'+model_name+'_Eta_level'+str(level)+'_time'+f'{time:03}'+'.png', 
                  bbox_inches = 'tight', pad_inches = 0.1)
      plt.close()

