#!/usr/bin/env python
# coding: utf-8

# Script to read in iterated run data (nc files) from multiple runs, and 'truth' data from purturbed 
# MITgcm runs.
# Spatial plots of the predicted field, the true field, and the difference are plotted for each of the
# runs, and for every required timestep. These can be catted together to show all variables at once,
# and animated to give a video of the evolution of predicted fields, true fields, and the difference,
# using cat_animate_iterations.sh
# The script then plots a timeseries for a given point, for the truth, the various NN runs, and the 
# perturbed MITgcm runs.
# It also plots a timeseries of RMS error and CC; Showing how RMS error (between NN predictions and 
# true MITgcm fields) evolves over time. And how the CC between MITgcm truth and NN predictions evolve
# over time.
# Finally the script prints (to the screen) the values of the RMS error and CC for various points along
# the timeseries.

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

#dir_names = ['Spits12hrly_UNet2dtransp_histlen1_rolllen1_seed30475',
#             'Spits12hrly_UNet2dtransp_histlen1_rolllen1_seed30475',
#             'MultiModel_Spits12hrly_UNet2dtransp_histlen1_rolllen1',
#             'Spits12hrly_UNet2dtransp_histlen3_rolllen1_seed30475',
#             'Spits12hrly_UNet2dtransp_histlen1_rolllen3_seed30475',
#             'Spits12hrly_UNetConvLSTM_histlen5_rolllen1_seed30475']
#labels = ['Standard Network', 'smoothing', 'Multi-model average', '3 past fields', 'Rollout loss length 3', 'ConvLSTM length 5']
#epochs = ['200', '200', '200', '200', '200', '200']
#iteration_methods = ['simple', 'simple', 'simple', 'simple', 'simple', 'simple']
#smooth_levels = ['0', '20', '0', '0', '0', '0']

dir_names = ['Spits12hrly_UNet2dtransp_histlen1_rolllen1_seed30475']
labels = ['']
epochs = ['200']
iteration_methods = ['simple']
smooth_levels = ['0']

iteration_len = 180 
smooth_steps = '0'

my_colors = ['red', 'cyan', 'orange', 'blue', 'green', 'purple']
my_alphas = [ 1., 1., 1., 1., 1., 1., 1. ]

plot_timeseries = True   
plot_rms_timeseries = True
print_spatially_av_rms = True 
print_cc = True 
plot_cc_timeseries = True 
make_animation_plots = True 
animation_end = 43
ts_end = 180 
print_times = [28, 56, 84, 112]  # times to print, in terms of how many 12 hour jumps; 2,4,6,8 weeks

#pert_list = ['50yr_smooth20', '50yr_smooth40', '50yr_smooth60', '50yr_smooth80', '50yr_smooth100', '50yr_smooth125', '50yr_smooth150']
pert_list = []

#---------------------------
level = point[0]
y_coord = point[1]
x_coord = point[2]

Temp_dict = {}
UVel_dict = {}
VVel_dict = {}
Eta_dict  = {}

RMS_Temp_dict = {}
RMS_UVel_dict = {}
RMS_VVel_dict = {}
RMS_Eta_dict  = {}
RMS_dict = {}

CC_Temp_dict = {}
CC_UVel_dict = {}
CC_VVel_dict = {}
CC_Eta_dict  = {}

colors = []
RMS_colors = []
alphas = []
RMS_alphas = []

#-------------------------------------
print('reading in NN iteration data')
#-------------------------------------
count=0
for model in dir_names:
   model_name = model+'_'+epochs[count]+'epochs_'+iteration_methods[count]+'_smth'+smooth_levels[count]+'stps'+smooth_steps
   rootdir = '../../../Channel_nn_Outputs/'+model
   
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
   
   masked_True_Temp = np.where( da_Temp_mask.values==0, np.nan, da_true_Temp.values )
   masked_Pred_Temp = np.where( da_Temp_mask.values==0, np.nan, da_pred_Temp.values )
   masked_True_U    = np.where( da_U_mask.values==0, np.nan, da_true_U.values )
   masked_Pred_U    = np.where( da_U_mask.values==0, np.nan, da_pred_U.values )
   masked_True_V    = np.where( da_V_mask.values==0, np.nan, da_true_V.values )
   masked_Pred_V    = np.where( da_V_mask.values==0, np.nan, da_pred_V.values )
   masked_True_Eta  = np.where( da_Eta_mask.values==0, np.nan, da_true_Eta.values )
   masked_Pred_Eta  = np.where( da_Eta_mask.values==0, np.nan, da_pred_Eta.values )

   if count==0:
      Temp_dict['MITgcm Temperature'] = masked_True_Temp[:ts_end]
      UVel_dict['MITgcm Eastward Velocity'] = masked_True_U[:ts_end]
      VVel_dict['MITgcm Northward Velocity'] = masked_True_V[:ts_end]
      Eta_dict['MITgcm Sea Surface Height'] = masked_True_Eta[:ts_end]
      colors.append('black')
      alphas.append(1)

   Temp_dict[labels[count]] = masked_Pred_Temp[:ts_end]
   UVel_dict[labels[count]] = masked_Pred_U[:ts_end]
   VVel_dict[labels[count]] = masked_Pred_V[:ts_end]
   Eta_dict[labels[count]]  = masked_Pred_Eta[:ts_end]

   colors.append(my_colors[count])
   alphas.append(my_alphas[count])
   
   Temp_RMS = np.sqrt(np.nanmean( np.square(masked_Pred_Temp[:ts_end,:,:,:]-masked_True_Temp[:ts_end,:,:,:]), axis=(1,2,3) ))
   UVel_RMS = np.sqrt(np.nanmean( np.square(masked_Pred_U[:ts_end,:,:,:]-masked_True_U[:ts_end,:,:,:]), axis=(1,2,3) ))
   VVel_RMS = np.sqrt(np.nanmean( np.square(masked_Pred_V[:ts_end,:,:,:]-masked_True_V[:ts_end,:,:,:]), axis=(1,2,3) ))
   Eta_RMS  = np.sqrt(np.nanmean( np.square(masked_Pred_Eta[:ts_end,:,:]-masked_True_Eta[:ts_end,:,:]), axis=(1,2) ))

   RMS_Temp_dict[labels[count]] = Temp_RMS
   RMS_UVel_dict[labels[count]] = UVel_RMS
   RMS_VVel_dict[labels[count]] = VVel_RMS
   RMS_Eta_dict[labels[count]] = Eta_RMS
   RMS_dict = { 'Temperature RMS':Temp_RMS, 'Eastward Velocity RMS':UVel_RMS, 'Northward Velocity RMS':VVel_RMS, 'Sea Surface Height RMS':Eta_RMS }

   if print_cc or plot_cc_timeseries:
      CC_Temp_dict[labels[count]] = np.zeros(ts_end)
      CC_UVel_dict[labels[count]] = np.zeros(ts_end)
      CC_VVel_dict[labels[count]] = np.zeros(ts_end)
      CC_Eta_dict[labels[count]] = np.zeros(ts_end)
      for time in range(ts_end):
         CC_Temp_dict[labels[count]][time] = np.corrcoef( np.reshape(masked_Pred_Temp[time,:,:,:][~np.isnan(masked_Pred_Temp[time,:,:,:])],(-1)),
                                                          np.reshape(masked_True_Temp[time,:,:,:][~np.isnan(masked_True_Temp[time,:,:,:])],(-1)) )[0,1]
         CC_UVel_dict[labels[count]][time] = np.corrcoef( np.reshape(masked_Pred_U[time,:,:,:][~np.isnan(masked_Pred_U[time,:,:,:])],(-1)),
                                                          np.reshape(masked_True_U[time,:,:,:][~np.isnan(masked_True_U[time,:,:,:])],(-1)) )[0,1]
         CC_VVel_dict[labels[count]][time] = np.corrcoef( np.reshape(masked_Pred_V[time,:,:,:][~np.isnan(masked_Pred_V[time,:,:,:])],(-1)),
                                                          np.reshape(masked_True_V[time,:,:,:][~np.isnan(masked_True_V[time,:,:,:])],(-1)) )[0,1]
         CC_Eta_dict[labels[count]][time]  = np.corrcoef( np.reshape(masked_Pred_Eta[time,:,:][~np.isnan(masked_Pred_Eta[time,:,:])],(-1)),
                                                          np.reshape(masked_True_Eta[time,:,:][~np.isnan(masked_True_Eta[time,:,:])],(-1)) )[0,1]

   RMS_colors.append(my_colors[count])
   RMS_alphas.append(my_alphas[count])

   #-------------------------------
   # Plot spatial plots to animate
   #-------------------------------
   if make_animation_plots:
      for time in range(0,animation_end):
         fig = ChnPlt.plot_depth_fld_diff(masked_True_Temp[time,level,:,:], 'MITgcm Temperature',
                                          masked_Pred_Temp[time,level,:,:], 'Predicted Temperature',
                                          level, da_X.values, da_Y.values, da_Z.values, title=None,
                                          flds_min_value=0., flds_max_value=6.5, diff_min_value=-1, diff_max_value=1, extend='max' )
                                          #flds_min_value=0., flds_max_value=6.5, diff_min_value=-1, diff_max_value=1, extend='max' )
         plt.savefig(rootdir+'/ITERATED_FORECAST/PLOTS/'+model_name+'_Temp_level'+str(level)+'_time'+f'{time:03}'+'.png',
                     bbox_inches = 'tight', pad_inches = 0.1)
         plt.close()
         
         fig = ChnPlt.plot_depth_fld_diff(masked_True_U[time,level,:,:], 'MITgcm Eastward Velocity',
                                          masked_Pred_U[time,level,:,:], 'Predicted Eastward Velocity',
                                          level, da_X.values, da_Y.values, da_Z.values,
                                          flds_min_value=-1.2, flds_max_value=1.2, diff_min_value=-0.5, diff_max_value=0.5, extend='both' )
                                          #flds_min_value=-1.2, flds_max_value=1.2, diff_min_value=-1. , diff_max_value=1. , extend='both' )
         plt.savefig(rootdir+'/ITERATED_FORECAST/PLOTS/'+model_name+'_U_level'+str(level)+'_time'+f'{time:03}'+'.png',
                     bbox_inches = 'tight', pad_inches = 0.1)
         plt.close()
         
         fig = ChnPlt.plot_depth_fld_diff(masked_True_V[time,level,:,:], 'MITgcm Northward Velocity',
                                          masked_Pred_V[time,level,:,:], 'Predicted Northward Velocity',
                                          level, da_X.values, da_Y.values, da_Z.values, title=None,
                                          flds_min_value=-1.2, flds_max_value=1.2, diff_min_value=-0.5, diff_max_value=0.5, extend='both' )
                                          #flds_min_value=-1.2, flds_max_value=1.2, diff_min_value=-1.0, diff_max_value=1.0, extend='both' )
         plt.savefig(rootdir+'/ITERATED_FORECAST/PLOTS/'+model_name+'_V_level'+str(level)+'_time'+f'{time:03}'+'.png',
                     bbox_inches = 'tight', pad_inches = 0.1)
         plt.close()
         
         fig = ChnPlt.plot_depth_fld_diff(masked_True_Eta[time,:,:], 'MITgcm Sea Surface Height',
                                          masked_Pred_Eta[time,:,:], 'Predicted Sea Surface Height',
                                          0, da_X.values, da_Y.values, da_Z.values, title=None,
                                          flds_min_value=-1, flds_max_value=1, diff_min_value=-0.3, diff_max_value=0.3 )
                                          #flds_min_value=-1, flds_max_value=1, diff_min_value=-1.0, diff_max_value=1.0 )
         plt.savefig(rootdir+'/ITERATED_FORECAST/PLOTS/'+model_name+'_Eta_level'+str(level)+'_time'+f'{time:03}'+'.png', 
                     bbox_inches = 'tight', pad_inches = 0.1)
         plt.close()
   
   fig = ChnPlt.plt_timeseries( [], ts_end, RMS_dict, y_label='RMS errors', ylim=[0,1]) 
   plt.savefig(rootdir+'/ITERATED_FORECAST/'+model_name+'_RMS_timeseries_'+str(ts_end)+'.png',
               bbox_inches = 'tight', pad_inches = 0.1)

   count=count+1

#--------------------------------------------
# Read in data from perturbed runs if needed
#--------------------------------------------
if plot_timeseries or plot_rms_timeseries or plot_cc_timeseries: 
   pert_main_dir = '/data/hpcdata/users/racfur/MITgcm/verification/MundayChannelConfig10km_LandSpits/runs/'
   pert_filename = '/12hrly_data.nc'

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
      alphas.append(0.3)
      RMS_colors.append('grey')
      RMS_alphas.append(0.3)

#----------------------------
# Plot timeseries at a point
#----------------------------

# If length of models is only one long, then save it to the relevant dir. But otherwise, save to the multimodel dir....

if plot_timeseries: 
   fig = ChnPlt.plt_timeseries( point, ts_end, Temp_dict, y_label='Temperature ('+u'\xb0'+'C)', colors=colors, alphas=alphas )
   if len(dir_names)==1:
      plt.savefig(rootdir+'/ITERATED_FORECAST/'+model_name+'_Temp_timeseries_z'+str(point[0])+'y'+str(point[1])+'x'+str(point[2])+
                  '_'+str(ts_end)+'.png', bbox_inches = 'tight', pad_inches = 0.1)
   else:
      plt.savefig('../../../Channel_nn_Outputs/MULTIMODEL_PLOTS/Temp_timeseries_z'+str(point[0])+'y'+str(point[1])+'x'+str(point[2])+
                  '_'+str(ts_end)+'.png', bbox_inches = 'tight', pad_inches = 0.1)

   fig = ChnPlt.plt_timeseries( point, ts_end, UVel_dict, y_label='Eastward\nVelocity $(m s^{-1})$', colors=colors, alphas=alphas ) 
   if len(dir_names)==1:
      plt.savefig(rootdir+'/ITERATED_FORECAST/'+model_name+'_U_timeseries_z'+str(point[0])+'y'+str(point[1])+'x'+str(point[2])+
                  '_'+str(ts_end)+'.png', bbox_inches = 'tight', pad_inches = 0.1)
   else:
      plt.savefig('../../../Channel_nn_Outputs/MULTIMODEL_PLOTS/U_timeseries_z'+str(point[0])+'y'+str(point[1])+'x'+str(point[2])+
                  '_'+str(ts_end)+'.png', bbox_inches = 'tight', pad_inches = 0.1)

   fig = ChnPlt.plt_timeseries( point, ts_end, VVel_dict, y_label='Northward\nVelocity $(m s^{-1})$', colors=colors, alphas=alphas ) 
   if len(dir_names)==1:
      plt.savefig(rootdir+'/ITERATED_FORECAST/'+model_name+'_V_timeseries_z'+str(point[0])+'y'+str(point[1])+'x'+str(point[2])+
                  '_'+str(ts_end)+'.png', bbox_inches = 'tight', pad_inches = 0.1)
   else:
      plt.savefig('../../../Channel_nn_Outputs/MULTIMODEL_PLOTS/V_timeseries_z'+str(point[0])+'y'+str(point[1])+'x'+str(point[2])+
                  '_'+str(ts_end)+'.png', bbox_inches = 'tight', pad_inches = 0.1)
   
   fig = ChnPlt.plt_timeseries( point[1:], ts_end, Eta_dict, y_label='Sea Surface\nHeight $(m)$', colors=colors, alphas=alphas ) 
   if len(dir_names)==1:
      plt.savefig(rootdir+'/ITERATED_FORECAST/'+model_name+'_Eta_timeseries_z'+str(point[0])+'y'+str(point[1])+'x'+str(point[2])+
                  '_'+str(ts_end)+'.png', bbox_inches = 'tight', pad_inches = 0.1)
   else:
      plt.savefig('../../../Channel_nn_Outputs/MULTIMODEL_PLOTS/Eta_timeseries_z'+str(point[0])+'y'+str(point[1])+'x'+str(point[2])+
                  '_'+str(ts_end)+'.png', bbox_inches = 'tight', pad_inches = 0.1)

#----------------------------------------
# Plot spatially averaged RMS timeseries
#----------------------------------------
if plot_rms_timeseries:
   fig = ChnPlt.plt_timeseries( [], ts_end, RMS_Temp_dict, y_label='Temperature\nRMS error ('+u'\xb0'+'C)',
                                colors=RMS_colors, alphas=RMS_alphas, ylim=[0,1] )
   if len(dir_names)==1:
      plt.savefig(rootdir+'/ITERATED_FORECAST/'+model_name+'_Temp_RMS_timeseries_'+str(ts_end)+'.png',
                  bbox_inches = 'tight', pad_inches = 0.1)
   else:
      plt.savefig('../../../Channel_nn_Outputs/MULTIMODEL_PLOTS/Temp_RMS_timeseries_'+str(ts_end)+'.png',
                  bbox_inches = 'tight', pad_inches = 0.1)

   fig = ChnPlt.plt_timeseries( [], ts_end, RMS_UVel_dict, y_label='Eastward Velocity\nRMS error $(m s^{-1})$',
                                colors=RMS_colors, alphas=RMS_alphas, ylim=[0,0.3] ) 
   if len(dir_names)==1:
      plt.savefig(rootdir+'/ITERATED_FORECAST/'+model_name+'_U_RMS_timeseries_'+str(ts_end)+'.png',
                  bbox_inches = 'tight', pad_inches = 0.1)
   else:
      plt.savefig('../../../Channel_nn_Outputs/MULTIMODEL_PLOTS/U_RMS_timeseries_'+str(ts_end)+'.png',
                  bbox_inches = 'tight', pad_inches = 0.1)

   fig = ChnPlt.plt_timeseries( [], ts_end, RMS_VVel_dict, y_label='Northward Velocity\nRMS error $(m s^{-1})$',
                                colors=RMS_colors, alphas=RMS_alphas, ylim=[0,0.3]) 
   if len(dir_names)==1:
      plt.savefig(rootdir+'/ITERATED_FORECAST/'+model_name+'_V_RMS_timeseries_'+str(ts_end)+'.png',
                  bbox_inches = 'tight', pad_inches = 0.1)
   else:
      plt.savefig('../../../Channel_nn_Outputs/MULTIMODEL_PLOTS/V_RMS_timeseries_'+str(ts_end)+'.png',
                  bbox_inches = 'tight', pad_inches = 0.1)
   
   fig = ChnPlt.plt_timeseries( [], ts_end, RMS_Eta_dict, y_label='Sea Surface Height\nRMS error $(m)$',
                                colors=RMS_colors, alphas=RMS_alphas, ylim=[0,0.3]) 
   if len(dir_names)==1:
      plt.savefig(rootdir+'/ITERATED_FORECAST/'+model_name+'_Eta_RMS_timeseries_'+str(ts_end)+'.png',
                  bbox_inches = 'tight', pad_inches = 0.1)
   else:
      plt.savefig('../../../Channel_nn_Outputs/MULTIMODEL_PLOTS/Eta_RMS_timeseries_'+str(ts_end)+'.png',
                  bbox_inches = 'tight', pad_inches = 0.1)

#--------------------
# Plot CC timeseries
#--------------------
if plot_cc_timeseries:
   fig = ChnPlt.plt_timeseries( [], ts_end, CC_Temp_dict, y_label='Temperature\nCC coeffient ('+u'\xb0'+'C)',
                                colors=RMS_colors, alphas=RMS_alphas, ylim=[-0.1,1.1] )
   if len(dir_names)==1:
      plt.savefig(rootdir+'/ITERATED_FORECAST/'+model_name+'_Temp_CC_timeseries_'+str(ts_end)+'.png',
                  bbox_inches = 'tight', pad_inches = 0.1)
   else:
      plt.savefig('../../../Channel_nn_Outputs/MULTIMODEL_PLOTS/Temp_CC_timeseries_'+str(ts_end)+'.png',
                  bbox_inches = 'tight', pad_inches = 0.1)

   fig = ChnPlt.plt_timeseries( [], ts_end, CC_UVel_dict, y_label='Eastward Velocity\nCC error $(m s^{-1})$',
                                colors=RMS_colors, alphas=RMS_alphas, ylim=[-0.1,1.1] ) 
   if len(dir_names)==1:
      plt.savefig(rootdir+'/ITERATED_FORECAST/'+model_name+'_U_CC_timeseries_'+str(ts_end)+'.png',
                  bbox_inches = 'tight', pad_inches = 0.1)
   else:
      plt.savefig('../../../Channel_nn_Outputs/MULTIMODEL_PLOTS/U_CC_timeseries_'+str(ts_end)+'.png',
                  bbox_inches = 'tight', pad_inches = 0.1)

   fig = ChnPlt.plt_timeseries( [], ts_end, CC_VVel_dict, y_label='Northward Velocity\nCC error $(m s^{-1})$',
                                colors=RMS_colors, alphas=RMS_alphas, ylim=[-0.1,1.1]) 
   if len(dir_names)==1:
      plt.savefig(rootdir+'/ITERATED_FORECAST/'+model_name+'_V_CC_timeseries_'+str(ts_end)+'.png',
                  bbox_inches = 'tight', pad_inches = 0.1)
   else:
      plt.savefig('../../../Channel_nn_Outputs/MULTIMODEL_PLOTS/V_CC_timeseries_'+str(ts_end)+'.png',
                  bbox_inches = 'tight', pad_inches = 0.1)
   
   fig = ChnPlt.plt_timeseries( [], ts_end, CC_Eta_dict, y_label='Sea Surface Height\nCC error $(m)$',
                                colors=RMS_colors, alphas=RMS_alphas, ylim=[-0.1,1.1]) 
   if len(dir_names)==1:
      plt.savefig(rootdir+'/ITERATED_FORECAST/'+model_name+'_Eta_CC_timeseries_'+str(ts_end)+'.png',
                  bbox_inches = 'tight', pad_inches = 0.1)
   else:
      plt.savefig('../../../Channel_nn_Outputs/MULTIMODEL_PLOTS/Eta_CC_timeseries_'+str(ts_end)+'.png',
                  bbox_inches = 'tight', pad_inches = 0.1)

#-----------------------------------------------------------------
# Print out spatially averaged RMS values for various time points
#-----------------------------------------------------------------
count=0
if print_spatially_av_rms:
   for model in dir_names:
      print('')
      print('')
      print(labels[count])
      for time in print_times:
         print('Temp RMS at time '+str(time)+'; '+str(RMS_Temp_dict[labels[count]][time]))
         #print('UVel RMS at time '+str(time)+'; '+str(RMS_UVel_dict[labels[count]][time]))
         #print('VVel RMS at time '+str(time)+'; '+str(RMS_VVel_dict[labels[count]][time]))
         #print('Eta RMS at time '+str(time)+'; '+str(RMS_Eta_dict[labels[count]][time]))
      count=count+1

#----------------------------------------------------------
# Print out correlation coeficient for various time points
#----------------------------------------------------------
count=0
if print_cc:
   for model in dir_names:
      print('')
      print('')
      print(labels[count])
      for time in print_times:
         print('Temp CC at time '+str(time)+'; '+str(CC_Temp_dict[labels[count]][time]))
         #print('UVel CC at time '+str(time)+'; '+str(CC_UVel_dict[labels[count]][time]))
         #print('VVel CC at time '+str(time)+'; '+str(CC_VVel_dict[labels[count]][time]))
         #print('Eta CC at time '+str(time)+'; '+str(CC_Eta_dict[labels[count]][time]))
      count=count+1


