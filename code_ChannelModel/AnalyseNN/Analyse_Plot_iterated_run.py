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

dir_names = ['IncLand12hrly_UNet2dtransp_histlen1_rolllen1_seed30475',
             'IncLand12hrly_UNet2dtransp_histlen1_rolllen1_seed30475',
             'IncLand12hrly_UNet2dtransp_histlen1_rolllen1_seed30475',
             'IncLand12hrly_UNet2dtransp_histlen1_rolllen1_seed30475',
             'MultiModel_IncLand12hrly_UNet2dtransp_histlen1_rolllen1',
             'MultiModel_IncLand12hrly_UNet2dtransp_histlen1_rolllen1',
             'IncLand12hrly_UNet2dtransp_histlen1_rolllen3_seed30475',
             'IncLand12hrly_UNet2dtransp_histlen3_rolllen1_seed30475',
             'IncLand12hrly_UNetConvLSTM_histlen3_rolllen1_seed30475']
labels = ['Standard CNN', 'Smoothed CNN', 'AB2 iterated CNN', 'AB3 iterated CNN', 'Multi-model average',
          'Random multi-model', 'Rollout loss CNN', 'Past fields CNN']#, 'ConvLSTM']
epochs = ['200', '200', '200', '200', '200', '200', '200', '200']
iteration_methods = ['simple', 'simple', 'AB2', 'AB3', 'simple', 'simple', 'simple', 'simple', 'simple']
smooth_levels = ['0', '20', '0', '0', '0', '0', '0', '0', '0']

#dir_names = ['IncLand12hrly_UNet2dtransp_histlen1_rolllen1_seed30475']
#labels = ['']
#epochs = ['200']
#iteration_methods = ['simple']
#smooth_levels = ['0']

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

RMS_Temp_dict = {}
CC_Temp_dict = {}

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
   da_pred_Temp = iter_ds['Pred_Temp']
   da_Temp_errors = iter_ds['Temp_Errors']
   da_Temp_mask   = iter_ds['Temp_Mask']
   da_X = iter_ds['X']
   da_Y = iter_ds['Y']
   da_Z = iter_ds['Z']
   
   masked_True_Temp = np.where( da_Temp_mask.values==0, np.nan, da_true_Temp.values )
   masked_Pred_Temp = np.where( da_Temp_mask.values==0, np.nan, da_pred_Temp.values )

   if count==0:
      Temp_dict['MITgcm Temperature'] = masked_True_Temp[:ts_end]
      colors.append('black')
      alphas.append(1)

   Temp_dict[labels[count]] = masked_Pred_Temp[:ts_end]

   colors.append(my_colors[count])
   alphas.append(my_alphas[count])
   
   Temp_RMS = np.sqrt(np.nanmean( np.square(masked_Pred_Temp[:ts_end,:,:,:]-masked_True_Temp[:ts_end,:,:,:]), axis=(1,2,3) ))

   RMS_Temp_dict[labels[count]] = Temp_RMS

   if print_cc or plot_cc_timeseries:
      CC_Temp_dict[labels[count]] = np.zeros(ts_end)
      for time in range(ts_end):
         CC_Temp_dict[labels[count]][time] = np.corrcoef( np.reshape(masked_Pred_Temp[time,:,:,:][~np.isnan(masked_Pred_Temp[time,:,:,:])],(-1)),
                                                          np.reshape(masked_True_Temp[time,:,:,:][~np.isnan(masked_True_Temp[time,:,:,:])],(-1)) )[0,1]

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
   
      
   count=count+1

#--------------------------------------------
# Read in data from perturbed runs if needed
#--------------------------------------------
if plot_timeseries or plot_rms_timeseries or plot_cc_timeseries: 
   pert_main_dir = '/data/hpcdata/users/racfur/MITgcm/verification/MundayChannelConfig10km_LandIncLand/runs/'
   pert_filename = '/12hrly_data.nc'

   for pert in pert_list:
      print(pert)
      pert_file =  pert_main_dir+pert+pert_filename
      Pert_ds = xr.open_dataset(pert_file).isel( T=slice( 0, int(ts_end) ) )
   
      da_PertTemp = Pert_ds['THETA']
      masked_PertTemp = np.where( da_Temp_mask.values==0, np.nan, da_PertTemp.values )
      Temp_dict[pert] = masked_PertTemp[:ts_end]
      RMS_Temp_dict[pert] =  np.sqrt(np.nanmean( np.square(masked_PertTemp[:ts_end,:,:,:]-masked_True_Temp[:ts_end,:,:,:]), axis=(1,2,3) ))
   
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
      count=count+1


