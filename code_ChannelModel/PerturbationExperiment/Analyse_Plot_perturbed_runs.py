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

import scipy.stats as stats

plt.rcParams.update({'font.size': 14})

#----------------------------
# Set variables for this run
#----------------------------
point = [ 2, 48, 120]

xaxis = [np.arange(0,90,.5), np.arange(0,90,.5), np.arange(0,90,.5), np.arange(0,90,.5)]
ts_ends = [90*2, 90*2, 90*2, 90*2]
alphas = [ 1., 1., 1., 1. ]

plot_timeseries = True   
plot_rms_timeseries = True 
print_spatially_av_rms = True 
print_cc = True 
plot_cc_timeseries = True 
make_animation_plots = False 
animation_end = 6*2*7+1
ts_days = 90   

#-----------------------------------
# Read in stats file for mean field 
#-----------------------------------

#--------------
# Read in data
#--------------
Temp_dict={}
RMS_Temp_dict={}
NormRMS_Temp_dict={}
MAE_Temp_dict={}
NormMAE_Temp_dict={}
ME_Temp_dict={}
CC_Temp_dict={}

Cntrl_filename = '../../../Channel_nn_Outputs/PerturbationExp_Cntrl/ITERATED_FORECAST/PerturbationExp_Cntrl_200epochs_simple_smth0stps0_Forlen180.nc'
Perturb_filename = '../../../Channel_nn_Outputs/PerturbationExp_Perturbed/ITERATED_FORECAST/PerturbationExp_Perturbed_200epochs_simple_smth0stps0_Forlen180.nc'
rootdir = '../../../Channel_nn_Outputs/PerturbationPlots/'

Cntrl_ds = xr.open_dataset(Cntrl_filename)
da_X = Cntrl_ds['X']
da_Y = Cntrl_ds['Y']
da_Z = Cntrl_ds['Z']

da_Cntrl_Mitgcm_Temp = Cntrl_ds['True_Temp']
da_Temp_mask = Cntrl_ds['Temp_Mask']
masked_Cntrl_Mitgcm_Temp = np.where( da_Temp_mask.values==0, np.nan, da_Cntrl_Mitgcm_Temp.values )

da_Cntrl_UNet_Temp = Cntrl_ds['Pred_Temp']
da_Cntrl_UNet_Temp_errors = Cntrl_ds['Temp_Errors']
masked_Cntrl_UNet_Temp = np.where( da_Temp_mask.values==0, np.nan, da_Cntrl_UNet_Temp.values )

Perturb_ds = xr.open_dataset(Perturb_filename)

da_Perturb_Mitgcm_Temp = Perturb_ds['True_Temp']
da_Temp_mask = Perturb_ds['Temp_Mask']
masked_Perturb_Mitgcm_Temp = np.where( da_Temp_mask.values==0, np.nan, da_Perturb_Mitgcm_Temp.values )

da_Perturb_UNet_Temp = Perturb_ds['Pred_Temp']
da_Perturb_UNet_Temp_errors = Perturb_ds['Temp_Errors']
masked_Perturb_UNet_Temp = np.where( da_Temp_mask.values==0, np.nan, da_Perturb_UNet_Temp.values )

Temp_dict['Mitgcm Control Run'] = masked_Cntrl_Mitgcm_Temp
Temp_dict['Mitgcm Perturbed Run'] = masked_Perturb_Mitgcm_Temp
Temp_dict['UNet Control Run'] = masked_Cntrl_UNet_Temp
Temp_dict['UNet Perturbed Run'] = masked_Perturb_UNet_Temp

MITgcm_Temp_RMS = np.sqrt(np.nanmean( np.square(masked_Cntrl_Mitgcm_Temp[:ts_ends[0],:,:,:]-masked_Perturb_Mitgcm_Temp[:ts_ends[0],:,:,:]), axis=(1,2,3) ))
RMS_Temp_dict['MITgcm RMS difference'] = MITgcm_Temp_RMS
UNet_Temp_RMS = np.sqrt(np.nanmean( np.square(masked_Cntrl_UNet_Temp[:ts_ends[0],:,:,:]-masked_Perturb_UNet_Temp[:ts_ends[0],:,:,:]), axis=(1,2,3) ))
RMS_Temp_dict['UNet RMS difference'] = UNet_Temp_RMS

MITgcm_Temp_NormRMS = np.sqrt(np.nanmean( np.square(masked_Cntrl_Mitgcm_Temp[:ts_ends[0],:,:,:]-masked_Perturb_Mitgcm_Temp[:ts_ends[0],:,:,:])/
                                          masked_Perturb_Mitgcm_Temp[:ts_ends[0],:,:,:], axis=(1,2,3) ))
NormRMS_Temp_dict['MITgcm RMS difference'] = MITgcm_Temp_NormRMS
UNet_Temp_NormRMS = np.sqrt(np.nanmean( np.square(masked_Cntrl_UNet_Temp[:ts_ends[0],:,:,:]-masked_Perturb_UNet_Temp[:ts_ends[0],:,:,:])/
                                    masked_Perturb_UNet_Temp[:ts_ends[0],:,:,:], axis=(1,2,3) ))
NormRMS_Temp_dict['UNet RMS difference'] = UNet_Temp_RMS

MITgcm_Temp_MAE = np.nanmean( np.sqrt(np.square(masked_Cntrl_Mitgcm_Temp[:ts_ends[0],:,:,:]-masked_Perturb_Mitgcm_Temp[:ts_ends[0],:,:,:])), axis=(1,2,3) )
MAE_Temp_dict['MITgcm MAE difference'] = MITgcm_Temp_MAE
UNet_Temp_MAE = np.nanmean( np.sqrt(np.square(masked_Cntrl_UNet_Temp[:ts_ends[0],:,:,:]-masked_Perturb_UNet_Temp[:ts_ends[0],:,:,:])), axis=(1,2,3) )
MAE_Temp_dict['UNet MAE difference'] = UNet_Temp_MAE

MITgcm_Temp_ME = np.nanmean( masked_Cntrl_Mitgcm_Temp[:ts_ends[0],:,:,:]-masked_Perturb_Mitgcm_Temp[:ts_ends[0],:,:,:], axis=(1,2,3) )
ME_Temp_dict['MITgcm ME difference'] = MITgcm_Temp_ME
UNet_Temp_ME = np.nanmean( masked_Cntrl_UNet_Temp[:ts_ends[0],:,:,:]-masked_Perturb_UNet_Temp[:ts_ends[0],:,:,:], axis=(1,2,3) )
ME_Temp_dict['UNet ME difference'] = UNet_Temp_ME

if print_cc or plot_cc_timeseries:
   CC_Temp_dict['MITgcm CC'] = np.zeros(ts_ends[0])
   CC_Temp_dict['UNet CC'] = np.zeros(ts_ends[0])
   for time in range(ts_ends[0]):
      CC_Temp_dict['MITgcm CC'][time] = np.corrcoef( np.reshape(masked_Cntrl_Mitgcm_Temp[time,:,:,:][~np.isnan(masked_Cntrl_Mitgcm_Temp[time,:,:,:])],(-1)),
                                                     np.reshape(masked_Perturb_Mitgcm_Temp[time,:,:,:][~np.isnan(masked_Perturb_Mitgcm_Temp[time,:,:,:])],(-1)) )[0,1]
      CC_Temp_dict['UNet CC'][time] = np.corrcoef( np.reshape(masked_Cntrl_UNet_Temp[time,:,:,:][~np.isnan(masked_Cntrl_UNet_Temp[time,:,:,:])],(-1)),
                                                   np.reshape(masked_Perturb_UNet_Temp[time,:,:,:][~np.isnan(masked_Perturb_UNet_Temp[time,:,:,:])],(-1)) )[0,1]

##-------------------------------
## Plot spatial plots to animate
##-------------------------------
#if make_animation_plots:
#   for time in range(0,animation_end):
#      print(time)
#      fig = ChnPlt.plot_depth_fld_diff(masked_True_Temp[time,level,:,:], 'MITgcm Temperature',
#                                       masked_Pred_Temp[time,level,:,:], 'Predicted Temperature',
#                                       level, da_X.values, da_Y.values, da_Z.values, title=None,
#                                       flds_minmax = [0., 6.5], diff_minmax = [-1, 1], extend='max' )
#                                       #flds_minmax = 0., 6.5], diff_minmax = [-1, 1], extend='max' )
#      plt.savefig(rootdir+'/ITERATED_FORECAST/PLOTS/'+model_name+'_Temp_level'+str(level)+'_time'+f'{time:03}'+'.png',
#                  bbox_inches = 'tight', pad_inches = 0.1)
#      plt.close()
#
#   
#count=count+1
#
#---------------------------------------
print('Plotting timeseries at a point')
#---------------------------------------
# If length of models is only one long, then save it to the relevant dir. But otherwise, save to the multimodel dir....

if plot_timeseries: 

   fig = ChnPlt.plt_timeseries( point, ts_ends, Temp_dict, y_label='Temperature ('+u'\xb0'+'C)', colors=['red', 'magenta', 'blue', 'cyan'],
                                alphas=alphas, myfigsize=(20,10), x_label='number of days', xaxis=xaxis, ylim=[2.2,7])
   plt.savefig(rootdir+'/Pert_Temp_timeseries_z'+str(point[0])+'y'+str(point[1])+'x'+str(point[2])+
               '_'+str(ts_days)+'.png', bbox_inches = 'tight', pad_inches = 0.1)
   plt.close()

#---------------------------------------------------
print('plotting spatially averaged RMS timeseries')
#---------------------------------------------------
if plot_rms_timeseries:
   fig = ChnPlt.plt_timeseries( [], ts_ends, RMS_Temp_dict, y_label='Temperature\nRMS difference ('+u'\xb0'+'C)',
                                colors=['red', 'blue'], alphas=alphas, ylim=[0,1], x_label='number of days', xaxis=xaxis )
   plt.savefig(rootdir+'/Pert_Temp_RMS_timeseries_'+str(ts_days)+'.png',
               bbox_inches = 'tight', pad_inches = 0.1)
   plt.close()

   #fig = ChnPlt.plt_timeseries( [], ts_ends, NormRMS_Temp_dict, y_label='Temperature\nRMS difference ('+u'\xb0'+'C)',
   #                             colors=['red', 'blue'], alphas=alphas, ylim=[0,1], x_label='number of days', xaxis=xaxis )
   #plt.savefig(rootdir+'/Pert_Temp_NormRMS_timeseries_'+str(ts_days)+'.png',
   #            bbox_inches = 'tight', pad_inches = 0.1)
   #plt.close()

##---------------------------------------------------
#print('plotting spatially averaged MAE timeseries')
##---------------------------------------------------
#if plot_rms_timeseries:
#   fig = ChnPlt.plt_timeseries( [], ts_ends, MAE_Temp_dict, y_label='Temperature\nMean absolute difference ('+u'\xb0'+'C)',
#                                colors=['red', 'blue'], alphas=alphas, ylim=[0,1], x_label='number of days', xaxis=xaxis )
#   plt.savefig(rootdir+'/Pert_Temp_MAE_timeseries_'+str(ts_days)+'.png',
#               bbox_inches = 'tight', pad_inches = 0.1)
#   plt.close()
#
##---------------------------------------------------
#print('plotting spatially averaged ME timeseries')
##---------------------------------------------------
#if plot_rms_timeseries:
#   fig = ChnPlt.plt_timeseries( [], ts_ends, ME_Temp_dict, y_label='Temperature\nMean difference ('+u'\xb0'+'C)',
#                                colors=['red', 'blue'], alphas=alphas, ylim=[-0.005,0.001], x_label='number of days', xaxis=xaxis )
#   plt.savefig(rootdir+'/Pert_Temp_ME_timeseries_'+str(ts_days)+'.png',
#               bbox_inches = 'tight', pad_inches = 0.1)
#   plt.close()
#
#-------------------------------
print('plotting CC timeseries')
#-------------------------------
if plot_cc_timeseries:
   fig = ChnPlt.plt_timeseries( [], ts_ends, CC_Temp_dict, y_label='Temperature\nCC coeffient ('+u'\xb0'+'C)',
                                colors=['red', 'blue'], alphas=alphas, ylim=[0.95,1.005], x_label='number of days', xaxis=xaxis )
   plt.savefig(rootdir+'/Pert_Temp_CC_timeseries_'+str(ts_days)+'.png',
               bbox_inches = 'tight', pad_inches = 0.1)
   plt.close()


