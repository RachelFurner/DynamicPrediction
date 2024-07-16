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

import scipy.stats as stats

plt.rcParams.update({'font.size': 14})

#----------------------------
# Set variables for this run
#----------------------------
point = [ 2, 48, 120]

dir_names = ['IncLand12hrly_UNet2dtransp_histlen1_rolllen1_seed30475',
             'IncLand12hrly_UNet2dtransp_histlen1_rolllen3_seed30475',
             'IncLand12hrly_UNet2dtransp_histlen3_rolllen1_seed30475',
             'IncLand12hrly_UNetConvLSTM_histlen3_rolllen1_seed30475',
             'IncLand12hrly_UNet2dtransp_histlen1_rolllen1_seed30475',
             'IncLand12hrly_UNet2dtransp_histlen1_rolllen1_seed30475',
             'MultiModel_average_IncLand12hrly_UNet2dtransp_histlen1_rolllen1',
             'MultiModel_random_IncLand12hrly_UNet2dtransp_histlen1_rolllen1']
labels = ['Standard UNet', 'Rollout loss UNet', 'Past fields UNet', 'ConvLSTM', 
          'Smoothed UNet', 'AB2 iterated UNet', 'Multi-model average', 'Random multi-model']
epochs = ['200', '200', '200', '200',
          '200', '200', '200', '200']
iteration_methods = ['simple', 'simple', 'simple', 'simple',
                     'simple', 'AB2', 'simple', 'simple']
smooth_levels = ['0', '0', '0', '0', 
                 '20', '0', '0', '0']
smooth_steps = '0'
iteration_len = [180, 180, 180, 180, 180, 180, 180, 180]  
ts_days = 90  
my_xaxis = [np.arange(1,ts_days+1,.5), np.arange(1,ts_days+1,.5), np.arange(1,ts_days+1,.5), np.arange(1,ts_days+1,.5),
            np.arange(1,ts_days+1,.5), np.arange(1,ts_days+1,.5), np.arange(1,ts_days+1,.5), np.arange(1,ts_days+1,.5)]
my_ts_ends = [ts_days*2, ts_days*2, ts_days*2, ts_days*2, ts_days*2, ts_days*2, ts_days*2, ts_days*2]

#dir_names = ['MultiModel_average_IncLand12hrly_UNet2dtransp_histlen1_rolllen1']
#labels = ['Multi-model average']
#epochs = ['200']
#iteration_methods = ['simple']
#smooth_levels = ['0']
#smooth_steps = '0'
#iteration_len = [500]  
#ts_days = 250 
#my_xaxis = [np.arange(1,ts_days+1,.5)]
#my_ts_ends = [ts_days*2]

my_colors = ['red', 'blue', 'green', 'cyan', 'magenta', 'lime', 'brown', 'orange']
my_alphas = [ 1, 1, 1, 1, 1, 1, 1, 1, 1 ]
my_linestyles = ['solid', 'solid', 'solid', 'solid', 'solid', 'solid', 'solid', 'solid']

plot_timeseries = True   
plot_rms_timeseries = True 
print_spatially_av_rms = True 
print_cc = True 
plot_cc_timeseries = True 
make_animation_plots = False # Plots for thesis come from the other plotting routine.
power_spectrum = False
plot_conserved = False
animation_end = 6*2*7+1
print_times = [28, 56, 84, 112]  # times to print, in terms of how many 12 hour jumps; 2,4,6,8 weeks

plot_persistence=True 
plot_climatology=True 
#pert_list = ['50yr_smooth20', '50yr_smooth40', '50yr_smooth60', '50yr_smooth80', '50yr_smooth100', '50yr_smooth125', '50yr_smooth150']
pert_list = []

#---------------------------
level = point[0]
y_coord = point[1]
x_coord = point[2]

Temp_dict = {}
Mass_dict = {}
TKE_dict = {}

RMS_Temp_dict = {}
CC_Temp_dict = {}

colors = []
linestyles = []
alphas = []
xaxis = []
ts_ends = []
RMS_colors = []
RMS_linestyles = []
RMS_alphas = []
RMS_xaxis = []
RMS_ts_ends = []

#-------------------------------------
print('reading in NN iteration data')
#-------------------------------------
count=0
for model in dir_names:
   print(' ')
   print(' ')
   print(model)
   print(' ')
   model_name = model+'_'+epochs[count]+'epochs_'+iteration_methods[count]+'_smth'+smooth_levels[count]+'stps'+smooth_steps
   rootdir = '../../../Channel_nn_Outputs/'+model
   
   iter_data_filename=rootdir+'/ITERATED_FORECAST/'+model_name+'_Forlen'+str(iteration_len[count])+'.nc'
   print(iter_data_filename)
   iter_ds = xr.open_dataset(iter_data_filename)
   da_X = iter_ds['X']
   da_Y = iter_ds['Y']
   da_Z = iter_ds['Z']

   da_true_Temp = iter_ds['True_Temp']
   da_Temp_mask   = iter_ds['Temp_Mask']
   masked_True_Temp = np.where( da_Temp_mask.values==0, np.nan, da_true_Temp.values )

   da_true_Eta = iter_ds['True_Eta']
   masked_True_Eta = np.where( da_Temp_mask.values[0,:,:]==0, np.nan, da_true_Eta.values )

   da_true_U = iter_ds['True_U']
   da_U_mask   = iter_ds['U_Mask']
   masked_True_U = np.where( da_U_mask.values==0, np.nan, da_true_U.values )
   da_true_V = iter_ds['True_V']
   da_V_mask   = iter_ds['V_Mask']
   masked_True_V = np.where( da_V_mask.values==0, np.nan, da_true_V.values )

   if count==0:
      Temp_dict['MITgcm Temperature'] = masked_True_Temp[:ts_days*2]
      Mass_dict['MITgcm SSH'] = np.nansum( masked_True_Eta[:ts_days*2], axis=(1,2) )
      TKE_dict['MITgcm TKE'] = np.nansum( np.add( np.square(masked_True_U[:ts_days*2]), np.square(masked_True_V[:ts_days*2]) ), axis=(1,2,3) )
      colors.append('black')
      linestyles.append('solid')
      alphas.append(1)
      xaxis.append(np.arange(1,ts_days+1,.5))
      ts_ends.append(np.shape(xaxis[-1])[0])
      if plot_persistence == True:
         pers = np.zeros(masked_True_Temp.shape)
         pers[:,:,:,:] = masked_True_Temp[0,:,:,:]
         Temp_dict['Persistence'] = pers[:ts_days*2,:,:,:]
         colors.append('grey')
         linestyles.append('dashed')
         alphas.append(1)
         xaxis.append(np.arange(1,ts_days+1,.5))
         ts_ends.append(np.shape(xaxis[-1])[0])
      if plot_climatology == True:
         stats_filename='/data/hpcdata/users/racfur/MITgcm/verification/MundayChannelConfig10km_noSpits/runs/50yr_Cntrl/IncLand_stats.nc'
         stats_ds = xr.open_dataset(stats_filename)
         da_MeanTemp = stats_ds['MeanTemp']
         clim = np.zeros(masked_True_Temp.shape)
         clim[:,:,:,:] = da_MeanTemp[:,:,:]
         Temp_dict['Climatology'] = clim[:ts_days*2]
         colors.append('grey')
         linestyles.append('dotted')
         alphas.append(1)
         xaxis.append(np.arange(1,ts_days+1,.5))
         ts_ends.append(np.shape(xaxis[-1])[0])

   xaxis.append(my_xaxis[count])
   ts_ends.append(my_ts_ends[count])

   da_pred_Temp = iter_ds['Pred_Temp']
   da_Temp_errors = iter_ds['Temp_Errors']
   masked_Pred_Temp = np.where( da_Temp_mask.values==0, np.nan, da_pred_Temp.values )
   Temp_dict[labels[count]] = masked_Pred_Temp[:my_ts_ends[count]]

   da_pred_Eta = iter_ds['Pred_Eta']
   masked_Pred_Eta = np.where( da_Temp_mask.values[0,:,:]==0, np.nan, da_pred_Eta.values )
   Mass_dict[labels[count]] = np.nansum( masked_Pred_Eta[:my_ts_ends[count]], axis=(1,2) )

   da_pred_U = iter_ds['Pred_U']
   masked_Pred_U = np.where( da_U_mask.values==0, np.nan, da_pred_U.values )
   da_pred_V = iter_ds['Pred_V']
   masked_Pred_V = np.where( da_V_mask.values==0, np.nan, da_pred_V.values )
   TKE_dict[labels[count]] = np.nansum( np.add( np.square(masked_Pred_U[:my_ts_ends[count]]), np.square(masked_Pred_V[:my_ts_ends[count]]) ), axis=(1,2,3) )

   colors.append(my_colors[count])
   linestyles.append(my_linestyles[count])
   alphas.append(my_alphas[count])
   
   if count==0:
      if plot_persistence == True:
         pers_RMS = np.sqrt(np.nanmean( np.square(pers[:ts_days*2,:,:,:]-masked_True_Temp[:ts_days*2,:,:,:]), axis=(1,2,3) ))
         RMS_Temp_dict['Persistence'] = pers_RMS
         RMS_colors.append('grey')
         RMS_linestyles.append('dashed')
         RMS_alphas.append(1)
         RMS_xaxis.append(np.arange(1,ts_days+1,.5))
         RMS_ts_ends.append(np.shape(xaxis[-1])[0])
         if print_cc or plot_cc_timeseries:
            CC_Temp_dict['Persistence'] = np.zeros(my_ts_ends[count])
            for time in range(my_ts_ends[count]):
               CC_Temp_dict['Persistence'][time] = np.corrcoef( np.reshape(pers[time,:,:,:][~np.isnan(masked_True_Temp[time,:,:,:])],(-1)),
                                                                np.reshape(masked_True_Temp[time,:,:,:][~np.isnan(masked_True_Temp[time,:,:,:])],(-1)) )[0,1]
      if plot_climatology == True:
         clim_RMS = np.sqrt(np.nanmean( np.square(clim[:ts_days*2,:,:,:]-masked_True_Temp[:ts_days*2,:,:,:]), axis=(1,2,3) ))
         RMS_Temp_dict['Climatology'] = clim_RMS
         RMS_colors.append('grey')
         RMS_linestyles.append('dotted')
         RMS_alphas.append(1)
         RMS_xaxis.append(np.arange(1,ts_days+1,.5))
         RMS_ts_ends.append(np.shape(xaxis[-1])[0])
         if print_cc or plot_cc_timeseries:
            CC_Temp_dict['Climatology'] = np.zeros(my_ts_ends[count])
            for time in range(my_ts_ends[count]):
               CC_Temp_dict['Climatology'][time] = np.corrcoef( np.reshape(clim[time,:,:,:][~np.isnan(masked_True_Temp[time,:,:,:])],(-1)),
                                                                np.reshape(masked_True_Temp[time,:,:,:][~np.isnan(masked_True_Temp[time,:,:,:])],(-1)) )[0,1]
         
   Temp_RMS = np.sqrt(np.nanmean( np.square(masked_Pred_Temp[:my_ts_ends[count],:,:,:]-masked_True_Temp[:my_ts_ends[count],:,:,:]), axis=(1,2,3) ))
   RMS_Temp_dict[labels[count]] = Temp_RMS
   if print_cc or plot_cc_timeseries:
      CC_Temp_dict[labels[count]] = np.zeros(my_ts_ends[count])
      for time in range(my_ts_ends[count]):
         CC_Temp_dict[labels[count]][time] = np.corrcoef( np.reshape(masked_Pred_Temp[time,:,:,:][~np.isnan(masked_Pred_Temp[time,:,:,:])],(-1)),
                                                          np.reshape(masked_True_Temp[time,:,:,:][~np.isnan(masked_True_Temp[time,:,:,:])],(-1)) )[0,1]

   RMS_colors.append(my_colors[count])
   RMS_linestyles.append('solid')
   RMS_alphas.append(1)
   RMS_xaxis.append(my_xaxis[count])
   RMS_ts_ends.append(my_ts_ends[count])

   #-------------------------------
   # Plot spatial plots to animate
   #-------------------------------
   if make_animation_plots:
      for time in range(0,animation_end):
         print(time)
         fig = ChnPlt.plot_depth_fld_diff(masked_True_Temp[time,level,:,:], 'MITgcm Temperature',
                                          masked_Pred_Temp[time,level,:,:], 'Predicted Temperature',
                                          level, da_X.values, da_Y.values, da_Z.values, title=None,
                                          flds_minmax = [0., 6.5], diff_minmax = [-1, 1], extend='max' )
                                          #flds_minmax = 0., 6.5], diff_minmax = [-1, 1], extend='max' )
         plt.savefig(rootdir+'/ITERATED_FORECAST/PLOTS/'+model_name+'_Temp_level'+str(level)+'_time'+f'{time:03}'+'.png',
                     bbox_inches = 'tight', pad_inches = 0.1)
         plt.close()
   
      
   count=count+1

#------------------------------------------------------
print('Reading in data from perturbed runs if needed')
#------------------------------------------------------
if plot_timeseries or plot_rms_timeseries or plot_cc_timeseries or power_spectrum or plot_conserved: 
   pert_main_dir = '/data/hpcdata/users/racfur/MITgcm/verification/MundayChannelConfig10km_noSpits/runs/'
   pert_filename = '/12hrly_data.nc'

   for pert in pert_list:
      print(pert)
      pert_file =  pert_main_dir+pert+pert_filename
      Pert_ds = xr.open_dataset(pert_file).isel( T=slice( 11, int(ts_days*2)+11 ) )
   
      da_PertTemp = Pert_ds['THETA']
      masked_PertTemp = np.where( da_Temp_mask.values==0, np.nan, da_PertTemp.values )
      Temp_dict[pert] = masked_PertTemp[:ts_days*2]
      RMS_Temp_dict[pert] =  np.sqrt(np.nanmean( np.square(masked_PertTemp[:ts_days*2,:,:,:]-masked_True_Temp[:ts_days*2,:,:,:]), axis=(1,2,3) ))
   
      if print_cc or plot_cc_timeseries:
         CC_Temp_dict[pert] = np.zeros(ts_days*2)
         for time in range(ts_days*2):
            CC_Temp_dict[pert][time] = np.corrcoef( np.reshape(masked_PertTemp[time,:,:,:][~np.isnan(masked_PertTemp[time,:,:,:])],(-1)),
                                                             np.reshape(masked_True_Temp[time,:,:,:][~np.isnan(masked_True_Temp[time,:,:,:])],(-1)) )[0,1]
   
      da_PertEta = Pert_ds['ETAN']
      masked_PertEta = np.where( da_Temp_mask.values[0,:,:]==0, np.nan, da_PertEta.values )
      Mass_dict[pert] = np.nansum( masked_PertEta[:ts_days*2], axis=(1,2,3) )

      da_PertU = Pert_ds['UVEL']
      masked_PertU = np.where( da_U_mask.values==0, np.nan, da_PertU.values[:,:,:,:240] )  # Ignore last U value as a repeat of the first
      da_PertV = Pert_ds['VVEL']
      masked_PertV = np.where( da_V_mask.values==0, np.nan, da_PertV.values[:,:,:104,:] )  # Ignore last line -- its land
      TKE_dict[pert] = np.nansum( np.add( np.square(masked_PertU[:ts_days*2]), np.square(masked_PertV[:ts_days*2]) ), axis=(1,2,3) )

      colors.append('grey')
      alphas.append(0.3)
      xaxis.append(np.arange(1,ts_days+1,.5))
      ts_ends.append(np.shape(xaxis[-1])[0])
      linestyles.append('solid')
      RMS_colors.append('grey')
      RMS_alphas.append(0.3)
      RMS_xaxis.append(np.arange(1,ts_days+1,.5))
      RMS_ts_ends.append(np.shape(xaxis[-1])[0])
      RMS_linestyles.append('solid')


print(Temp_dict.keys())
print(RMS_Temp_dict.keys())
#---------------------------------------
print('Plotting timeseries at a point')
#---------------------------------------
# If length of models is only one long, then save it to the relevant dir. But otherwise, save to the multimodel dir....

if plot_timeseries: 
   fig = ChnPlt.plt_timeseries( point, ts_ends, Temp_dict, y_label='Temperature ('+u'\xb0'+'C)', colors=colors, alphas=alphas, myfigsize=(20,8),
                                x_label='number of days', xaxis=xaxis, ylim=[1., 6.5], linestyles=linestyles)
                                #x_label='number of days', xaxis=xaxis, ylim=[2., 5.2], linestyles=linestyles)
                                #x_label='number of days', xaxis=xaxis, ylim=[2.9,4.0], linestyles=linestyles) #manually force xlim in Channel_model_plotting
   if len(dir_names)==1:
      plt.savefig(rootdir+'/ITERATED_FORECAST/'+model_name+'_Temp_timeseries_z'+str(point[0])+'y'+str(point[1])+'x'+str(point[2])+
                  '_'+str(ts_days)+'.png', bbox_inches = 'tight', pad_inches = 0.1)
   else:
      plt.savefig('../../../Channel_nn_Outputs/MULTIMODEL_PLOTS/Temp_timeseries_z'+str(point[0])+'y'+str(point[1])+'x'+str(point[2])+
                  '_'+str(ts_days)+'.png', bbox_inches = 'tight', pad_inches = 0.1)
   plt.close()

#---------------------------------------------------
print('plotting spatially averaged RMS timeseries')
#---------------------------------------------------
if plot_rms_timeseries:
   fig = ChnPlt.plt_timeseries( [], RMS_ts_ends, RMS_Temp_dict, y_label='Temperature\nRMS error ('+u'\xb0'+'C)',
                                colors=RMS_colors, alphas=RMS_alphas, ylim=[0,4.5], x_label='number of days', xaxis=RMS_xaxis, linestyles=RMS_linestyles )
                                #colors=RMS_colors, alphas=RMS_alphas, ylim=[0,1.2], x_label='number of days', xaxis=RMS_xaxis, linestyles=RMS_linestyles )
                                #colors=RMS_colors, alphas=RMS_alphas, ylim=[0,.5], x_label='number of days', xaxis=RMS_xaxis, linestyles=RMS_linestyles )
   if len(dir_names)==1:
      plt.savefig(rootdir+'/ITERATED_FORECAST/'+model_name+'_Temp_RMS_timeseries_'+str(ts_days)+'.png',
                  bbox_inches = 'tight', pad_inches = 0.1)
   else:
      plt.savefig('../../../Channel_nn_Outputs/MULTIMODEL_PLOTS/Temp_RMS_timeseries_'+str(ts_days)+'.png',
                  bbox_inches = 'tight', pad_inches = 0.1)
   plt.close()

#-------------------------------
print('plotting CC timeseries')
#-------------------------------
if plot_cc_timeseries:
   fig = ChnPlt.plt_timeseries( [], RMS_ts_ends, CC_Temp_dict, y_label='Temperature\nCC coeffient ('+u'\xb0'+'C)',
                                colors=RMS_colors, alphas=RMS_alphas, ylim=[-0.1,1.1], x_label='number of days', xaxis=RMS_xaxis, linestyles=RMS_linestyles )
                                #colors=RMS_colors, alphas=RMS_alphas, ylim=[0.96,1.005], x_label='number of days', xaxis=RMS_xaxis, linestyles=RMS_linestyles )
   if len(dir_names)==1:
      plt.savefig(rootdir+'/ITERATED_FORECAST/'+model_name+'_Temp_CC_timeseries_'+str(ts_days)+'.png',
                  bbox_inches = 'tight', pad_inches = 0.1)
   else:
      plt.savefig('../../../Channel_nn_Outputs/MULTIMODEL_PLOTS/Temp_CC_timeseries_'+str(ts_days)+'.png',
                  bbox_inches = 'tight', pad_inches = 0.1)
   plt.close()

#---------------------------------------------------------------------------
print('printing out spatially averaged RMS values for various time points')
#---------------------------------------------------------------------------
if print_spatially_av_rms:
   count=0
   for model in dir_names:
      print('')
      print('')
      print(labels[count])
      for time in print_times:
         print('Temp RMS at time '+str(time)+'; '+str(RMS_Temp_dict[labels[count]][time]))
      count=count+1

   count=0
   for model in dir_names:
      print('')
      print('')
      print(labels[count])
      print( 'Temp RMS exceeds 0.5 at point; '+str( np.argmax( RMS_Temp_dict[labels[count]] > .5 ) ) )
      print( 'Temp RMS exceeds 3 at point; '+str( np.argmax( RMS_Temp_dict[labels[count]] > 3. ) ) )
      count=count+1

#--------------------------------------------------------------------
print('Printing out correlation coeficient for various time points')
#--------------------------------------------------------------------
if print_cc:
   count=0
   for model in dir_names:
      print('')
      print('')
      print(labels[count])
      for time in print_times:
         print('Temp CC at time '+str(time)+'; '+str(CC_Temp_dict[labels[count]][time]))
      count=count+1

   count=0
   for model in dir_names:
      print('')
      print('')
      print(labels[count])
      print( 'Temp CC drops below .95 at point; '+str( np.argmax( CC_Temp_dict[labels[count]] < .95 ) ) )
      print( 'Temp CC drops below .8 at point; '+str( np.argmax( CC_Temp_dict[labels[count]] < .8 ) ) )
      count=count+1


#----------------------------
print('Plot power spectrum')
#----------------------------
if power_spectrum:
   print('done')

   #zonal_spectra = torch.fft.rfft2(psi.mean(-1)).abs().pow(2)
   #merid_spectra = torch.fft.rfft2(psi.mean(-2)).abs().pow(2)


   #print(Temp_dict.keys())
   ## Code based on https://bertvandenbroucke.netlify.app/2019/05/24/computing-a-power-spectrum-in-python/ 

   #npix = 240
   #kfreq = np.fft.fftfreq(npix) * npix
   #kfreq2D = np.meshgrid(kfreq, kfreq)
   #knrm = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2)
   #knrm = knrm.flatten()
   #kbins = np.arange(0.5, npix//2+1, 1.)
   #kvals = 0.5 * (kbins[1:] + kbins[:-1])

   #for time in [27, 59, 119, 179]:  # Plot after 2 weeks, 1,2,3 months

   #   my_legend=[]
   #   count=0

   #   # Take the fourier transform
   #   for name, dataset in Temp_dict.items():
   #      fourier_image = np.fft.fftn( dataset[time,2,3:101,:], s=(240, 240))    # Take just the ocean points§
   #      fourier_amplitudes = np.abs(fourier_image)**2
   #      fourier_amplitudes = fourier_amplitudes.flatten()
   #      Abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes,
   #                                     statistic = "mean",
   #                                     bins = kbins)
   #      Abins *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)
   #      my_legend.append(name)
   #      #if 'pert' not in name:
   #      #   my_legend.append(name)
   #
   #      plt.loglog(kvals, Abins, color=colors[count], alpha=alphas[count])

   #      count=count+1

   #   plt.xlabel("$k$")
   #   plt.ylabel("$P(k)$")
   #   plt.tight_layout()
   #   plt.legend(my_legend)
   #   if len(dir_names)==1:
   #      plt.savefig(rootdir+'/ITERATED_FORECAST/'+model_name+'_Temp_PowerSpectrum_'+str(time)+'.png',
   #                  bbox_inches = 'tight', pad_inches = 0.1)
   #   else:
   #      plt.savefig('../../../Channel_nn_Outputs/MULTIMODEL_PLOTS/Temp_PowerSpectrum_'+str(time)+'.png',
   #                  bbox_inches = 'tight', pad_inches = 0.1)
   #   plt.close()
  

#----------------------------------------------------------
# Plot timeseries of total ETA and TKE, summed over domain
#----------------------------------------------------------
if plot_conserved:
   fig = ChnPlt.plt_timeseries( [], ts_ends, Mass_dict, y_label='Total SSH Anomoly',
                                colors=colors, alphas=alphas, ylim=[-5000, 5000], yscale='symlog', x_label='number of days', xaxis=xaxis )
   if len(dir_names)==1:
      plt.savefig(rootdir+'/ITERATED_FORECAST/'+model_name+'_conservation_Eta.png',
                  bbox_inches = 'tight', pad_inches = 0.1)
   else:
      plt.savefig('../../../Channel_nn_Outputs/MULTIMODEL_PLOTS/conservation_Eta.png',
                  bbox_inches = 'tight', pad_inches = 0.1)
   plt.close()


   fig = ChnPlt.plt_timeseries( [], ts_ends, TKE_dict, y_label='Total Kinetic Energy',
                                colors=colors, alphas=alphas, ylim=[0, 150000], x_label='number of days', xaxis=xaxis)
   if len(dir_names)==1:
      plt.savefig(rootdir+'/ITERATED_FORECAST/'+model_name+'_conservation_TKE.png',
                  bbox_inches = 'tight', pad_inches = 0.1)
   else:
      plt.savefig('../../../Channel_nn_Outputs/MULTIMODEL_PLOTS/conservation_TKE.png',
                  bbox_inches = 'tight', pad_inches = 0.1)
   plt.close()
   
   
