#!/usr/bin/env python
# coding: utf-8

# Part of a multifile set up to calculate Means and Std - split across mutliple parts to solve memory issues, not neat, but works!

print('import packages')
import sys
sys.path.append('../Tools')

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import netCDF4 as nc4
import gc as gc


init=True
var_range=range(4)
plot_histograms = True 
calc_stats = True
nc_stats = True

#------------------------

def plot_histograms(data_name, histogram_inputs, varname, file_varname):
   plt.rcParams.update({'font.size': 22})
   no_bins = 500
   fig_histogram = plt.figure(figsize=(10, 8))
   ax1 = fig_histogram.add_subplot(111)
   ax1.hist(histogram_inputs, bins = no_bins)
   ax1.set_title(varname+' Histogram')
   # ax1.set_ylim(top=y_top[var])
   plt.savefig('../../../Channel_nn_Outputs/HISTOGRAMS/'+data_name+'_histogram_'+file_varname+'_inputs.png', 
               bbox_inches = 'tight', pad_inches = 0.1)
   plt.close()

#------------------------
datadir  = '/data/hpcdata/users/racfur/MITgcm/verification/MundayChannelConfig10km_LandSpits/runs/50yr_Cntrl_orig/'
data_filename=datadir + '12hrly_data.nc'
out_filename = datadir+'/stats.nc'
#data_filename=datadir + '12hrly_small_set.nc'
#out_filename = datadir+'/stats_small.nc'

mean_std_file = '../../../Channel_nn_Outputs/MeanStd_TEST.npz'
   
VarName = ['Temperature', 'U Velocity', 'V Velocity', 'Sea Surface Height']
ShortVarName = ['Temp', 'UVel', 'VVel', 'Eta']
ncVarName = ['THETA', 'UVEL', 'VVEL', 'ETAN']

if init:
   if calc_stats:
       input_mean  = np.zeros((4))
       input_std   = np.zeros((4))
       input_range = np.zeros((4))
       target_mean  = np.zeros((4))
       target_std   = np.zeros((4))
       target_range = np.zeros((4))
       np.savez( mean_std_file, 
                 input_mean, input_std, input_range,
                 target_mean, target_std, target_range,
               ) 
   
   if nc_stats:
      ds_inputs = xr.open_dataset(data_filename)
      ds_inputs = ds_inputs.isel( T=slice( 0, int(0.75*ds_inputs.dims['T']) ) )
      out_file = nc4.Dataset(out_filename,'w', format='NETCDF4') #'w' stands for write
      
      # Create Dimensions
      da_Z = ds_inputs['Zmd000038']
      da_Y = ds_inputs['Y']
      da_Yp1 = ds_inputs['Yp1']
      da_X = ds_inputs['X']
      da_Xp1 = ds_inputs['Xp1']
      out_file.createDimension('Z', da_Z.shape[0])
      out_file.createDimension('Y', da_Y.shape[0])
      out_file.createDimension('Yp1', da_Yp1.shape[0])
      out_file.createDimension('X', da_X.shape[0])
      out_file.createDimension('Xp1', da_Xp1.shape[0])
      
      # Create dimension variables
      nc_Z = out_file.createVariable('Z', 'i4', 'Z')
      nc_Y = out_file.createVariable('Y', 'i4', 'Y')  
      nc_Yp1 = out_file.createVariable('Yp1', 'i4', 'Yp1')  
      nc_X = out_file.createVariable('X', 'i4', 'X')
      nc_Xp1 = out_file.createVariable('Xp1', 'i4', 'Xp1')
      nc_Z[:] = da_Z.values
      nc_Yp1[:] = da_Yp1.values
      nc_Y[:] = da_Y.values
      nc_Xp1[:] = da_Xp1.values
      nc_X[:] = da_X.values
      out_file.close()
      del da_Z
      del da_Y
      del da_Yp1
      del da_X
      del da_Xp1
      del nc_Z 
      del nc_Y
      del nc_Yp1
      del nc_X
      del nc_Xp1
      gc.collect()
   
for var in var_range:
   print(var)
   print(ncVarName[var])

   ds_inputs = xr.open_dataset(data_filename)
   da = ds_inputs[ncVarName[var]]
   print(da.shape)

   if nc_stats:
      out_file = nc4.Dataset(out_filename,'r+', format='NETCDF4') 
      if var == 0:
         nc_Mean = out_file.createVariable( 'Mean'+ShortVarName[var]  , 'f4', ('Z', 'Y', 'X') )
         nc_Std = out_file.createVariable( 'Std'+ShortVarName[var]  , 'f4', ('Z', 'Y', 'X') )
         nc_Mean[:,:,:] = np.nanmean(da, 0)
         nc_Std[:,:,:] = np.nanstd(da, 0)
      elif var == 1:
         nc_Mean = out_file.createVariable( 'Mean'+ShortVarName[var]  , 'f4', ('Z', 'Y', 'Xp1') )
         nc_Std = out_file.createVariable( 'Std'+ShortVarName[var]  , 'f4', ('Z', 'Y', 'Xp1') )
         nc_Mean[:,:,:] = np.nanmean(da, 0)
         nc_Std[:,:,:] = np.nanstd(da, 0)
      elif var == 2:
         nc_Mean = out_file.createVariable( 'Mean'+ShortVarName[var]  , 'f4', ('Z', 'Yp1', 'X') )
         nc_Std = out_file.createVariable( 'Std'+ShortVarName[var]  , 'f4', ('Z', 'Yp1', 'X') )
         nc_Mean[:,:,:] = np.nanmean(da, 0)
         nc_Std[:,:,:] = np.nanstd(da, 0)
      elif var == 3:
         nc_Mean = out_file.createVariable( 'Mean'+ShortVarName[var]  , 'f4', ('Y', 'X') )
         nc_Std = out_file.createVariable( 'Std'+ShortVarName[var]  , 'f4', ('Y', 'X') )
         nc_Mean[:,:] = np.nanmean(da, 0)
         nc_Std[:,:] = np.nanstd(da, 0)
      out_file.close()
      del nc_Mean
      del nc_Std
      gc.collect()

   targets  = da.values[1:]-da.values[:-1]
   if plot_histograms:
      plot_histograms('Spits', da.values.reshape(-1), VarName[var], ShortVarName[var]+'Inputs')
      plot_histograms('Spits', targets.reshape(-1), VarName[var]+' targets', ShortVarName[var]+'Targets')

   if calc_stats:
      mean_std_data = np.load(mean_std_file)
      inputs_mean  = mean_std_data['arr_0']
      inputs_std   = mean_std_data['arr_1']
      inputs_range = mean_std_data['arr_2']
      targets_mean  = mean_std_data['arr_3']
      targets_std   = mean_std_data['arr_4']
      targets_range = mean_std_data['arr_5']

      input_mean[var]  = np.nanmean(da.values)
      input_std[var]   = np.nanstd(da.values)
      input_range[var] = np.amax(da.values) - np.amin(da.values)
      target_mean[var]  = np.nanmean(targets)
      target_std[var]   = np.nanstd(targets)
      target_range[var] = np.amax(targets) - np.amin(targets)

      np.savez( mean_std_file, 
                input_mean, input_std, input_range,
                target_mean, target_std, target_range,
              ) 

   del da
   del targets
   gc.collect()

