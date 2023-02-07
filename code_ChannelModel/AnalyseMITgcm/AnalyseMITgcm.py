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
import ReadRoutines as rr

for_jump = '12hrly'
#datadir = '/data/hpcdata/users/racfur/MITgcm/verification/MundayChannelConfig10km_LandSpits/runs/'+for_jump+'_output/'
datadir = '/data/hpcdata/users/racfur/MITgcm/verification/MundayChannelConfig10km_LandSpits/runs/50yr_Cntrl/'
data_filename=datadir + for_jump + '_data.nc'
#data_filename=datadir + for_jump + '_small_set.nc'
out_filename = datadir+'/stats.nc'
grid_filename = datadir+'grid.nc'

mean_std_file = '../../../Channel_nn_Outputs/DATASETS/new_'+for_jump+'_MeanStd.npz'

init=False
var_range=range(4)
plotting_histograms = True 
calc_stats = False
nc_stats = False

#------------------------

def plot_histograms(histogram_inputs, varname, file_varname, mean=None, std=None, datarange=None):
   plt.rcParams.update({'font.size': 22})
   no_bins = 500
   fig_histogram = plt.figure(figsize=(10, 8))
   ax1 = fig_histogram.add_subplot(111)
   ax1.hist(histogram_inputs, bins = no_bins)
   ax1.set_title(varname+' Histogram')
   # ax1.set_ylim(top=y_top[var])
   ax1.text(0.03, 0.94, 'Mean: '+str(np.format_float_scientific(mean, precision=3)), transform=ax1.transAxes, fontsize=14)
   ax1.text(0.03, 0.90, 'Standard deviation: '+str(np.format_float_scientific(std, precision=3)), transform=ax1.transAxes,fontsize=14)
   ax1.text(0.03, 0.86, 'Range: '+str(np.format_float_scientific(datarange, precision=3)), transform=ax1.transAxes, fontsize=14)
   plt.savefig('../../../Channel_nn_Outputs/HISTOGRAMS/'+for_jump+'_histogram_'+file_varname+'.png', 
               bbox_inches = 'tight', pad_inches = 0.1)
   plt.close()

#------------------------
   
VarName = ['Temperature', 'U Velocity', 'V Velocity', 'Sea Surface Height']
ShortVarName = ['Temp', 'UVel', 'VVel', 'Eta']
ncVarName = ['THETA', 'UVEL', 'VVEL', 'ETAN']
MaskVarName = ['HFacC', 'HFacW', 'HFacS', 'HFacC']

if init:
   if calc_stats:
       inputs_mean  = np.zeros((4))
       inputs_std   = np.zeros((4))
       inputs_range = np.zeros((4))
       targets_mean  = np.zeros((4))
       targets_std   = np.zeros((4))
       targets_range = np.zeros((4))
       np.savez( mean_std_file, 
                 inputs_mean, inputs_std, inputs_range,
                 targets_mean, targets_std, targets_range,
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
      out_file.createDimension('Z1', 1)
      
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
      #del da_Z
      #del da_Y
      #del da_Yp1
      #del da_X
      #del da_Xp1
      #del nc_Z 
      #del nc_Y
      #del nc_Yp1
      #del nc_X
      #del nc_Xp1
      #gc.collect()
   
for var in var_range:
   print(var)
   print(ncVarName[var])

   ds_grid = xr.open_dataset(grid_filename)
   da_mask = ds_grid[MaskVarName[var]]

   ds_inputs = xr.open_dataset(data_filename)
   ds_inputs = ds_inputs.isel( T=slice( 0, int(0.75*ds_inputs.dims['T']), 5 ) )
   da = ds_inputs[ncVarName[var]]
   if var == 3:
      da.values[:,:,:,:] = np.where( da_mask.values[0:1,:,:] > 0., da.values[:,:,:,:], np.nan )
   else:
      da.values[:,:,:,:] = np.where( da_mask.values[:,:,:] > 0., da.values[:,:,:,:], np.nan )
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
         nc_Mean = out_file.createVariable( 'Mean'+ShortVarName[var]  , 'f4', ('Z1', 'Y', 'X') )
         nc_Std = out_file.createVariable( 'Std'+ShortVarName[var]  , 'f4', ('Z1', 'Y', 'X') )
         nc_Mean[:,:,:] = np.nanmean(da, 0)
         nc_Std[:,:,:] = np.nanstd(da, 0)
      out_file.close()
      #del nc_Mean
      #del nc_Std
      #gc.collect()

   if calc_stats:
      mean_std_data = np.load(mean_std_file)
      inputs_mean  = mean_std_data['arr_0']
      inputs_std   = mean_std_data['arr_1']
      inputs_range = mean_std_data['arr_2']
      targets_mean  = mean_std_data['arr_3']
      targets_std   = mean_std_data['arr_4']
      targets_range = mean_std_data['arr_5']

      inputs_mean[var]  = np.nanmean(da.values)
      inputs_std[var]   = np.nanstd(da.values)
      inputs_range[var] = np.nanmax(da.values) - np.nanmin(da.values)
      targets_mean[var]  = np.nanmean(da.values[1:]-da.values[:-1])
      targets_std[var]   = np.nanstd(da.values[1:]-da.values[:-1])
      targets_range[var] = np.nanmax(da.values[1:]-da.values[:-1]) - np.nanmin(da.values[1:]-da.values[:-1])

      np.savez( mean_std_file, 
                inputs_mean, inputs_std, inputs_range,
                targets_mean, targets_std, targets_range,
              ) 

   if plotting_histograms:
      mean_std_data = np.load(mean_std_file)
      inputs_mean  = mean_std_data['arr_0']
      inputs_std   = mean_std_data['arr_1']
      inputs_range = mean_std_data['arr_2']
      targets_mean  = mean_std_data['arr_3']
      targets_std   = mean_std_data['arr_4']
      targets_range = mean_std_data['arr_5']
      plot_histograms( da.values.reshape(-1), VarName[var], ShortVarName[var]+'Inputs', inputs_mean[var], inputs_std[var], inputs_range[var] )
      plot_histograms( (da.values[1:]-da.values[:-1]).reshape(-1), VarName[var]+' targets', ShortVarName[var]+'Targets', inputs_mean[var], inputs_std[var], inputs_range[var])
      # Also plot normed data
      plot_histograms( ( (da.values - inputs_mean[var])/inputs_std[var] ).reshape(-1),
                       VarName[var], ShortVarName[var]+'Inputs_NormStd', inputs_mean[var], inputs_std[var], inputs_range[var] )
      plot_histograms( ( (da.values - inputs_mean[var])/inputs_range[var] ).reshape(-1),
                       VarName[var], ShortVarName[var]+'Inputs_NormRange', inputs_mean[var], inputs_std[var], inputs_range[var] )
      plot_histograms( ( ( (da.values[1:]-da.values[:-1]) - targets_mean[var] )/targets_std[var] ).reshape(-1),
                       VarName[var], ShortVarName[var]+'Targets_NormStd', inputs_mean[var], inputs_std[var], inputs_range[var] )
      plot_histograms( ( ( (da.values[1:]-da.values[:-1]) - targets_mean[var] )/targets_range[var] ).reshape(-1),
                       VarName[var], ShortVarName[var]+'Targets_NormRange', inputs_mean[var], inputs_std[var], inputs_range[var] )

   #del da
   #gc.collect()

mean_std_data = np.load(mean_std_file)
inputs_mean  = mean_std_data['arr_0']
inputs_std   = mean_std_data['arr_1']
inputs_range = mean_std_data['arr_2']
targets_mean  = mean_std_data['arr_3']
targets_std   = mean_std_data['arr_4']
targets_range = mean_std_data['arr_5']
print('inputs_mean, inputs_std, inputs_range, targets_mean, targets_std, targets_range')
print(inputs_mean)
print(inputs_std)
print(inputs_range)
print(targets_mean)
print(targets_std)
print(targets_range)
