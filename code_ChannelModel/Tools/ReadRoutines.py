#!/usr/bin/env python
# coding: utf-8

import numpy as np
import xarray as xr
from torch.utils import data
from torchvision import transforms, utils
import torch
import numpy.lib.stride_tricks as npst

import multiprocessing as mp
#import torch.multiprocessing
#torch.multiprocessing.set_sharing_strategy('file_system')

import gc
import time as time
import sys
sys.path.append('../NNregression')
#from WholeGridNetworkRegressorModules import TimeCheck 

# Create Dataset, which inherits the data.Dataset class
class MITGCM_Dataset(data.Dataset):

   '''
         MITGCM dataset

         This code is set up in '2d' mode so that the data is 2-d with different channels for each level of 
         each variable. This solves issues with eta being 2-d and other inputs being 3-d. This matches 
         what is done in Scher paper.
         Or in '3d' mode, so each variable is a channel, and each channel has 3d data. Here there are issues with
         Eta only having one depth level - we get around this by providing 38 identical fields... far from perfect!
   '''

   def __init__(self, MITGCM_filename, start_ratio, end_ratio, stride, hist_len, land, tic, bdy_weight, land_values, grid_filename, dim, transform=None):
       """
       Args:
          MITGCM_filename (string): Path to the MITGCM filename containing the data.
          start_ratio (float)     : Ratio point (between 0 and 1) at which to start sampling from MITGCM data
          end_ratio (float)       : Ratio point (between 0 and 1) at which to stop sampling from MITGCM data
          stride (integer)        : Rate at which to subsample in time when reading MITGCM data
       """
 
       self.ds        = xr.open_dataset(MITGCM_filename, lock=False)
       self.ds        = self.ds.isel( T=slice( int(start_ratio*self.ds.dims['T']), int(end_ratio*self.ds.dims['T']) ) )
       self.stride    = stride
       self.hist_len  = hist_len
       self.transform = transform
       self.land      = land
       self.tic       = tic 
       no_depth_levels = 38
       self.land_values = land_values
       self.mask_ds   = xr.open_dataset(grid_filename, lock=False)
       self.dim       = dim

       if self.land == 'ExcLand':
          # Set dims based on V grid
          self.z_dim = (self.ds['VVEL'].isel(T=0).values[:,4:101,:]).shape[0]
          self.y_dim = (self.ds['VVEL'].isel(T=0).values[:,4:101,:]).shape[1] 
          self.x_dim = (self.ds['VVEL'].isel(T=0).values[:,4:101,:]).shape[2]

       else:
          # Set dims based on T grid
          self.z_dim = (self.ds['THETA'].isel(T=0).values[:,:,:]).shape[0]
          self.y_dim = (self.ds['THETA'].isel(T=0).values[:,:,:]).shape[1]
          self.x_dim = (self.ds['THETA'].isel(T=0).values[:,:,:]).shape[2]
   
       # set mask, initialise as ocean (ones) everywhere
       if self.dim == '2d':
          self.masks = np.ones(( 3*self.z_dim+1, self.y_dim, self.x_dim ))
       elif self.dim == '3d':
          self.masks = np.ones(( 4, self.z_dim, self.y_dim, self.x_dim ))
    
       HfacC = self.mask_ds['HFacC'].values
       if self.dim == '2d':
          for i in range(3):
             self.masks[i*self.z_dim:(i+1)*self.z_dim,:,:] = np.where( HfacC > 0., 1, 0 )
          self.masks[3*self.z_dim,:,:] = np.where( HfacC[0,:,:] > 0., 1, 0 )
       elif self.dim == '3d':
          for i in range(4):
             self.masks[i,:,:,:] = np.where( HfacC > 0., 1, 0 )
       self.masks = torch.from_numpy(self.masks)

   def __len__(self):
       return int(self.ds.sizes['T']/self.stride)

   def __getitem__(self, idx):

       ds_InputSlice  = self.ds.isel(T=slice(idx*self.stride,idx*self.stride+self.hist_len))
       ds_OutputSlice = self.ds.isel(T=[idx*self.stride+self.hist_len])

       if self.land == 'IncLand' or self.land == 'Spits':
          # Read in the data

          da_T_in      = ds_InputSlice['THETA'].values[:,:,:,:] 
          da_T_out     = ds_OutputSlice['THETA'].values[:,:,:,:] - da_T_in[-1,:,:,:]
 
          da_U_in_tmp  = ds_InputSlice['UVEL'].values[:,:,:,:]
          da_U_in      = 0.5 * (da_U_in_tmp[:,:,:,:-1]+da_U_in_tmp[:,:,:,1:])   # average x dir onto same grid as T points
          da_U_out_tmp = ds_OutputSlice['UVEL'].values[:,:,:,:]
          da_U_out     = 0.5 * (da_U_out_tmp[:,:,:,:-1]+da_U_out_tmp[:,:,:,1:]) - da_U_in[-1,:,:,:] # average to get onto same grid as T points

          da_V_in  = ds_InputSlice['VVEL'].values[:,:,:-1,:]                           #ignore last land point to get to same as T grid
          da_V_out = ds_OutputSlice['VVEL'].values[:,:,:-1,:] - da_V_in[-1,:,:,:]      #ignore last land point to get to same as T grid 
   
          da_Eta_in  = ds_InputSlice['ETAN'].values[:,0,:,:]
          da_Eta_out = ds_OutputSlice['ETAN'].values[:,0,:,:] - da_Eta_in[-1,:,:]
       
       elif self.land == 'ExcLand':
       #   # Just cut out the ocean parts of the grid
   
          # Read in the data
          da_T_in_tmp  = ds_InputSlice['THETA'].values[:,:,3:101,:] 
          da_T_in      = 0.5 * (da_T_in_tmp[:,:,:-1,:]+da_T_in_tmp[:,:,1:,:])               # average to get onto same grid as V points  
          da_T_out_tmp = ds_OutputSlice['THETA'].values[:,:,3:101,:]
          da_T_out     = 0.5 * (da_T_out_tmp[:,:,:-1,:]+da_T_out_tmp[:,:,1:,:]) - da_T_in[-1,:,:,:]   # average to get onto same grid as V points  

          da_U_in_tmp  = ds_InputSlice['UVEL'].values[:,:,3:101,:]
          da_U_in_tmp  = 0.5 * (da_U_in_tmp[:,:,:,:-1]+da_U_in_tmp[:,:,:,1:]) # average x dir onto same grid as T points  
          da_U_in      = 0.5 * (da_U_in_tmp[:,:,:-1,:]+da_U_in_tmp[:,:,1:,:]) # average y dir onto same grid as V points  
          da_U_out_tmp = ds_OutputSlice['UVEL'].values[:,:,3:101,:]
          da_U_out_tmp = 0.5 * (da_U_out_tmp[:,:,:,:-1]+da_U_out_tmp[:,:,:,1:])             # average to get onto same grid as T points
          da_U_out     = 0.5 * (da_U_out_tmp[:,:,:-1,:]+da_U_out_tmp[:,:,1:,:]) - da_U_in[-1,:,:,:]   # average y dir onto same grid as V points  

          da_V_in      = ds_InputSlice['VVEL'].values[:,:,4:101,:]              # Extra land point at South compared to other variables
          da_V_out     = ds_OutputSlice['VVEL'].values[:,:,4:101,:] - da_V_in[-1,:,:,:]   # Extra land point at South compared to other variables 

          da_Eta_in_tmp = ds_InputSlice['ETAN'].values[:,0,3:101,:]
          da_Eta_in     = 0.5 * (da_Eta_in_tmp[:,:-1,:]+da_Eta_in_tmp[:,1:,:])                # average to get onto same grid as V points  
          da_Eta_out_tmp= ds_OutputSlice['ETAN'].values[:,0,3:101,:]
          da_Eta_out    = 0.5 * (da_Eta_out_tmp[:,:-1,:]+da_Eta_out_tmp[:,1:,:]) - da_Eta_in[-1,:,:]  # average to get onto same grid as V points  

       del ds_InputSlice
       del ds_OutputSlice
       gc.collect()
       torch.cuda.empty_cache()

       #TimeCheck(self.tic, 'Read in data')

       # Set up sample arrays and then fill (better for memory than concatenating)
       if self.dim == '2d':
          # Dims are channels,y,x
          sample_input  = np.zeros(( self.hist_len*(1+3*self.z_dim), self.y_dim, self.x_dim ))
          sample_output = np.zeros(( (1+3*self.z_dim), self.y_dim, self.x_dim ))
   
          for time in range(self.hist_len):
             sample_input[ self.z_dim*3*time+time : self.z_dim*3*time+time+self.z_dim, :, : ]  = \
                                             da_T_in[time,:,:,:].reshape(-1,self.y_dim,self.x_dim)
      
             sample_input[ self.z_dim*3*time+time+self.z_dim : self.z_dim*3*time+time+2*self.z_dim, :, : ]  = \
                                             da_U_in[time,:,:,:].reshape(-1,self.y_dim,self.x_dim) 
      
             sample_input[ self.z_dim*3*time+time+2*self.z_dim : self.z_dim*3*time+time+3*self.z_dim, :, : ]  = \
                                             da_V_in[time,:,:,:].reshape(-1,self.y_dim,self.x_dim)
      
             sample_input[ self.z_dim*3*time+time+3*self.z_dim, :, : ] = \
                                             da_Eta_in[time,:,:].reshape(-1,self.y_dim,self.x_dim)              
      
          sample_output[ :1*self.z_dim, :, : ] = \
                                          da_T_out[:,:,:,:].reshape(-1,self.y_dim,self.x_dim)
          sample_output[ 1*self.z_dim:2*self.z_dim, :, : ] = \
                                          da_U_out[:,:,:,:].reshape(-1,self.y_dim,self.x_dim)
          sample_output[ 2*self.z_dim:3*self.z_dim, :, : ] = \
                                          da_V_out[:,:,:,:].reshape(-1,self.y_dim,self.x_dim)
          sample_output[ 3*self.z_dim, :, : ] = \
                                          da_Eta_out[:,:,:].reshape(-1,self.y_dim,self.x_dim)              
   
          # mask values
          for time in range(self.hist_len):
              sample_input[time*(1+3*self.z_dim):(time+1)*(1+3*self.z_dim),:,:] = np.where( self.masks==1, 
                                                                                            sample_input[time*(1+3*self.z_dim):(time+1)*(1+3*self.z_dim),:,:],
                                                                                            np.expand_dims( self.land_values, axis=(1,2)) )

       elif self.dim == '3d':
          # Dims are channels,z,y,x
          sample_input  = np.zeros(( 4 * self.hist_len, self.z_dim, self.y_dim, self.x_dim ))  # shape: (no channels, z, y, x)
          sample_output = np.zeros(( 4, self.z_dim, self.y_dim, self.x_dim ))  # shape: (no channels, z, y, x)
   
          for time in range(self.hist_len):
             sample_input[time*4,:,:,:]   = da_T_in[time, :, :, :]
             sample_input[time*4+1,:,:,:] = da_U_in[time, :, :, :]
             sample_input[time*4+2,:,:,:] = da_V_in[time, :, :, :]
             sample_input[time*4+3,:,:,:] = np.broadcast_to(da_Eta_in[time, :, :], (1, self.z_dim, self.y_dim, self.x_dim))
   
          sample_output[0,:,:,:] = da_T_out[:, :, :, :]
          sample_output[1,:,:,:] = da_U_out[:, :, :, :]
          sample_output[2,:,:,:] = da_V_out[:, :, :, :]
          sample_output[3,:,:,:] = np.broadcast_to(da_Eta_out[:, :, :], (1, self.z_dim, self.y_dim, self.x_dim))
   
          # mask values
          for time in range(self.hist_len):
              sample_input[time*4:(time+1)*4,:,:,:] = np.where( self.masks==1, sample_input[time*4:(time+1)*4,:,:,:], np.expand_dims( self.land_values, axis=(1,2,3)) )

       del da_T_in
       del da_T_out
       del da_U_in
       del da_U_out
       del da_V_in
       del da_V_out
       del da_Eta_in
       del da_Eta_out
       gc.collect()
       torch.cuda.empty_cache()

       sample_input = torch.from_numpy(sample_input)
       sample_output = torch.from_numpy(sample_output)

       if self.transform:
          sample_input, sample_output = self.transform({'input':sample_input, 'output':sample_output})
 
       return sample_input, sample_output, self.masks


class RF_Normalise_sample(object):

    def __init__(self, inputs_mean, inputs_std, inputs_range, targets_mean, targets_std, targets_range, histlen, no_out_channels, dim):
        self.inputs_mean   = inputs_mean
        self.inputs_std    = inputs_std
        self.inputs_range  = inputs_range
        self.targets_mean  = targets_mean
        self.targets_std   = targets_std
        self.targets_range = targets_range
        self.histlen       = histlen
        self.no_out_channels = no_out_channels
        self.dim           = dim

    def __call__(self, sample):

        # Using transforms.Normalize returns the function, rather than the normalised array - can't figure out how to avoid this...
        #sample_input  = transforms.Normalize(sample_input, self.inputs_inputs_mean, self.inputs_inputs_std)
        #sample_output = transforms.Normalize(sample_output,self.outputs_inputs_mean, self.outputs_inputs_std)

        # only normalise for channels with physical variables (= no_out channels*histlen)..
        # the input channels also have mask channels, but we don't want to normalise these. 

        if self.dim == '2d':
           for time in range(self.histlen):
              sample['input'][time*self.no_out_channels:(time+1)*self.no_out_channels, :, :] =                   \
                                   RF_Normalise( sample['input'][time*self.no_out_channels:(time+1)*self.no_out_channels, :, :],
                                                             self.inputs_mean, self.inputs_std, self.inputs_range, self.dim )
        elif self.dim == '3d':
           for time in range(self.histlen):
              sample['input'][time*self.no_out_channels:(time+1)*self.no_out_channels, :, :, :] =                   \
                                   RF_Normalise( sample['input'][time*self.no_out_channels:(time+1)*self.no_out_channels, :, :, :],
                                                             self.inputs_mean, self.inputs_std, self.inputs_range,self.dim )

        sample['output'] = RF_Normalise( sample['output'], self.targets_mean, self.targets_mean, self.targets_range, self.dim)

        return sample['input'], sample['output']

def RF_Normalise(data, d_mean, d_std, d_range, dim):
    """Normalise data based on training means and std (given)

       ReNormalises prediction for use in iteration
       dims of data: channels(, z), y, x

    Args:
       TBD
    """

    if dim == '2d':
       data[:,:,:] = ( data[:,:,:] - d_mean[:, np.newaxis, np.newaxis] )                      \
                                    / d_range[:, np.newaxis, np.newaxis]
    elif dim == '3d':
       data[:,:,:,:] = ( data[:,:,:,:] - d_mean[:, np.newaxis, np.newaxis, np.newaxis] )    \
                                    / d_range[:, np.newaxis, np.newaxis, np.newaxis]

    return data

def RF_DeNormalise(samples, data_mean, data_std, data_range, dim):
    """de-Normalise data based on training means and std (given)

    Args:
       Array to de-norm (expected shape - no_samples, no_channels,( z,) y, x)
       Training output mean
       Training output std
    """
    if dim == '2d':
       samples[:,:,:,:]  = ( samples[:,:,:,:] * 
                                               data_range[np.newaxis, :, np.newaxis, np.newaxis] ) + \
                                               data_mean[np.newaxis, :, np.newaxis, np.newaxis]
    elif dim == '3d':
       samples[:,:,:,:,:]  = ( samples[:,:,:,:,:] * 
                                                  data_range[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis] ) + \
                                                  data_mean[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis]

    return (samples)

