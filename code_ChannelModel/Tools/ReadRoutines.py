#!/usr/bin/env python
# coding: utf-8

# Code developed by Rachel Furner to contain modules used in developing stats/nn based versions of GCMs.
# Routines contained here include:
# ReadMITGCM - Routine to read in MITGCM data into input and output arrays of single data points (plus halos), split into test and train
#                        portions of code, and normalise. The arrays are passed back on return but not saved - no disc space!
# ReadMITGCMfield - Routine to read in MITGCM data into input and output arrays of whole fields, split into test and train
#                        portions of code, and normalise. The arrays are saved, and also passed back on return. 

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
class MITGCM_Dataset_2d(data.Dataset):

   '''
         MITGCM dataset
        
         Note data not normalised here - needs to be done in network training routine

         Note currently code is set up so that the data is 2-d with different channels for each level of 
         each variable. This solves issues with eta being 2-d and other inputs being 3-d. This matches 
         what is done in Scher paper.
         The alternative is to have 3-d fields, with each channel being separate variables. Potentially 
         worth assessing impact of this at some stage
   '''

   def __init__(self, MITGCM_filename, start_ratio, end_ratio, stride, hist_len, land, tic, bdy_weight, land_values, grid_filename, transform=None):
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

       if self.land == 'ExcLand':
          # Set dims based on V grid
          self.z_dim = (self.ds['VVEL'].isel(T=0).values[:,4:101,:]).shape[0]
          self.y_dim = (self.ds['VVEL'].isel(T=0).values[:,4:101,:]).shape[1] 
          self.x_dim = (self.ds['VVEL'].isel(T=0).values[:,4:101,:]).shape[2]

          # Here mask is ones everywhere
          self.masks = np.ones(( 3*self.z_dim+1, self.y_dim, self.x_dim ))
  
       else:
          # Set dims based on T grid
          self.z_dim = (self.ds['THETA'].isel(T=0).values[:,:,:]).shape[0]
          self.y_dim = (self.ds['THETA'].isel(T=0).values[:,:,:]).shape[1]
          self.x_dim = (self.ds['THETA'].isel(T=0).values[:,:,:]).shape[2]
   
          # Set up masks 
          self.masks = np.ones(( 3*self.z_dim+1, self.y_dim, self.x_dim ))
    
          HfacC = self.mask_ds['HFacC'].values
          for i in range(3):
             self.masks[i*self.z_dim:(i+1)*self.z_dim,:,:] = np.where( HfacC > 0., 1, 0 )
          self.masks[3*self.z_dim,:,:] = np.where( HfacC[0,:,:] > 0., 1, 0 )

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
          da_U_out     = 0.5 * (da_U_out_tmp[:,:,:,:-1]+da_U_out_tmp[:,:,:,1:]) # average to get onto same grid as T points

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

       # Set up sample arrays ready to fill (better for memory than concatenating)
       # Dims are channels,y,x
       sample_input  = np.zeros(( self.hist_len*(1+3*self.z_dim), self.y_dim, self.x_dim ))
       sample_output = np.zeros(( (1+3*self.z_dim), self.y_dim, self.x_dim ))

       for time in range(self.hist_len):
          sample_input[ self.z_dim*3*time+time:self.z_dim*3*time+time+self.z_dim, :, : ]  = \
                                          da_T_in[time,:,:,:].reshape(-1,self.y_dim,self.x_dim)
   
          sample_input[ self.z_dim*3*time+time+self.z_dim:self.z_dim*3*time+time+2*self.z_dim, :, : ]  = \
                                          da_U_in[time,:,:,:].reshape(-1,self.y_dim,self.x_dim) 
   
          sample_input[ self.z_dim*3*time+time+2*self.z_dim:self.z_dim*3*time+time+3*self.z_dim, :, : ]  = \
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
       sample_input = np.where( self.masks==1, sample_input, np.expand_dims( self.land_values, axis=(1,2)) )

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


class MITGCM_Dataset_3d(data.Dataset):

   '''
         MITGCM dataset
        
         Note data not normalised here - needs to be done in network training routine

         Here we have 3-d fields, with each channel being separate variables.
         This is a problem for Eta which is only 2d... for now just filled out to a 3-d field, but this isn't ideal...
   '''

   def __init__(self, MITGCM_filename, start_ratio, end_ratio, stride, hist_len, land, tic, transform=None):
       """
       Args:
          MITGCM_filename (string): Path to the MITGCM filename containing the data.
          start_ratio (float)     : Ratio point (between 0 and 1) at which to start sampling from MITGCM data
          end_ratio (float)       : Ratio point (between 0 and 1) at which to stop sampling from MITGCM data
          stride (integer)        : Rate at which to subsample in time when reading MITGCM data
       """
 
       self.ds        = xr.open_dataset(MITGCM_filename, lock=False)
       self.ds        = self.ds.isel( T=slice( int(start_ratio*self.ds.dims['T']), int(end_ratio*self.ds.dims['T']) ) )
       #self.start   = int(start_ratio * dataset_end_index)
       #self.end     = int(end_ratio * dataset_end_index)
       self.stride    = stride
       self.hist_len  = hist_len
       self.transform = transform
       self.land      = land
       self.tic       = tic 
       
       # For now, manually set masks - eventually set to read in from MITgcm
       if self.land == 'IncLand':
          # Set dims based on T grid
          self.z_dim = (self.ds['THETA'].isel(T=0).values[:,:,:]).shape[0]
          self.y_dim = (self.ds['THETA'].isel(T=0).values[:,:,:]).shape[1]
          self.x_dim = (self.ds['THETA'].isel(T=0).values[:,:,:]).shape[2]

          # Use T mask for U and Eta
          self.T_mask          = np.ones(( self.z_dim, self.y_dim, self.x_dim ))
          self.T_mask[:,:3,:]  = 0.0
          self.T_mask[:,-3:,:] = 0.0
   
          # Later we cut off top row of V data, as this has an extra point and need dimensions to match
          # Note v-mask extends one point above and below t, but cutting off top row deals with top boundary
          self.V_mask          = np.ones(( self.z_dim, self.y_dim, self.x_dim ))
          self.V_mask[:,:4,:]  = 0.0
          self.V_mask[:,-3:,:] = 0.0
   
       elif self.land == 'ExcLand':
          # Set dims based on V grid
          self.z_dim = (self.ds['VVEL'].isel(T=0).values[:,4:101,:]).shape[0]
          self.y_dim = (self.ds['VVEL'].isel(T=0).values[:,4:101,:]).shape[1] 
          self.x_dim = (self.ds['VVEL'].isel(T=0).values[:,4:101,:]).shape[2]

          self.T_mask              = np.ones(( self.z_dim, self.y_dim, self.x_dim ))
          self.V_mask              = np.ones(( self.z_dim, self.y_dim, self.x_dim ))

       elif self.land == 'Spits':
          logging.info("Code Not set up for Land spits in 3d")
          sys.exit()
   
   def __len__(self):
       return int(self.ds.sizes['T']/self.stride)

   def __getitem__(self, idx):

       ds_InputSlice  = self.ds.isel(T=slice(idx*self.stride,idx*self.stride+self.hist_len))
       ds_OutputSlice = self.ds.isel(T=[idx*self.stride+self.hist_len])

       if self.land == 'IncLand':
          # Read in the data
          da_T_in      = ds_InputSlice['THETA'].values[:,:,:,:] 
          da_T_out     = ds_OutputSlice['THETA'].values[:,:,:,:] - da_T_in[-1,:,:,:]
   
          da_U_in_tmp  = ds_InputSlice['UVEL'].values[:,:,:,:]
          da_U_in      = 0.5 * (da_U_in_tmp[:,:,:,:-1]+da_U_in_tmp[:,:,:,1:])   # average x dir onto same grid as T points
          da_U_out_tmp = ds_OutputSlice['UVEL'].values[:,:,:,:]
          da_U_out     = 0.5 * (da_U_out_tmp[:,:,:,:-1]+da_U_out_tmp[:,:,:,1:]) - da_U_in[-1,:,:,:]  # average to get onto same grid as T points
   
          da_V_in      = ds_InputSlice['VVEL'].values[:,:,:-1,:]   # ignore last land point to get to same as T grid
          da_V_out     = ds_OutputSlice['VVEL'].values[:,:,:-1,:] -da_V_in[-1,:,:,:]  # ignore last land point to get to same as T grid 
   
          da_Eta_in    = ds_InputSlice['ETAN'].values[:,:,:,:]
          da_Eta_out   = ds_OutputSlice['ETAN'].values[:,:,:,:] - da_Eta_in[-1,:,:,:]
       
       elif self.land == 'ExcLand':
          # Just cut out the ocean parts of the grid, and leave mask as all zeros

          # Read in the data
          da_T_in_tmp  = ds_InputSlice['THETA'].values[:,:,3:101,:] 
          da_T_in      = 0.5 * (da_T_in_tmp[:,:,:-1,:]+da_T_in_tmp[:,:,1:,:]) # average to get onto same grid as V points  
          da_T_out_tmp = ds_OutputSlice['THETA'].values[:,:,3:101,:]
          da_T_out     = 0.5 * (da_T_out_tmp[:,:,:-1,:]+da_T_out_tmp[:,:,1:,:]) - da_T_in[-1,:,:,:]  # average to get onto same grid as V points  

          da_U_in_tmp  = ds_InputSlice['UVEL'].values[:,:,3:101,:]
          da_U_in_tmp  = 0.5 * (da_U_in_tmp[:,:,:,:-1]+da_U_in_tmp[:,:,:,1:]) # average x dir onto same grid as T points  
          da_U_in      = 0.5 * (da_U_in_tmp[:,:,:-1,:]+da_U_in_tmp[:,:,1:,:]) # average y dir onto same grid as V points  
          da_U_out_tmp = ds_OutputSlice['UVEL'].values[:,:,3:101,:]
          da_U_out_tmp = 0.5 * (da_U_out_tmp[:,:,:,:-1]+da_U_out_tmp[:,:,:,1:])  # average to get onto same grid as T points
          da_U_out     = 0.5 * (da_U_out_tmp[:,:,:-1,:]+da_U_out_tmp[:,:,1:,:]) - da_U_in[-1,:,:,:]  # average y dir onto same grid as V points  

          da_V_in      = ds_InputSlice['VVEL'].values[:,:,4:101,:]  # Ignore first point as zero
          da_V_out     = ds_OutputSlice['VVEL'].values[:,:,3:101,:] - da_V_in[-1,:,:,:]   # Ignore first point as zero

          da_Eta_in_tmp = ds_InputSlice['ETAN'].values[:,:,3:101,:]
          da_Eta_in     = 0.5 * (da_Eta_in_tmp[:,:,:-1,:]+da_Eta_in_tmp[:,:,1:,:]) # average to get onto same grid as V points  
          da_Eta_out_tmp= ds_OutputSlice['ETAN'].values[:,:,3:101,:]
          da_Eta_out    = 0.5 * (da_Eta_out_tmp[:,:,:-1,:]+da_Eta_out_tmp[:,:,1:,:]) - da_Eta_in[-1,:,:,:] # average to get onto same grid as V points  

       del ds_InputSlice
       del ds_OutputSlice
       gc.collect()
       torch.cuda.empty_cache()

       ## Shouldn't need to mask the data, as land values set to zero in MITgcm output already
       #da_T_in  = np.where(self.T_mask==0, 0, da_T_in)
       #da_T_out = np.where(self.T_mask==0, 0, da_T_out)
      
       #da_U_in  = np.where(self.T_mask==0, 0, da_U_in)
       #da_U_out = np.where(self.T_mask==0, 0, da_U_out)
      
       #da_V_in  = np.where(self.V_mask==0, 0, da_V_in)
       #da_V_out = np.where(self.V_mask==0, 0, da_V_out)
      
       #da_Eta_in  = np.where(self.T_mask==0, 0, da_Eta_in)
       #da_Eta_out = np.where(self.T_mask==0, 0, da_Eta_out)


       # Fill input and output arrays
       sample_input  = np.zeros(( 4*self.hist_len+2, self.z_dim, self.y_dim, self.x_dim ))  # shape: (no channels, z, y, x)
       sample_output = np.zeros(( 4, self.z_dim, self.y_dim, self.x_dim ))  # shape: (no channels, z, y, x)

       for time in range(self.hist_len):
          sample_input[time*4,:,:,:]   = da_T_in[time, :, :, :]
          sample_input[time*4+1,:,:,:] = da_U_in[time, :, :, :]
          sample_input[time*4+2,:,:,:] = da_V_in[time, :, :, :]
          sample_input[time*4+3,:,:,:] = np.broadcast_to(da_Eta_in[time, :, :, :], (1, self.z_dim, self.y_dim, self.x_dim))

       sample_output[0,:,:,:] = da_T_out[:, :, :, :]
       sample_output[1,:,:,:] = da_U_out[:, :, :, :]
       sample_output[2,:,:,:] = da_V_out[:, :, :, :]
       sample_output[3,:,:,:] = np.broadcast_to(da_Eta_out[0, :, :, :], (self.z_dim, self.y_dim, self.x_dim))

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

       # Cat masks onto inputs but not outputs
       sample_input[4*self.hist_len,:,:,:]  = self.T_mask[:, :, :]
       sample_input[4*self.hist_len+1,:,:,:]  = self.V_mask[:, :, :]

       sample_input = torch.from_numpy(sample_input)
       sample_output = torch.from_numpy(sample_output)
 
       #sample = {'input':sample_input, 'output':sample_output}
       
       if self.transform:
          sample = self.transform(sample)
 
       return sample_input, sample_output


class RF_Normalise_sample(object):
    """Normalise data based on inputsing means and std (given)

    Args:
       TBD
    """
    def __init__(self, inputs_mean, inputs_std, inputs_range, targets_mean, targets_std, targets_range, histlen, no_in_channels, no_out_channels, dim):
        if dim == '2d':
           self.inputs_mean  = inputs_mean[:, np.newaxis, np.newaxis]
           self.inputs_std   = inputs_std[:, np.newaxis, np.newaxis]
           self.inputs_range = inputs_range[:, np.newaxis, np.newaxis]
           self.targets_mean  = targets_mean[:, np.newaxis, np.newaxis]
           self.targets_std   = targets_std[:, np.newaxis, np.newaxis]
           self.targets_range = targets_range[:, np.newaxis, np.newaxis]
        elif dim == '3d':
           self.inputs_mean  = inputs_mean[:, np.newaxis, np.newaxis, np.newaxis]
           self.inputs_std   = inputs_std[:, np.newaxis, np.newaxis, np.newaxis]
           self.inputs_range = inputs_range[:, np.newaxis, np.newaxis, np.newaxis]
           self.targets_mean  = targets_mean[:, np.newaxis, np.newaxis, np.newaxis]
           self.targets_std   = targets_std[:, np.newaxis, np.newaxis, np.newaxis]
           self.targets_range = targets_range[:, np.newaxis, np.newaxis, np.newaxis]
        self.histlen     = histlen
        self.no_in_channels = no_in_channels
        self.no_out_channels = no_out_channels
        self.dim         = dim

    def __call__(self, sample):

        # Using transforms.Normalize returns the function, rather than the normalised array - can't figure out how to avoid this...
        #sample_input  = transforms.Normalize(sample_input, self.inputs_inputs_mean, self.inputs_inputs_std)
        #sample_output = transforms.Normalize(sample_output,self.outputs_inputs_mean, self.outputs_inputs_std)

        # only normalise for channels with physical variables (= no_out channels*histlen)..
        # the input channels also have mask channels, but we don't want to normalise these. 

        if self.dim == '2d':
           
           for time in range(self.histlen):
              sample['input'][time*self.no_out_channels:(time+1)*self.no_out_channels, :, :] = \
                                       ( sample['input'][time*self.no_out_channels:(time+1)*self.no_out_channels, :, :] -  
                                         self.inputs_mean[:self.no_out_channels,:,:] ) / self.inputs_range[:self.no_out_channels,:,:]
           sample['output'][:self.no_out_channels, :, :] = ( sample['output'][:self.no_out_channels, :, :] -
                                                             self.targets_mean[:self.no_out_channels,:,:] ) /  \
                                                             self.targets_range[:self.no_out_channels,:,:]

        elif self.dim == '3d':

           for time in range(self.histlen):
               sample['input'][time*self.no_out_channels:(time+1)*self.no_out_channels, :, :, :] = \
                          ( sample['input'][time*self.no_out_channels:(time+1)*self.no_out_channels, :, :, :]
                            - self.inputs_mean[:self.no_out_channels,:,:,:]) / self.inputs_range[self.no_out_channels,:,:]
           sample['output'][:, :, :, :] = ( sample['output'][:, :, :, :] - self.targets_mean[:self.no_out_channels,:,:,:]) \
                                            / self.targets_range[self.no_out_channels,:,:,:]

        return sample['input'], sample['output']

def RF_ReNormalise_predicted(prediction, data_mean, data_std, data_range, no_out_channels, dim):
    """Normalise data based on training means and std (given)

       ReNormalises prediction for use in iteration

    Args:
       TBD
    """

    if dim == '2d':
       prediction[:, :no_out_channels, :, :] = ( prediction[:, :no_out_channels, :, :] - 
                                                 data_mean[np.newaxis, :no_out_channels, np.newaxis, np.newaxis] ) /  \
                                                 data_range[np.newaxis, :no_out_channels, np.newaxis, np.newaxis]
    elif dim == '3d':
       prediction[:, :no_out_channels, :, :, :] = ( prediction[:, :no_out_channels, :, :, :] - 
                                                    data_mean[np.newaxis, :no_out_channels, np.newaxis, np.newaxis, np.newaxis]) /  \
                                                    data_range[np.newaxis, :no_out_channels, np.newaxis, np.newaxis, np.newaxis]

    return prediction

def RF_DeNormalise(samples, data_mean, data_std, data_range, no_out_channels, dim):
    """de-Normalise data based on training means and std (given)

    Args:
       Array to de-norm (expected shape - no_samples, no_channels, y, x)
       Training output mean
       Training output std
    """

    if dim == '2d':
       samples[:, :no_out_channels, :, :]  = ( samples[:, :no_out_channels, :, :] * 
                                               data_range[np.newaxis, :no_out_channels, np.newaxis, np.newaxis] ) + \
                                               data_mean[np.newaxis, :no_out_channels, np.newaxis, np.newaxis]
    elif dim == '3d':
       samples[:, :no_out_channels, :, :, :]  = ( samples[:, :no_out_channels, :, :, :] * 
                                                  data_range[np.newaxis, :no_out_channels, np.newaxis, np.newaxis, np.newaxis] ) + \
                                                  data_mean[np.newaxis, :no_out_channels, np.newaxis, np.newaxis, np.newaxis]

    return (samples)

