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

import multiprocessing as mp
#import torch.multiprocessing
#torch.multiprocessing.set_sharing_strategy('file_system')

import time as time
import sys
sys.path.append('../NNregression')
from WholeGridNetworkRegressorModules import TimeCheck 

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

   def __init__(self, MITGCM_filename, start_ratio, end_ratio, stride, dataset_end_index, hist_len, land, tic, transform=None):
       """
       Args:
          MITGCM_filename (string): Path to the MITGCM filename containing the data.
          start_ratio (float)     : Ratio point (between 0 and 1) at which to start sampling from MITGCM data
          end_ratio (float)       : Ratio point (between 0 and 1) at which to stop sampling from MITGCM data
          stride (integer)        : Rate at which to subsample in time when reading MITGCM data
          dataset_end_index       : Point of dataset file after which to ignore (i.e. cause model is no longer
                                    dynamically active
       """
 
       self.ds      = xr.open_dataset(MITGCM_filename, lock=False)
       self.start   = int(start_ratio * dataset_end_index)
       self.end     = int(end_ratio * dataset_end_index)
       self.stride  = stride
       self.hist_len= hist_len
       self.transform = transform
       self.land = land
       self.tic  = tic 
       
       #self.ds_inputs  = self.ds.isel(T=slice(self.start+3, self.end+3, self.stride))
       #self.ds_outputs = self.ds.isel(T=slice(self.start+4, self.end+4, self.stride))
  
       land=97  # T grid point where land starts

       # manually set masks for now - later look to output from MITgcm and read in
       if self.land == 'IncLand':
          # Set dims based on T grid
          self.z_dim = (self.ds['THETA'].isel(T=0).values[:,:,:]).shape[0]
          self.y_dim = (self.ds['THETA'].isel(T=0).values[:,:,:]).shape[1] - 2   # Eventually re-run MITgcm with better grid size, for now ignore last rows of land!
          self.x_dim = (self.ds['THETA'].isel(T=0).values[:,:,:]).shape[2]
   
          self.T_mask            = np.ones(( self.z_dim, self.y_dim, self.x_dim ))
          self.T_mask[:,land:,:] = 0.0

          self.U_mask            = np.ones(( self.z_dim, self.y_dim, self.x_dim ))
          self.U_mask[:,land:,:] = 0.0
   
          self.V_mask            = np.ones(( self.z_dim, self.y_dim, self.x_dim ))
          self.V_mask[:,0,:]     = 0.0  # Set bottom row to land
          self.V_mask[:,land:,:] = 0.0
   
          self.Eta_mask          = np.ones(( self.y_dim, self.x_dim ))
          self.Eta_mask[land:,:] = 0.0

       elif self.land == 'ExcLand':
          # Set dims based on V grid
          self.z_dim = (self.ds['VVEL'].isel(T=0).values[:,1:land,:]).shape[0]
          self.y_dim = (self.ds['VVEL'].isel(T=0).values[:,1:land,:]).shape[1] 
          self.x_dim = (self.ds['VVEL'].isel(T=0).values[:,1:land,:]).shape[2]

          # manually set masks for now - later look to output from MITgcm and read in. 
          # Here mask is ones everywhere
          self.T_mask              = np.ones(( self.z_dim, self.y_dim, self.x_dim ))
          self.U_mask              = np.ones(( self.z_dim, self.y_dim, self.x_dim ))
          self.V_mask              = np.ones(( self.z_dim, self.y_dim, self.x_dim ))
          self.Eta_mask            = np.ones(( self.y_dim, self.x_dim ))
   
   def __len__(self):
       return int(self.ds.sizes['T']/self.stride)

   def __getitem__(self, idx):

       land=97  # T grid point where land starts

       ds_InputSlice  = self.ds.isel(T=slice(idx*self.stride,idx*self.stride+self.hist_len))
       ds_OutputSlice = self.ds.isel(T=[idx*self.stride+self.hist_len])

       if self.land == 'IncLand':
          # Read in the data

          da_T_in      = ds_InputSlice['THETA'].values[:,:,:-2,:] 
          da_T_out     = ds_OutputSlice['THETA'].values[:,:,:-2,:]
   
          da_U_in_tmp  = ds_InputSlice['UVEL'].values[:,:,:-2,:]
          da_U_in      = 0.5 * (da_U_in_tmp[:,:,:,:-1]+da_U_in_tmp[:,:,:,1:]) # average x dir onto same grid as T points
          da_U_out_tmp = ds_OutputSlice['UVEL'].values[:,:,:-2,:]
          da_U_out     = 0.5 * (da_U_out_tmp[:,:,:,:-1]+da_U_out_tmp[:,:,:,1:]) # average to get onto same grid as T points
   
          da_V_in      = ds_InputSlice['VVEL'].values[:,:,:-3,:] #ignore last 2 land points to get to same as T grid
          da_V_out     = ds_OutputSlice['VVEL'].values[:,:,:-3,:] #ignore last 2 land points to get to same as T grid 
   
          da_Eta_in    = ds_InputSlice['ETAN'].values[:,0,:-2,:]
          da_Eta_out   = ds_OutputSlice['ETAN'].values[:,0,:-2,:]
       
       elif self.land == 'ExcLand':
          # Just cut out the ocean parts of the grid, and leave mask as all ones
   
          # Read in the data
          da_T_in_tmp  = ds_InputSlice['THETA'].values[:,:,:land,:] 
          da_T_in      = 0.5 * (da_T_in_tmp[:,:,:-1,:]+da_T_in_tmp[:,:,1:,:]) # average to get onto same grid as V points  
          da_T_out_tmp = ds_OutputSlice['THETA'].values[:,:,:land,:]
          da_T_out     = 0.5 * (da_T_out_tmp[:,:,:-1,:]+da_T_out_tmp[:,:,1:,:]) # average to get onto same grid as V points  

          da_U_in_tmp  = ds_InputSlice['UVEL'].values[:,:,:land,:]
          da_U_in_tmp  = 0.5 * (da_U_in_tmp[:,:,:,:-1]+da_U_in_tmp[:,:,:,1:]) # average x dir onto same grid as T points  
          da_U_in      = 0.5 * (da_U_in_tmp[:,:,:-1,:]+da_U_in_tmp[:,:,1:,:]) # average y dir onto same grid as V points  
          da_U_out_tmp = ds_OutputSlice['UVEL'].values[:,:,:land,:]
          da_U_out_tmp = 0.5 * (da_U_out_tmp[:,:,:,:-1]+da_U_out_tmp[:,:,:,1:])  # average to get onto same grid as T points
          da_U_out     = 0.5 * (da_U_out_tmp[:,:,:-1,:]+da_U_out_tmp[:,:,1:,:])  # average y dir onto same grid as V points  

          da_V_in      = ds_InputSlice['VVEL'].values[:,:,1:land,:]  # Ignore first point as zero
          da_V_out     = ds_OutputSlice['VVEL'].values[:,:,1:land,:]    # Ignore first point as zero

          da_Eta_in_tmp = ds_InputSlice['ETAN'].values[:,0,:land,:]
          da_Eta_in     = 0.5 * (da_Eta_in_tmp[:,:-1,:]+da_Eta_in_tmp[:,1:,:]) # average to get onto same grid as V points  
          da_Eta_out_tmp= ds_OutputSlice['ETAN'].values[:,0,:land,:]
          da_Eta_out    = 0.5 * (da_Eta_out_tmp[:,:-1,:]+da_Eta_out_tmp[:,1:,:]) # average to get onto same grid as V points  

       #TimeCheck(self.tic, 'Read in data')

       # Mask the data
       da_T_in  = np.where(self.T_mask==0, 0, da_T_in)
       da_T_out = np.where(self.T_mask==0, 0, da_T_out)
      
       da_U_in  = np.where(self.U_mask==0, 0, da_U_in)
       da_U_out = np.where(self.U_mask==0, 0, da_U_out)
      
       da_V_in  = np.where(self.V_mask==0, 0, da_V_in)
       da_V_out = np.where(self.V_mask==0, 0, da_V_out)
      
       da_Eta_in  = np.where(self.Eta_mask==0, 0, da_Eta_in)
       da_Eta_out = np.where(self.Eta_mask==0, 0, da_Eta_out)

       #TimeCheck(self.tic, 'masked data')

       # Set up sample arrays ready to fill (better for memory than concatenating)
       # Dims are channels,y,x
       sample_input  = np.zeros(( self.hist_len*(1+3*self.z_dim)+(1+3*self.z_dim), self.y_dim, self.x_dim ))
       sample_output = np.zeros(( (1+3*self.z_dim), self.y_dim, self.x_dim ))

       sample_input[ :self.hist_len*1*self.z_dim, :, : ]  = \
                                       da_T_in[:,:,:,:].reshape(-1,self.y_dim,self.x_dim)
       sample_output[ :1*self.z_dim, :, : ] = \
                                       da_T_out[:,:,:,:].reshape(-1,self.y_dim,self.x_dim)

       sample_input[ self.hist_len*1*self.z_dim:self.hist_len*2*self.z_dim, :, : ]  = \
                                       da_U_in[:,:,:,:].reshape(-1,self.y_dim,self.x_dim) 
       sample_output[ 1*self.z_dim:2*self.z_dim, :, : ] = \
                                       da_U_out[:,:,:,:].reshape(-1,self.y_dim,self.x_dim)

       sample_input[ self.hist_len*2*self.z_dim:self.hist_len*3*self.z_dim, :, : ]  = \
                                       da_V_in[:,:,:,:].reshape(-1,self.y_dim,self.x_dim)
       sample_output[ 2*self.z_dim:3*self.z_dim, :, : ] = \
                                       da_V_out[:,:,:,:].reshape(-1,self.y_dim,self.x_dim)

       sample_input[ self.hist_len*3*self.z_dim:self.hist_len*3*self.z_dim+self.hist_len, :, : ] = \
                                       da_Eta_in[:,:,:].reshape(-1,self.y_dim,self.x_dim)              
       sample_output[ 3*self.z_dim, :, : ] = \
                                       da_Eta_out[:,:,:].reshape(-1,self.y_dim,self.x_dim)              

       #TimeCheck(self.tic, 'Catted data')

       # Add masks onto inputs but not outputs
       sample_input[ self.hist_len*3*self.z_dim+self.hist_len:self.hist_len*3*self.z_dim+self.hist_len+self.z_dim, :, :] = \
                                             self.T_mask[:, :, :]
       sample_input[ self.hist_len*3*self.z_dim+self.hist_len+self.z_dim:self.hist_len*3*self.z_dim+self.hist_len+2*self.z_dim, :, :] = \
                                             self.U_mask[:, :, :]
       sample_input[ self.hist_len*3*self.z_dim+self.hist_len+2*self.z_dim:self.hist_len*3*self.z_dim+self.hist_len+3*self.z_dim, :, :]  = \
                                             self.V_mask[:, :, :]
       sample_input[ self.hist_len*3*self.z_dim+self.hist_len+3*self.z_dim, :, :]  = self.Eta_mask[:, :] 

       #TimeCheck(self.tic, 'Catted masks on')
 
       sample_input = torch.from_numpy(sample_input)
       sample_output = torch.from_numpy(sample_output)
 
       sample = {'input':sample_input, 'output':sample_output}
       #TimeCheck(self.tic, 'converted to torch tensor and stored as sample')

       if self.transform:
          sample = self.transform(sample)
          #TimeCheck(self.tic, 'Normalised data')
 
       return sample


class MITGCM_Dataset_3d(data.Dataset):

   '''
         MITGCM dataset
        
         Note data not normalised here - needs to be done in network training routine

         Here we have 3-d fields, with each channel being separate variables.
         This is a problem for Eta which is only 2d... for now just filled out to a 3-d field, but this isn't ideal...
   '''

   def __init__(self, MITGCM_filename, start_ratio, end_ratio, stride, dataset_end_index, hist_len, land, tic, transform=None):
       """
       Args:
          MITGCM_filename (string): Path to the MITGCM filename containing the data.
          start_ratio (float)     : Ratio point (between 0 and 1) at which to start sampling from MITGCM data
          end_ratio (float)       : Ratio point (between 0 and 1) at which to stop sampling from MITGCM data
          stride (integer)        : Rate at which to subsample in time when reading MITGCM data
          dataset_end_index       : Point of dataset file after which to ignore (i.e. cause model is no longer
                                    dynamically active
       """
 
       self.ds      = xr.open_dataset(MITGCM_filename, lock=False)
       self.start   = int(start_ratio * dataset_end_index)
       self.end     = int(end_ratio * dataset_end_index)
       self.stride  = stride
       self.hist_len= hist_len
       self.transform = transform
       self.land = land
       self.tic  = tic 
       
       #self.ds_inputs  = self.ds.isel(T=slice(self.start, self.end, self.stride))
       #self.ds_outputs = self.ds.isel(T=slice(self.start+1, self.end+1, self.stride))
  
       land=97  # T grid point where land starts

       # For now, manually set masks - eventually set to read in from MITgcm
       if self.land == 'IncLand':
          # Set dims based on T grid
          self.z_dim = (self.ds['THETA'].isel(T=0).values[:,:,:]).shape[0]
          self.y_dim = (self.ds['THETA'].isel(T=0).values[:,:,:]).shape[1] - 2   # Eventually re-run MITgcm with better grid size, for now ignore last rows of land!
          self.x_dim = (self.ds['THETA'].isel(T=0).values[:,:,:]).shape[2]

          self.T_mask              = np.ones(( self.z_dim, self.y_dim, self.x_dim ))
          self.T_mask[:,0,:]       = 0.0
          self.T_mask[:,land+1:,:] = 0.0
   
          self.U_mask              = np.ones(( self.z_dim, self.y_dim, self.x_dim ))
          self.U_mask[:,0,:]       = 0.0
          self.U_mask[:,land+1:,:] = 0.0
   
          self.V_mask              = np.ones(( self.z_dim, self.y_dim, self.x_dim ))
          self.V_mask[:,0,:]       = 0.0  # Set bottom row to land
          self.V_mask[:,1,:]       = 0.0
          self.V_mask[:,land+1:,:] = 0.0
   
          self.Eta_mask            = np.ones(( self.y_dim, self.x_dim ))
          self.Eta_mask[0,:]       = 0.0
          self.Eta_mask[land+1:,:] = 0.0
   
       elif self.land == 'ExcLand':
          # Set dims based on V grid
          self.z_dim = (self.ds['VVEL'].isel(T=0).values[:,1:land,:]).shape[0]
          self.y_dim = (self.ds['VVEL'].isel(T=0).values[:,1:land,:]).shape[1] 
          self.x_dim = (self.ds['VVEL'].isel(T=0).values[:,1:land,:]).shape[2]

          self.T_mask              = np.ones(( self.z_dim, self.y_dim, self.x_dim ))
          self.U_mask              = np.ones(( self.z_dim, self.y_dim, self.x_dim ))
          self.V_mask              = np.ones(( self.z_dim, self.y_dim, self.x_dim ))
          self.Eta_mask            = np.ones(( self.y_dim, self.x_dim ))
   
   def __len__(self):
       return int(self.ds.sizes['T']/self.stride)

   def __getitem__(self, idx):

       ds_InputSlice  = self.ds.isel(T=slice(idx*self.stride,idx*self.stride+self.hist_len))
       ds_OutputSlice = self.ds.isel(T=[idx*self.stride+self.hist_len])

       land=97  # T grid point where land starts

       if self.land == 'IncLand':
          # Read in the data
          da_T_in      = ds_InputSlice['THETA'].values[:,:,:-2,:] 
          da_T_out     = ds_OutputSlice['THETA'].values[:,:,:-2,:]
   
          da_U_in_tmp  = ds_InputSlice['UVEL'].values[:,:,:-2,:]
          da_U_in      = 0.5 * (da_U_in_tmp[:,:,:,:-1]+da_U_in_tmp[:,:,:,1:]) # average x dir onto same grid as T points
          da_U_out_tmp = ds_OutputSlice['UVEL'].values[:,:,:-2,:]
          da_U_out     = 0.5 * (da_U_out_tmp[:,:,:,:-1]+da_U_out_tmp[:,:,:,1:]) # average to get onto same grid as T points
   
          da_V_in      = ds_InputSlice['VVEL'].values[:,:,:-3,:] #ignore last 2 land points to get to same as T grid
          da_V_out     = ds_OutputSlice['VVEL'].values[:,:,:-3,:] #ignore last 2 land points to get to same as T grid 
   
          da_Eta_in    = ds_InputSlice['ETAN'].values[:,:,:-2,:]
          da_Eta_out   = ds_OutputSlice['ETAN'].values[:,:,:-2,:]
       
       elif self.land == 'ExcLand':
          # Just cut out the ocean parts of the grid, and leave mask as all zeros

          # Read in the data
          da_T_in_tmp  = ds_InputSlice['THETA'].values[:,:,:land,:] 
          da_T_in      = 0.5 * (da_T_in_tmp[:,:,:-1,:]+da_T_in_tmp[:,:,1:,:]) # average to get onto same grid as V points  
          da_T_out_tmp = ds_OutputSlice['THETA'].values[:,:,:land,:]
          da_T_out     = 0.5 * (da_T_out_tmp[:,:,:-1,:]+da_T_out_tmp[:,:,1:,:]) # average to get onto same grid as V points  

          da_U_in_tmp  = ds_InputSlice['UVEL'].values[:,:,:land,:]
          da_U_in_tmp  = 0.5 * (da_U_in_tmp[:,:,:,:-1]+da_U_in_tmp[:,:,:,1:]) # average x dir onto same grid as T points  
          da_U_in      = 0.5 * (da_U_in_tmp[:,:,:-1,:]+da_U_in_tmp[:,:,1:,:]) # average y dir onto same grid as V points  
          da_U_out_tmp = ds_OutputSlice['UVEL'].values[:,:,:land,:]
          da_U_out_tmp = 0.5 * (da_U_out_tmp[:,:,:,:-1]+da_U_out_tmp[:,:,:,1:])  # average to get onto same grid as T points
          da_U_out     = 0.5 * (da_U_out_tmp[:,:,:-1,:]+da_U_out_tmp[:,:,1:,:])  # average y dir onto same grid as V points  

          da_V_in      = ds_InputSlice['VVEL'].values[:,:,1:land,:]  # Ignore first point as zero
          da_V_out     = ds_OutputSlice['VVEL'].values[:,:,1:land,:]    # Ignore first point as zero

          da_Eta_in_tmp = ds_InputSlice['ETAN'].values[:,:,:land,:]
          da_Eta_in     = 0.5 * (da_Eta_in_tmp[:,:,:-1,:]+da_Eta_in_tmp[:,:,1:,:]) # average to get onto same grid as V points  
          da_Eta_out_tmp= ds_OutputSlice['ETAN'].values[:,:,:land,:]
          da_Eta_out    = 0.5 * (da_Eta_out_tmp[:,:,:-1,:]+da_Eta_out_tmp[:,:,1:,:]) # average to get onto same grid as V points  

       # Mask the data
       da_T_in  = np.where(self.T_mask==0, 0, da_T_in)
       da_T_out = np.where(self.T_mask==0, 0, da_T_out)
      
       da_U_in  = np.where(self.U_mask==0, 0, da_U_in)
       da_U_out = np.where(self.U_mask==0, 0, da_U_out)
      
       da_V_in  = np.where(self.V_mask==0, 0, da_V_in)
       da_V_out = np.where(self.V_mask==0, 0, da_V_out)
      
       da_Eta_in  = np.where(self.Eta_mask==0, 0, da_Eta_in)
       da_Eta_out = np.where(self.Eta_mask==0, 0, da_Eta_out)

       sample_input  = np.zeros(( 4*self.hist_len+4, self.z_dim, self.y_dim, self.x_dim ))  # shape: (no channels, z, y, x)
       sample_output = np.zeros(( 4, self.z_dim, self.y_dim, self.x_dim ))  # shape: (no channels, z, y, x)

       sample_input[:1*self.hist_len,:,:,:]  = da_T_in[:, :, :, :]
       sample_output[0,:,:,:] = da_T_out[:, :, :, :]

       sample_input[1*self.hist_len:2*self.hist_len,:,:,:]  = da_U_in[:, :, :, :]
       sample_output[1,:,:,:] = da_U_out[:, :, :, :]

       sample_input[2*self.hist_len:3*self.hist_len,:,:,:]  = da_V_in[:, :, :, :]
       sample_output[2,:,:,:] = da_V_out[:, :, :, :]

       sample_input[3*self.hist_len:4*self.hist_len,:,:,:]  = np.broadcast_to(da_Eta_in[:, :, :, :], (self.hist_len, self.z_dim, self.y_dim, self.x_dim))
       sample_output[3,:,:,:] = np.broadcast_to(da_Eta_out[0, :, :, :], (self.z_dim, self.y_dim, self.x_dim))

       # Cat masks onto inputs but not outputs
       sample_input[4*self.hist_len,:,:,:]  = self.T_mask[:, :, :]
       sample_input[4*self.hist_len+1,:,:,:]  = self.U_mask[:, :, :]
       sample_input[4*self.hist_len+2,:,:,:]  = self.V_mask[:, :, :]
       sample_input[4*self.hist_len+3,:,:,:]  = np.broadcast_to(self.Eta_mask[:, :],(self.z_dim, self.y_dim, self.x_dim))

       sample_input = torch.from_numpy(sample_input)
       sample_output = torch.from_numpy(sample_output)
 
       sample = {'input':sample_input, 'output':sample_output}
       
       if self.transform:
          sample = self.transform(sample)
 
       return (sample)


class RF_Normalise(object):
    """Normalise data based on training means and std (given)

    Args:
       Training input mean
       Training input std
       Training output mean
       Training output std
    """
    def __init__(self, train_mean, train_std, train_range):
        self.train_mean  = train_mean
        self.train_std   = train_std
        self.train_range = train_range

    def __call__(self, sample):
        sample_input, sample_output = sample['input'], sample['output']

        # Using transforms.Normalize returns the function, rather than the normalised array - can't figure out how to avoid this...
        #sample_input  = transforms.Normalize(sample_input, self.inputs_train_mean, self.inputs_train_std)
        #sample_output = transforms.Normalize(sample_output,self.outputs_train_mean, self.outputs_train_std)

        ## ONLY NORMALISE FOR NO OF CHANNELS... the input channels also have mask channels, but we don't want to normalise these. 
        for channel in range(sample_output.shape[0]):
           sample_input[channel, :, :]  = (sample_input[channel, :, :] - self.train_mean[channel]) / self.train_range[channel]
           sample_output[channel, :, :] = (sample_output[channel, :, :] - self.train_mean[channel]) / self.train_range[channel]

        return {'input':sample_input, 'output':sample_output}

def RF_DeNormalise(samples, train_mean, train_std, train_range, no_out_channels):
    """de-Normalise data based on training means and std (given)

    Args:
       Array to de-norm (expected shape - no_samples, no_channels, y, x)
       Training output mean
       Training output std
    """

    for channel in range(no_out_channels):
       samples[:,channel, :, :]  = (samples[:,channel, :, :] * train_range[channel] ) + train_mean[channel]

    return (samples)

