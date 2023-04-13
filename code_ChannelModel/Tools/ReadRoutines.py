#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import xarray as xr
from torch.utils import data
from torchvision import transforms, utils
import torch
import netCDF4 as nc4
import sys
sys.path.append('../CreateAndTrainModels')
from WholeGridNetworkRegressorModules import *

class ReadProcData_Dataset(data.Dataset):
   
   ''' 
        Dataset which reads in the processes data, and returns training/validation/test samples
   ''' 
   def __init__(self, model_name, ProcDataFilename, start_ratio, end_ratio, seed):

       os.environ['PYTHONHASHSEED'] = str(seed)
       np.random.seed(seed)
       torch.manual_seed(seed)
       torch.cuda.manual_seed(seed)
       torch.cuda.manual_seed_all(seed)
       torch.backends.cudnn.enabled = False
       torch.backends.cudnn.deterministic = True
       torch.backends.cudnn.benchmark = False
       torch.use_deterministic_algorithms(True, warn_only=True)

       self.ds          = xr.open_dataset(ProcDataFilename, lock=False)
       self.ds          = self.ds.isel(samples=slice( int(start_ratio*self.ds.dims['samples']), int(end_ratio*self.ds.dims['samples']) ) )
       self.da_inputs   = self.ds['inputs']
       self.da_targets  = self.ds['targets']
       self.da_extrafluxes = self.ds['extra_fluxes']
       self.masks       = self.ds['masks'].values
       self.out_masks   = self.ds['out_masks'].values
       self.masks       = torch.from_numpy(self.masks)
       self.out_masks   = torch.from_numpy(self.out_masks)
       self.seed        = seed

   def __len__(self):
       return self.ds.sizes['samples']

   def __getitem__(self, idx):

       os.environ['PYTHONHASHSEED'] = str(self.seed)
       np.random.seed(self.seed)
       torch.manual_seed(self.seed)
       torch.cuda.manual_seed(self.seed)
       torch.cuda.manual_seed_all(self.seed)
       torch.backends.cudnn.enabled = False
       torch.backends.cudnn.deterministic = True
       torch.backends.cudnn.benchmark = False
       torch.use_deterministic_algorithms(True, warn_only=True)

       sample_input = self.da_inputs.isel(samples=idx).values
       sample_target = self.da_targets.isel(samples=idx).values
       sample_extrafluxes = self.da_extrafluxes.isel(samples=idx).values

       sample_input = torch.from_numpy(sample_input)
       sample_target = torch.from_numpy(sample_target)
       sample_extrafluxes = torch.from_numpy(sample_extrafluxes)

       return sample_input, sample_target, sample_extrafluxes, self.masks, self.out_masks
   

# Create Dataset, which inherits the data.Dataset class
class MITgcm_Dataset(data.Dataset):

   '''
        MITgcm dataset

         This code is set up in '2d' mode so that the data is 2-d with different channels for each level of 
         each variable. This solves issues with eta being 2-d and other inputs being 3-d. This matches 
         what is done in Scher paper.
         Or in '3d' mode, so each variable is a channel, and each channel has 3d data. Here there are issues with
         Eta only having one depth level - we get around this by providing 38 identical fields... far from perfect!
   '''

   def __init__(self, MITgcm_filename, start_ratio, end_ratio, stride, histlen, rolllen, land, bdy_weight, land_values, 
                grid_filename, dim, model_style, no_phys_in_channels, no_out_channels, seed, transform=None):
       """
       Args:
          MITgcm_filename (string): Path to the MITgcm filename containing the data.
          start_ratio (float)     : Ratio point (between 0 and 1) at which to start sampling from MITgcm data
          end_ratio (float)       : Ratio point (between 0 and 1) at which to stop sampling from MITgcm data
          stride (integer)        : Rate at which to subsample in time when reading MITgcm data
       """
 
       os.environ['PYTHONHASHSEED'] = str(seed)
       np.random.seed(seed)
       torch.manual_seed(seed)
       torch.cuda.manual_seed(seed)
       torch.cuda.manual_seed_all(seed)
       torch.backends.cudnn.enabled = False
       torch.backends.cudnn.deterministic = True
       torch.backends.cudnn.benchmark = False
       torch.use_deterministic_algorithms(True, warn_only=True)

       self.seed      = seed
       self.ds        = xr.open_dataset(MITgcm_filename, lock=False)
       self.ds        = self.ds.isel( T=slice( int(start_ratio*self.ds.dims['T']), int(end_ratio*self.ds.dims['T']) ) )
       self.stride    = stride
       self.histlen   = histlen
       self.rolllen   = rolllen
       self.transform = transform
       self.land      = land
       no_depth_levels = 38
       self.land_values = land_values
       self.grid_ds   = xr.open_dataset(grid_filename, lock=False)
       self.dim       = dim
       self.model_style = model_style
       self.no_phys_in_channels = no_phys_in_channels
       self.no_out_channels = no_out_channels

       HfacC = self.grid_ds['HFacC'].values

       if self.land == 'ExcLand':
          # Set dims based on T grid
          self.z_dim = (self.ds['THETA'].isel(T=0).values[:,3:101,:]).shape[0]
          self.y_dim = (self.ds['THETA'].isel(T=0).values[:,3:101,:]).shape[1] 
          self.x_dim = (self.ds['THETA'].isel(T=0).values[:,3:101,:]).shape[2]
          # set mask, initialise as ocean (ones) everywhere
          self.masks = np.ones(( self.z_dim, self.y_dim, self.x_dim ))
          self.masks[:,:,:] = np.where( HfacC[:,3:101,:] > 0., 1, 0 )

       else:
          # Set dims based on T grid
          self.z_dim = (self.ds['THETA'].isel(T=0).values[:,:,:]).shape[0]
          self.y_dim = (self.ds['THETA'].isel(T=0).values[:,:,:]).shape[1]
          self.x_dim = (self.ds['THETA'].isel(T=0).values[:,:,:]).shape[2]
          # set mask, initialise as ocean (ones) everywhere
          self.masks = np.ones(( self.z_dim, self.y_dim, self.x_dim ))
          self.masks[:,:,:] = np.where( HfacC > 0., 1, 0 )

       if self.model_style == 'ConvLSTM' or self.model_style == 'UNetConvLSTM':
          self.out_masks = np.concatenate( (self.masks, self.masks, self.masks, self.masks[0:1,:,:]), axis=0)
          self.masks = np.broadcast_to( self.masks, (self.histlen, self.z_dim, self.y_dim, self.x_dim) )
       elif self.dim == '2d':
          self.out_masks = np.concatenate( (self.masks, self.masks, self.masks, self.masks[0:1,:,:]), axis=0)
       elif self.dim == '3d':
          self.out_masks = np.stack( (self.masks, self.masks, self.masks, self.masks), axis=0)
          self.masks = np.expand_dims( self.masks, axis=0 )  # Add channel dimension at front for catting onto inputs later

       self.masks = torch.from_numpy(self.masks)
       self.out_masks = torch.from_numpy(self.out_masks)

   def __len__(self):
       return int( (self.ds.sizes['T']-self.histlen) / self.stride )

   def __getitem__(self, idx):

       os.environ['PYTHONHASHSEED'] = str(self.seed)
       np.random.seed(self.seed)
       torch.manual_seed(self.seed)
       torch.cuda.manual_seed(self.seed)
       torch.cuda.manual_seed_all(self.seed)
       torch.backends.cudnn.enabled = False
       torch.backends.cudnn.deterministic = True
       torch.backends.cudnn.benchmark = False
       torch.use_deterministic_algorithms(True, warn_only=True)

       ds_InputSlice  = self.ds.isel(T=slice(idx*self.stride,idx*self.stride+self.histlen)) 
       ds_OutputSlice = self.ds.isel(T=slice(idx*self.stride+self.histlen,idx*self.stride+self.histlen+self.rolllen))
 
       if self.land == 'IncLand' or self.land == 'Spits':
          # Read in the data

          da_T_in        = ds_InputSlice['THETA'].values[:,:,:,:] 
          da_U_in_tmp    = ds_InputSlice['UVEL'].values[:,:,:,:]
          da_U_in        = 0.5 * (da_U_in_tmp[:,:,:,:-1]+da_U_in_tmp[:,:,:,1:])   # average x dir onto same grid as T points
          da_V_in_tmp    = ds_InputSlice['VVEL'].values[:,:,:,:]
          da_V_in        = 0.5 * (da_V_in_tmp[:,:,:-1,:]+da_V_in_tmp[:,:,1:,:])   # average y dir onto same grid as T points
          da_Eta_in      = ds_InputSlice['ETAN'].values[:,0,:,:]
          da_gtForc_in   = ds_InputSlice['gT_Forc'].values[:,0,:,:]
          da_taux_tmp    = ds_InputSlice['oceTAUX'].values[:,0,:,:]
          da_taux_in     = 0.5 * (da_taux_tmp[:,:,:-1]+da_taux_tmp[:,:,1:])

          da_T_out_tmp   = ds_OutputSlice['THETA'].values[:,:,:,:]
          da_U_out_tmp   = ds_OutputSlice['UVEL'].values[:,:,:,:]
          da_U_out_tmp   = 0.5 * (da_U_out_tmp[:,:,:,:-1]+da_U_out_tmp[:,:,:,1:]) # average to get onto same grid as T points
          da_V_out_tmp   = ds_OutputSlice['VVEL'].values[:,:,:,:]
          da_V_out_tmp   = 0.5 * (da_V_out_tmp[:,:,:-1,:]+da_V_out_tmp[:,:,1:,:]) # average to get onto same grid as T points
          da_Eta_out_tmp = ds_OutputSlice['ETAN'].values[:,0,:,:]
       
          da_gtForc_extra = ds_OutputSlice['gT_Forc'].values[:,0,:,:]
          da_taux_tmp     = ds_OutputSlice['oceTAUX'].values[:,0,:,:]
          da_taux_extra   = 0.5 * (da_taux_tmp[:,:,:-1]+da_taux_tmp[:,:,1:])

       elif self.land == 'ExcLand':
          # Just cut out the ocean parts of the grid
   
          # Read in the data
          da_T_in      = ds_InputSlice['THETA'].values[:,:,3:101,:] 
          da_U_in_tmp  = ds_InputSlice['UVEL'].values[:,:,3:101,:]
          da_U_in      = 0.5 * (da_U_in_tmp[:,:,:,:-1]+da_U_in_tmp[:,:,:,1:]) # average x dir onto same grid as T points  
          da_V_in_tmp  = ds_InputSlice['VVEL'].values[:,:,3:102,:]             # Extra land point in y dir compared to other variables
          da_V_in      = 0.5 * (da_V_in_tmp[:,:,:-1,:]+da_V_in_tmp[:,:,1:,:]) # average y dir onto same grid as T points  
          da_Eta_in    = ds_InputSlice['ETAN'].values[:,0,3:101,:]
          da_gtForc_in = ds_InputSlice['gT_Forc'].values[:,0,3:101,:]
          da_taux_tmp  = ds_InputSlice['oceTAUX'].values[:,0,3:101,:]
          da_taux_in   = 0.5 * (da_taux_tmp[:,:,:-1]+da_taux_tmp[:,:,1:])     # average x dir onto same grid as T points

          da_T_out_tmp   = ds_OutputSlice['THETA'].values[:,:,3:101,:]
          da_U_out_tmp   = ds_OutputSlice['UVEL'].values[:,:,3:101,:]
          da_U_out_tmp   = 0.5 * (da_U_out_tmp[:,:,:,:-1]+da_U_out_tmp[:,:,:,1:]) # average to get onto same grid as T points
          da_V_out_tmp   = ds_OutputSlice['VVEL'].values[:,:,3:102,:]              # Extra land point at South
          da_V_out_tmp   = 0.5 * (da_V_out_tmp[:,:,:-1,:]+da_V_out_tmp[:,:,1:,:]) # average to get onto same grid as T points
          da_Eta_out_tmp = ds_OutputSlice['ETAN'].values[:,0,3:101,:]

          da_gtForc_extra = ds_OutputSlice['gT_Forc'].values[:,0,:,:]
          da_taux_tmp     = ds_OutputSlice['oceTAUX'].values[:,0,:,:]
          da_taux_extra   = 0.5 * (da_taux_tmp[:,:,:-1]+da_taux_tmp[:,:,1:])

       da_T_out       = np.zeros(da_T_out_tmp.shape)
       da_U_out       = np.zeros(da_U_out_tmp.shape)
       da_V_out       = np.zeros(da_V_out_tmp.shape)
       da_Eta_out     = np.zeros(da_Eta_out_tmp.shape)

       da_T_out[0,:,:,:] = da_T_out_tmp[0,:,:,:] - da_T_in[-1,:,:,:]
       da_U_out[0,:,:,:] = da_U_out_tmp[0,:,:,:] - da_U_in[-1,:,:,:]
       da_V_out[0,:,:,:] = da_V_out_tmp[0,:,:,:] - da_V_in[-1,:,:,:]
       da_Eta_out[0,:,:] = da_Eta_out_tmp[0,:,:] - da_Eta_in[-1,:,:]

       if self.rolllen > 1:
          da_T_out[1:self.rolllen,:,:,:] = da_T_out[1:self.rolllen,:,:,:] - da_T_out[0:self.rolllen-1,:,:,:]
          da_U_out[1:self.rolllen,:,:,:] = da_U_out[1:self.rolllen,:,:,:] - da_U_out[0:self.rolllen-1,:,:,:]
          da_V_out[1:self.rolllen,:,:,:] = da_V_out[1:self.rolllen,:,:,:] - da_V_out[0:self.rolllen-1,:,:,:]
          da_Eta_out[1:self.rolllen,:,:] = da_Eta_out[1:self.rolllen,:,:] - da_Eta_out[0:self.rolllen-1,:,:]


       # Set up sample arrays and then fill (better for memory than concatenating)

       if self.model_style == 'ConvLSTM' or self.model_style == 'UNetConvLSTM':
          sample_input  = np.zeros(( self.histlen, self.no_phys_in_channels, self.y_dim, self.x_dim ))
          sample_target = np.zeros(( self.no_out_channels, self.y_dim, self.x_dim ))
          sample_extrafluxes = np.zeros(( 2*self.rolllen, self.y_dim, self.x_dim ))  # For use when iterating for roll out loss, not used here

          for time in range(self.histlen):
             sample_input[ time, :self.z_dim, :, : ]  = \
                                             da_T_in[time,:,:,:].reshape(-1,self.y_dim,self.x_dim)
             sample_input[ time, self.z_dim:2*self.z_dim, :, : ]  = \
                                             da_U_in[time,:,:,:].reshape(-1,self.y_dim,self.x_dim) 
             sample_input[ time, 2*self.z_dim:3*self.z_dim, :, : ]  = \
                                             da_V_in[time,:,:,:].reshape(-1,self.y_dim,self.x_dim)
             sample_input[ time, 3*self.z_dim, :, : ] = \
                                             da_Eta_in[time,:,:].reshape(-1,self.y_dim,self.x_dim)              
             sample_input[ time, 3*self.z_dim+1, :, : ] = \
                                             da_gtForc_in[time,:,:].reshape(-1,self.y_dim,self.x_dim)              
             sample_input[ time, 3*self.z_dim+2, :, : ] = \
                                             da_taux_in[time,:,:].reshape(-1,self.y_dim,self.x_dim)              

          for time in range(self.rolllen):
             sample_target[ (self.z_dim*3+1)*time : (self.z_dim*3+1)*time+self.z_dim, :, : ] = \
                                             da_T_out[time,:,:,:].reshape(-1,self.y_dim,self.x_dim)
             sample_target[ (self.z_dim*3+1)*time+self.z_dim : (self.z_dim*3+1)*time+2*self.z_dim, :, : ] = \
                                             da_U_out[time,:,:,:].reshape(-1,self.y_dim,self.x_dim)
             sample_target[ (self.z_dim*3+1)*time+2*self.z_dim : (self.z_dim*3+1)*time+3*self.z_dim, :, : ] = \
                                             da_V_out[time,:,:,:].reshape(-1,self.y_dim,self.x_dim)
             sample_target[ (self.z_dim*3+1)*time+3*self.z_dim, :, : ] = \
                                             da_Eta_out[time,:,:].reshape(-1,self.y_dim,self.x_dim)              

          # Mask data
          sample_input[:,:self.z_dim,:,:] = np.where( self.masks==1, sample_input[:,:self.z_dim,:,:],
                                                      np.expand_dims( self.land_values[:self.z_dim], axis=(0,2,3) ) )
          sample_input[:,self.z_dim:self.z_dim*2,:,:] = np.where( self.masks==1, sample_input[:,self.z_dim:self.z_dim*2,:,:],
                                                                  np.expand_dims( self.land_values[self.z_dim:self.z_dim*2], axis=(0,2,3) ) )
          sample_input[:,self.z_dim*2:self.z_dim*3,:,:] = np.where( self.masks==1, sample_input[:,self.z_dim*2:self.z_dim*3,:,:],
                                                                    np.expand_dims( self.land_values[self.z_dim*2:self.z_dim*3], axis=(0,2,3) ) )
          sample_input[:,self.z_dim*3,:,:] = np.where( self.masks[:,0,:,:]==1, sample_input[:,self.z_dim*3,:,:],
                                                       np.expand_dims( self.land_values[self.z_dim*3], axis=(0,1,2) ) )
          sample_input[:,self.z_dim*3+1,:,:] = np.where( self.masks[:,0,:,:]==1, sample_input[:,self.z_dim*3+1,:,:],
                                                         np.expand_dims( self.land_values[self.z_dim*3+1], axis=(0,1,2) ) )
          sample_input[:,self.z_dim*3+2,:,:] = np.where( self.masks[:,0,:,:]==1, sample_input[:,self.z_dim*3+2,:,:],
                                                         np.expand_dims( self.land_values[self.z_dim*3+2], axis=(0,1,2) ) )

          for time in range(self.rolllen):
             sample_target[time*(self.z_dim*3+1):(time+1)*(self.z_dim*3+1),:,:] = np.where( self.out_masks==1,
                                                                                  sample_target[time*(self.z_dim*3+1):(time+1)*(self.z_dim*3+1),:,:], 
                                                                                  np.expand_dims( self.land_values[:self.z_dim*3+1], axis=(1,2) ) )

       else:
          if self.dim == '2d':
             # Dims are channels,y,x
             sample_input  = np.zeros(( self.no_phys_in_channels*self.histlen, self.y_dim, self.x_dim ))
             sample_target = np.zeros(( self.no_out_channels*self.rolllen, self.y_dim, self.x_dim ))
             sample_extrafluxes = np.zeros(( 2*self.rolllen, self.y_dim, self.x_dim ))  # For use when iterating for roll out loss

             for time in range(self.histlen):
                sample_input[ (self.z_dim*3+3)*time : (self.z_dim*3+3)*time+self.z_dim, :, : ]  = \
                               np.where( self.masks==1, da_T_in[time,:,:,:].reshape(-1,self.y_dim,self.x_dim),
                                         np.expand_dims( self.land_values[0:self.z_dim], axis=(1,2)) )
         
                sample_input[ (self.z_dim*3+3)*time+self.z_dim : (self.z_dim*3+3)*time+2*self.z_dim, :, : ]  = \
                               np.where( self.masks==1, da_U_in[time,:,:,:].reshape(-1,self.y_dim,self.x_dim), 
                                         np.expand_dims( self.land_values[self.z_dim:2*self.z_dim], axis=(1,2)) )
         
                sample_input[ (self.z_dim*3+3)*time+2*self.z_dim : (self.z_dim*3+3)*time+3*self.z_dim, :, : ]  = \
                               np.where( self.masks==1, da_V_in[time,:,:,:].reshape(-1,self.y_dim,self.x_dim),
                                         np.expand_dims( self.land_values[2*self.z_dim:3*self.z_dim], axis=(1,2)) )
         
                sample_input[ (self.z_dim*3+3)*time+3*self.z_dim, :, : ] = \
                               np.where( self.masks[0,:,:]==1, da_Eta_in[time,:,:].reshape(-1,self.y_dim,self.x_dim),
                                         np.expand_dims( self.land_values[3*self.z_dim], axis=(0,1,2)) )
         
                sample_input[ (self.z_dim*3+3)*time+3*self.z_dim+1, :, : ] = \
                               np.where( self.masks[0,:,:]==1, da_gtForc_in[time,:,:].reshape(-1,self.y_dim,self.x_dim),
                                         np.expand_dims( self.land_values[3*self.z_dim+1], axis=(0,1,2)) )    

                sample_input[ (self.z_dim*3+3)*time+3*self.z_dim+2, :, : ] = \
                               np.where( self.masks[0,:,:]==1, da_taux_in[time,:,:].reshape(-1,self.y_dim,self.x_dim),
                                         np.expand_dims( self.land_values[3*self.z_dim+2] , axis=(0,1,2)) )

             for time in range(self.rolllen):
                sample_target[ (self.z_dim*3+1)*time:(self.z_dim*3+1)*time+self.z_dim, :, : ] = \
                                                da_T_out[time,:,:,:].reshape(-1,self.y_dim,self.x_dim)
                sample_target[ (self.z_dim*3+1)*time+self.z_dim:(self.z_dim*3+1)*time+2*self.z_dim, :, : ] = \
                                                da_U_out[time,:,:,:].reshape(-1,self.y_dim,self.x_dim)
                sample_target[ (self.z_dim*3+1)*time+2*self.z_dim:(self.z_dim*3+1)*time+3*self.z_dim, :, : ] = \
                                                da_V_out[time,:,:,:].reshape(-1,self.y_dim,self.x_dim)
                sample_target[ (self.z_dim*3+1)*time+3*self.z_dim, :, : ] = \
                                                da_Eta_out[time,:,:].reshape(-1,self.y_dim,self.x_dim)              
      
                sample_extrafluxes[ 2*time, :, : ] = \
                                    np.where( self.masks[0,:,:]==1, da_gtForc_extra[time,:,:].reshape(-1,self.y_dim,self.x_dim),
                                         np.expand_dims( self.land_values[3*self.z_dim+1], axis=(0,1,2)) )    

                sample_extrafluxes[ 2*time+1, :, : ] = \
                                     np.where( self.masks[0,:,:]==1, da_taux_extra[time,:,:].reshape(-1,self.y_dim,self.x_dim),
                                         np.expand_dims( self.land_values[3*self.z_dim+2] , axis=(0,1,2)) )

             # mask values
             for time in range(self.rolllen):
                sample_target[time*(self.no_out_channels):(time+1)*(self.no_out_channels),:,:] = np.where( self.out_masks==1,
                                                                   sample_target[time*self.no_out_channels:(time+1)*self.no_out_channels,:,:], 
                                                                   np.expand_dims( self.land_values[:self.no_out_channels], axis=(1,2) ) )

          elif self.dim == '3d':
             # Dims are channels,z,y,x
             sample_input  = np.zeros(( self.no_phys_in_channels*histlen, self.z_dim, self.y_dim, self.x_dim ))  # shape: (no channels, z, y, x)
             sample_target = np.zeros(( self.no_out_channels, self.z_dim, self.y_dim, self.x_dim ))  # shape: (no channels, z, y, x)
             sample_extrafluxes = np.zeros(( 2*self.rolllen, self.z_dim, self.y_dim, self.x_dim ))  # For iterating for roll out loss, not used here
      
             for time in range(self.histlen):
                sample_input[time*4,:,:,:]   = da_T_in[time, :, :, :]
                sample_input[time*4+1,:,:,:] = da_U_in[time, :, :, :]
                sample_input[time*4+2,:,:,:] = da_V_in[time, :, :, :]
                sample_input[time*4+3,:,:,:] = np.broadcast_to(da_Eta_in[time, :, :], (1, self.z_dim, self.y_dim, self.x_dim))
                sample_input[time*4+4,:,:,:] = np.broadcast_to(da_gtForc_in[time, :, :], (1, self.z_dim, self.y_dim, self.x_dim))
                sample_input[time*4+5,:,:,:] = np.broadcast_to(da_taux_in[time, :, :], (1, self.z_dim, self.y_dim, self.x_dim))
      
             sample_target[0,:,:,:] = da_T_out[:, :, :, :]
             sample_target[1,:,:,:] = da_U_out[:, :, :, :]
             sample_target[2,:,:,:] = da_V_out[:, :, :, :]
             sample_target[3,:,:,:] = np.broadcast_to(da_Eta_out[:, :, :], (1, self.z_dim, self.y_dim, self.x_dim))
      
             # mask values
             for time in range(self.histlen):
                 sample_input[time*6:(time+1)*6,:,:,:] = np.where( self.out_masks==1,
                                                                   sample_input[time*6:(time+1)*6,:,:,:], 
                                                                   np.expand_dims( self.land_values, axis=(1,2,3)) )
             sample_target[:,:,:,:] = np.where( self.out_masks==1, sample_target[:,:,:], np.expand_dims( self.land_values, axis=(1,2,3) ) )

       sample_input = torch.from_numpy(sample_input)
       sample_target = torch.from_numpy(sample_target)
       sample_extrafluxes = torch.from_numpy(sample_extrafluxes)

       if self.transform:
          sample_input, sample_target = self.transform({'input':sample_input, 'target':sample_target})
 
       return sample_input, sample_target, sample_extrafluxes, self.masks, self.out_masks

def CreateProcDataset( MITgcm_filename, ProcDataFilename, subsample_rate, histlen, rolllen, land, bdyweight, landvalues, grid_filename, dim, 
                       modelstyle, mean_std_file, z_dim, y_dim_used, no_phys_in_channels, no_out_channels, normmethod, model_name, seed):

    print(ProcDataFilename)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    batch_size = 256   # hard coded as set to max viable and no benefit to changing

    # Open file for dimension etc later
    MITgcm_ds = xr.open_dataset(MITgcm_filename, lock=False)
    MITgcm_ds.close()
  
    # Read in mean, std, range
    inputs_mean, inputs_std, inputs_range, targets_mean, targets_std, targets_range = \
                                          ReadMeanStd(mean_std_file, dim, no_phys_in_channels, no_out_channels, z_dim, seed)

    # Get data using pytorch dataloader
    MITgcm_Dataset = rr.MITgcm_Dataset( MITgcm_filename, 0.0, 1.0, subsample_rate, histlen, rolllen, land, bdyweight, landvalues,
                                        grid_filename, dim, modelstyle, no_phys_in_channels, no_out_channels, seed,
                                        transform = transforms.Compose( [ rr.RF_Normalise_sample(inputs_mean, inputs_std, inputs_range,
                                                                          targets_mean, targets_std, targets_range, histlen, rolllen,
                                                                          no_phys_in_channels, no_out_channels, dim, normmethod, modelstyle, seed)] ) )

    MITgcm_loader = torch.utils.data.DataLoader(MITgcm_Dataset, batch_size=batch_size, shuffle=False,
                                                num_workers=1, pin_memory=True )

    no_samples = len(MITgcm_Dataset)
    X_size = MITgcm_ds['X'].shape[0]
    Y_size = MITgcm_ds['Y'].shape[0]
    Z_size = MITgcm_ds['Zmd000038'].shape[0]

    print(X_size)
    print(Y_size)
    print(Z_size)

    batch = 0

    if batch == 0:
       out_file = nc4.Dataset(ProcDataFilename,'w', format='NETCDF4')
       # Create dimensions
       out_file.createDimension('samples', no_samples)
       out_file.createDimension('histlen', histlen)
       out_file.createDimension('out_channels', no_out_channels*rolllen)
       out_file.createDimension('out_mask_channels', no_out_channels)
       out_file.createDimension('flux_channels', 2*rolllen)
       out_file.createDimension('X', X_size)
       out_file.createDimension('Y', y_dim_used)
       out_file.createDimension('Z', Z_size)
       # Create dimension variables
       nc_samples = out_file.createVariable( 'samples', 'i4', 'samples' )
       nc_histlen = out_file.createVariable( 'histlen', 'i4', 'histlen' )
       nc_out_channels = out_file.createVariable( 'out_channels', 'i4', 'out_channels' )
       nc_out_mask_channels = out_file.createVariable( 'out_mask_channels', 'i4', 'out_mask_channels' )
       nc_flux_channels = out_file.createVariable( 'flux_channels', 'i4', 'flux_channels' )
       nc_X = out_file.createVariable( 'X', 'i4', 'X' )
       nc_Y = out_file.createVariable( 'Y', 'i4', 'Y' )
       nc_Z = out_file.createVariable( 'Z', 'i4', 'Z' )
       # Fill dimension variables
       nc_samples[:] = np.arange(no_samples)
       nc_histlen[:] = np.arange(histlen)
       nc_out_channels[:] = np.arange(no_out_channels*rolllen)
       nc_out_mask_channels[:] = np.arange(no_out_channels)
       nc_flux_channels[:] = np.arange(2*rolllen)
       nc_X[:] = MITgcm_ds['X'].values
       if land == 'ExcLand':
          nc_Y[:] = MITgcm_ds['Y'].values[3:101]
       else:
          nc_Y[:] = MITgcm_ds['Y'].values
       nc_Z[:] = MITgcm_ds['Zmd000038'].values
       # Create variables
       if modelstyle == 'ConvLSTM' or modelstyle == 'UNetConvLSTM':
          out_file.createDimension('in_channels', no_phys_in_channels)
          nc_in_channels = out_file.createVariable( 'in_channels', 'i4',  'in_channels')
          nc_in_channels[:] = np.arange(no_phys_in_channels)
          nc_inputs    = out_file.createVariable( 'inputs',    'f4', ('samples', 'histlen', 'in_channels', 'Y', 'X') )
          nc_targets   = out_file.createVariable( 'targets',   'f4', ('samples', 'out_channels', 'Y', 'X') )
          nc_extrafluxes = out_file.createVariable( 'extra_fluxes',   'f4', ('samples', 'flux_channels', 'Y', 'X') )
          nc_masks     = out_file.createVariable( 'masks',     'f4', ('histlen', 'Z', 'Y', 'X') )
          nc_out_masks = out_file.createVariable( 'out_masks', 'f4', ('out_mask_channels', 'Y', 'X') )
       elif dim == '2d':
          out_file.createDimension('in_channels', no_phys_in_channels*histlen)
          nc_in_channels = out_file.createVariable( 'in_channels', 'i4',  'in_channels')
          nc_in_channels[:] = np.arange(no_phys_in_channels*histlen)
          nc_inputs    = out_file.createVariable( 'inputs',    'f4', ('samples', 'in_channels', 'Y', 'X') )
          nc_targets   = out_file.createVariable( 'targets',   'f4', ('samples', 'out_channels', 'Y', 'X') )
          nc_extrafluxes = out_file.createVariable( 'extra_fluxes',   'f4', ('samples', 'flux_channels', 'Y', 'X') )
          nc_masks     = out_file.createVariable( 'masks',     'f4', ('Z', 'Y', 'X') )
          nc_out_masks = out_file.createVariable( 'out_masks', 'f4', ('out_mask_channels', 'Y', 'X') )
       elif dim == '3d':
          out_file.createDimension('in_channels', no_phys_in_channels*histlen)
          nc_in_channels = out_file.createVariable( 'in_channels', 'i4',  'in_channels')
          nc_in_channels[:] = np.arange(no_phys_in_channels*histlen)
          nc_inputs    = out_file.createVariable( 'inputs',    'f4', ('samples', 'in_channels', 'Z', 'Y', 'X') )
          nc_targets   = out_file.createVariable( 'targets',   'f4', ('samples', 'out_channels', 'Z', 'Y', 'X') )
          nc_extrafluxes = out_file.createVariable( 'extra_fluxes',   'f4', ('samples', 'flux_channels', 'Z', 'Y', 'X') )
          nc_masks     = out_file.createVariable( 'masks',     'f4', (1, 'Z', 'Y', 'X') )
          nc_out_masks = out_file.createVariable( 'out_masks', 'f4', ('out_mask_channels', 'Z', 'Y', 'X') )
       out_file.close()

    # Fill Dataset
    for input_batch, target_batch, extrafluxes_batch, masks, out_masks in MITgcm_loader:
       print('batch number : '+str(batch))
       out_file = nc4.Dataset(ProcDataFilename,'r+', format='NETCDF4')
       nc_inputs[batch*batch_size:(batch+1)*batch_size] = input_batch
       nc_targets[batch*batch_size:(batch+1)*batch_size] = target_batch
       nc_extrafluxes[batch*batch_size:(batch+1)*batch_size] = extrafluxes_batch
       if batch == 0:
          nc_masks[:] = masks[0,]
          print(nc_out_masks.shape)
          print(out_masks.shape)
          print(out_masks[0,].shape)
          nc_out_masks[:] = out_masks[0,]
       batch = batch+1
       out_file.close()


class RF_Normalise_sample(object):

    def __init__(self, inputs_mean, inputs_std, inputs_range, targets_mean, targets_std, targets_range, 
                 histlen, rolllen, no_phys_in_channels, no_out_channels, dim, norm_method, model_style, seed):

        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)

        self.seed          = seed
        self.inputs_mean   = inputs_mean
        self.inputs_std    = inputs_std
        self.inputs_range  = inputs_range
        self.targets_mean  = targets_mean
        self.targets_std   = targets_std
        self.targets_range = targets_range
        self.histlen       = histlen
        self.rolllen       = rolllen
        self.no_phys_in_channels = no_phys_in_channels
        self.no_out_channels = no_out_channels
        self.dim           = dim
        self.norm_method   = norm_method
        self.model_style   = model_style

    def __call__(self, sample):

        # Using transforms.Normalize returns the function, rather than the normalised array - can't figure out how to avoid this...
        #sample_input  = transforms.Normalize(sample_input, self.inputs_inputs_mean, self.inputs_inputs_std)
        #sample_target = transforms.Normalize(sample_target,self.targets_inputs_mean, self.targets_inputs_std)

        # only normalise for channels with physical variables (= no_out channels*histlen)..
        # the input channels also have mask channels, but we don't want to normalise these. 

        os.environ['PYTHONHASHSEED'] = str(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)

        if self.model_style == 'ConvLSTM' or self.model_style == 'UNetConvLSTM':
           for time in range(self.histlen):
              sample['input'][time,:self.no_phys_in_channels, :, :] =                   \
                                   RF_Normalise( sample['input'][time,:self.no_phys_in_channels, :, :],
                                                 self.inputs_mean, self.inputs_std, self.inputs_range, self.dim, self.norm_method, self.seed )
           for time in range(self.rolllen):
              sample['target'][time*self.no_out_channels:(time+1)*self.no_out_channels, :, :] =                   \
                                   RF_Normalise( sample['target'][time*self.no_out_channels:(time+1)*self.no_out_channels, :, :],
                                                 self.targets_mean, self.targets_mean, self.targets_range, self.dim, self.norm_method, self.seed )
        elif self.dim == '2d':
           for time in range(self.histlen):
              sample['input'][time*self.no_phys_in_channels:(time+1)*self.no_phys_in_channels, :, :] =                   \
                                   RF_Normalise( sample['input'][time*self.no_phys_in_channels:(time+1)*self.no_phys_in_channels, :, :],
                                                 self.inputs_mean, self.inputs_std, self.inputs_range, self.dim, self.norm_method, self.seed )
           for time in range(self.rolllen):
              sample['target'][time*self.no_out_channels:(time+1)*self.no_out_channels, :, :] =                   \
                                   RF_Normalise( sample['target'][time*self.no_out_channels:(time+1)*self.no_out_channels, :, :],
                                                 self.targets_mean, self.targets_mean, self.targets_range, self.dim, self.norm_method, self.seed )
        elif self.dim == '3d':
           for time in range(self.histlen):
              sample['input'][time*self.no_phys_in_channels:(time+1)*self.no_phys_in_channels, :, :, :] =                   \
                                   RF_Normalise( sample['input'][time*self.no_phys_in_channels:(time+1)*self.no_phys_in_channels, :, :, :],
                                                 self.inputs_mean, self.inputs_std, self.inputs_range,self.dim, self.norm_method, self.seed )
           for time in range(self.rolllen):
              sample['target'][time*self.no_out_channels:(time+1)*self.no_out_channels, :, :, :] =                   \
                                   RF_Normalise( sample['target'][time*self.no_out_channels:(time+1)*self.no_out_channels, :, :, :],
                                                 self.targets_mean, self.targets_mean, self.targets_range, self.dim, self.norm_method, self.seed )


        return sample['input'], sample['target']

def RF_Normalise(data, d_mean, d_std, d_range, dim, norm_method, seed ):
    """Normalise data based on training means and std (given)

       ReNormalises prediction for use in iteration
       dims of data: channels(, z), y, x

    Args:
       TBD
    """

    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    if norm_method == 'range':
       if dim == '2d':
          data[:,:,:] = ( data[:,:,:] - d_mean[:, np.newaxis, np.newaxis] )                      \
                                       / d_range[:, np.newaxis, np.newaxis]
       elif dim == '3d':
          data[:,:,:,:] = ( data[:,:,:,:] - d_mean[:, np.newaxis, np.newaxis, np.newaxis] )    \
                                       / d_range[:, np.newaxis, np.newaxis, np.newaxis]
    elif norm_method == 'std':
       if dim == '2d':
          data[:,:,:] = ( data[:,:,:] - d_mean[:, np.newaxis, np.newaxis] )                      \
                                       / d_std[:, np.newaxis, np.newaxis]
       elif dim == '3d':
          data[:,:,:,:] = ( data[:,:,:,:] - d_mean[:, np.newaxis, np.newaxis, np.newaxis] )    \
                                       / d_std[:, np.newaxis, np.newaxis, np.newaxis]
   
    return data

def RF_DeNormalise(samples, data_mean, data_std, data_range, dim, norm_method, seed):
    """de-Normalise data based on training means and std (given)

    Args:
       Array to de-norm (expected shape - no_samples, no_channels,( z,) y, x)
       Training output mean
       Training output std
    """

    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
 
    if norm_method == 'range':
       if dim == '2d':
          samples[:,:,:,:]  = ( samples[:,:,:,:] * 
                                                  data_range[np.newaxis, :, np.newaxis, np.newaxis] ) + \
                                                  data_mean[np.newaxis, :, np.newaxis, np.newaxis]
       elif dim == '3d':
          samples[:,:,:,:,:]  = ( samples[:,:,:,:,:] * 
                                                     data_range[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis] ) + \
                                                     data_mean[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis]
    elif norm_method == 'std':
       if dim == '2d':
          samples[:,:,:,:]  = ( samples[:,:,:,:] * 
                                                  data_std[np.newaxis, :, np.newaxis, np.newaxis] ) + \
                                                  data_mean[np.newaxis, :, np.newaxis, np.newaxis]
       elif dim == '3d':
          samples[:,:,:,:,:]  = ( samples[:,:,:,:,:] * 
                                                     data_std[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis] ) + \
                                                     data_mean[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis]

    return (samples)

