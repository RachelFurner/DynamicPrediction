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
from WholeGridNetworkRegressorModules import *

import netCDF4 as nc4

class ProcData_Dataset(data.Dataset):
   
   ''' 
        Dataset which reads in the processes data, and returns training/validation/test samples
   ''' 
   def __init__(self, model_name, start_ratio, end_ratio):
       ProcDataFilename = '../../../Channel_nn_Outputs/DATASETS/Dataset_'+model_name+'.nc'
       self.ds          = xr.open_dataset(ProcDataFilename, lock=False)
       self.ds          = self.ds.isel(samples=slice( int(start_ratio*self.ds.dims['samples']), int(end_ratio*self.ds.dims['samples']) ) )
       self.da_inputs   = self.ds['inputs']
       self.da_targets  = self.ds['targets']
       self.masks       = self.ds['masks'].values
       self.out_masks   = self.ds['out_masks'].values
       self.masks       = torch.from_numpy(self.masks)
       self.out_masks   = torch.from_numpy(self.out_masks)

   def __len__(self):
       return self.ds.sizes['samples']

   def __getitem__(self, idx):

       sample_input = self.da_inputs.isel(samples=idx).values
       sample_output = self.da_targets.isel(samples=idx).values
       sample_input = torch.from_numpy(sample_input)
       sample_output = torch.from_numpy(sample_output)

       return sample_input, sample_output, self.masks, self.out_masks
   

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

   def __init__(self, MITgcm_filename, start_ratio, end_ratio, stride, histlen, predlen, land, tic, bdy_weight, land_values, grid_filename, dim, model_style, transform=None):
       """
       Args:
          MITgcm_filename (string): Path to the MITgcm filename containing the data.
          start_ratio (float)     : Ratio point (between 0 and 1) at which to start sampling from MITgcm data
          end_ratio (float)       : Ratio point (between 0 and 1) at which to stop sampling from MITgcm data
          stride (integer)        : Rate at which to subsample in time when reading MITgcm data
       """
 
       self.ds        = xr.open_dataset(MITgcm_filename, lock=False)
       self.ds        = self.ds.isel( T=slice( int(start_ratio*self.ds.dims['T']), int(end_ratio*self.ds.dims['T']) ) )
       self.stride    = stride
       self.histlen   = histlen
       self.predlen   = predlen
       self.transform = transform
       self.land      = land
       self.tic       = tic 
       no_depth_levels = 38
       self.land_values = land_values
       self.grid_ds   = xr.open_dataset(grid_filename, lock=False)
       self.dim       = dim
       self.model_style = model_style

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
       HfacC = self.grid_ds['HFacC'].values

       self.masks = np.ones(( self.z_dim, self.y_dim, self.x_dim ))
       self.masks[:,:,:] = np.where( HfacC > 0., 1, 0 )

       if self.model_style == 'ConvLSTM' or self.model_style == 'UNetConvLSTM':
          self.out_masks = np.concatenate( (self.masks, self.masks, self.masks, self.masks[0:1,:,:]), axis=0)
          self.masks = np.broadcast_to( self.masks, (self.histlen, self.z_dim, self.y_dim, self.x_dim) )
       elif self.dim == '2d':
          self.out_masks = np.concatenate( (self.masks, self.masks, self.masks, self.masks[0:1,:,:]), axis=0)
       elif self.dim == '3d':
          self.out_masks = np.stack( (self.masks, self.masks, self.masks, self.masks), axis=0)
          self.masks = np.expand_dims( self.masks, axis=0 )

       self.masks = torch.from_numpy(self.masks)
       self.out_masks = torch.from_numpy(self.out_masks)

   def __len__(self):
       return int(self.ds.sizes['T']/self.stride)

   def __getitem__(self, idx):

       ds_InputSlice  = self.ds.isel(T=slice(idx*self.stride,idx*self.stride+self.histlen))
       ds_OutputSlice = self.ds.isel(T=slice(idx*self.stride+self.histlen,idx*self.stride+self.histlen+self.predlen))

       if self.land == 'IncLand' or self.land == 'Spits':
          # Read in the data

          da_T_in        = ds_InputSlice['THETA'].values[:,:,:,:] 
          da_U_in_tmp    = ds_InputSlice['UVEL'].values[:,:,:,:]
          da_U_in        = 0.5 * (da_U_in_tmp[:,:,:,:-1]+da_U_in_tmp[:,:,:,1:])   # average x dir onto same grid as T points
          da_V_in        = ds_InputSlice['VVEL'].values[:,:,:-1,:]                #ignore last land point to get to same as T grid
          da_Eta_in      = ds_InputSlice['ETAN'].values[:,0,:,:]

          da_T_out_tmp   = ds_OutputSlice['THETA'].values[:,:,:,:]
          da_U_out_tmp   = ds_OutputSlice['UVEL'].values[:,:,:,:]
          da_U_out_tmp   = 0.5 * (da_U_out_tmp[:,:,:,:-1]+da_U_out_tmp[:,:,:,1:]) # average to get onto same grid as T points
          da_V_out_tmp   = ds_OutputSlice['VVEL'].values[:,:,:-1,:]               #ignore last land point to get to same as T grid 
          da_Eta_out_tmp = ds_OutputSlice['ETAN'].values[:,0,:,:]
       
          da_T_out       = np.zeros(da_T_out_tmp.shape)
          da_U_out       = np.zeros(da_U_out_tmp.shape)
          da_V_out       = np.zeros(da_V_out_tmp.shape)
          da_Eta_out     = np.zeros(da_Eta_out_tmp.shape)

          da_T_out[0,:,:,:] = da_T_out_tmp[0,:,:,:] - da_T_in[-1,:,:,:]
          da_U_out[0,:,:,:] = da_U_out_tmp[0,:,:,:] - da_U_in[-1,:,:,:]
          da_V_out[0,:,:,:] = da_V_out_tmp[0,:,:,:] - da_V_in[-1,:,:,:]
          da_Eta_out[0,:,:] = da_Eta_out_tmp[0,:,:] - da_Eta_in[-1,:,:]

          if self.predlen > 1:
             da_T_out[1:self.predlen,:,:,:] = da_T_out[1:self.predlen,:,:,:] - da_T_out[0:self.predlen-1,:,:,:]
             da_U_out[1:self.predlen,:,:,:] = da_U_out[1:self.predlen,:,:,:] - da_U_out[0:self.predlen-1,:,:,:]
             da_V_out[1:self.predlen,:,:,:] = da_V_out[1:self.predlen,:,:,:] - da_V_out[0:self.predlen-1,:,:,:]
             da_Eta_out[1:self.predlen,:,:] = da_Eta_out[1:self.predlen,:,:] - da_Eta_out[0:self.predlen-1,:,:]

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

       if self.model_style == 'ConvLSTM' or self.model_style == 'UNetConvLSTM':
          if self.dim == '2d':
             sample_input  = np.zeros(( self.histlen, 3*self.z_dim+1, self.y_dim, self.x_dim ))
             sample_output = np.zeros(( (1+3*self.z_dim)*self.predlen, self.y_dim, self.x_dim ))
             for time in range(self.histlen):
                sample_input[ time, :self.z_dim, :, : ]  = \
                                                da_T_in[time,:,:,:].reshape(-1,self.y_dim,self.x_dim)
                sample_input[ time, self.z_dim:2*self.z_dim, :, : ]  = \
                                                da_U_in[time,:,:,:].reshape(-1,self.y_dim,self.x_dim) 
                sample_input[ time, 2*self.z_dim:3*self.z_dim, :, : ]  = \
                                                da_V_in[time,:,:,:].reshape(-1,self.y_dim,self.x_dim)
                sample_input[ time, 3*self.z_dim, :, : ] = \
                                                da_Eta_in[time,:,:].reshape(-1,self.y_dim,self.x_dim)              

             for time in range(self.predlen):
                sample_output[ (self.z_dim*3+1)*time : (self.z_dim*3+1)*time+self.z_dim, :, : ] = \
                                                da_T_out[time,:,:,:].reshape(-1,self.y_dim,self.x_dim)
                sample_output[ (self.z_dim*3+1)*time+self.z_dim : (self.z_dim*3+1)*time+2*self.z_dim, :, : ] = \
                                                da_U_out[time,:,:,:].reshape(-1,self.y_dim,self.x_dim)
                sample_output[ (self.z_dim*3+1)*time+2*self.z_dim : (self.z_dim*3+1)*time+3*self.z_dim, :, : ] = \
                                                da_V_out[time,:,:,:].reshape(-1,self.y_dim,self.x_dim)
                sample_output[ (self.z_dim*3+1)*time+3*self.z_dim, :, : ] = \
                                                da_Eta_out[time,:,:].reshape(-1,self.y_dim,self.x_dim)              

             # Mask data
             for time in range(self.histlen):
                sample_input[time,:3*self.z_dim+1,:,:] = np.where( self.out_masks==1,
                                                     sample_input[time,:3*self.z_dim+1,:,:], np.expand_dims( self.land_values, axis=(0,2,3) ) )

             for time in range(self.predlen):
                sample_output[time*(self.z_dim*3+1):(time+1)*(self.z_dim*3+1),:,:] = np.where( self.out_masks==1,
                                                                                     sample_output[time*(self.z_dim*3+1):(time+1)*(self.z_dim*3+1),:,:], 
                                                                                     np.expand_dims( self.land_values, axis=(1,2) ) )

       else:
          if self.dim == '2d':
             # Dims are channels,y,x
             sample_input  = np.zeros(( self.histlen*(1+3*self.z_dim), self.y_dim, self.x_dim ))
             sample_output = np.zeros(( self.predlen*(1+3*self.z_dim), self.y_dim, self.x_dim ))
      
             for time in range(self.histlen):
                sample_input[ self.z_dim*3*time+time : self.z_dim*3*time+time+self.z_dim, :, : ]  = \
                                                da_T_in[time,:,:,:].reshape(-1,self.y_dim,self.x_dim)
         
                sample_input[ self.z_dim*3*time+time+self.z_dim : self.z_dim*3*time+time+2*self.z_dim, :, : ]  = \
                                                da_U_in[time,:,:,:].reshape(-1,self.y_dim,self.x_dim) 
         
                sample_input[ self.z_dim*3*time+time+2*self.z_dim : self.z_dim*3*time+time+3*self.z_dim, :, : ]  = \
                                                da_V_in[time,:,:,:].reshape(-1,self.y_dim,self.x_dim)
         
                sample_input[ self.z_dim*3*time+time+3*self.z_dim, :, : ] = \
                                                da_Eta_in[time,:,:].reshape(-1,self.y_dim,self.x_dim)              
         
             for time in range(self.predlen):
                sample_output[ self.z_dim*3*time+time:self.z_dim*3*time+time+self.z_dim, :, : ] = \
                                                da_T_out[time,:,:,:].reshape(-1,self.y_dim,self.x_dim)
                sample_output[ self.z_dim*3*time+time+self.z_dim:self.z_dim*3*time+time+2*self.z_dim, :, : ] = \
                                                da_U_out[time,:,:,:].reshape(-1,self.y_dim,self.x_dim)
                sample_output[ self.z_dim*3*time+time+2*self.z_dim:self.z_dim*3*time+time+3*self.z_dim, :, : ] = \
                                                da_V_out[time,:,:,:].reshape(-1,self.y_dim,self.x_dim)
                sample_output[ self.z_dim*3*time+time+3*self.z_dim, :, : ] = \
                                                da_Eta_out[time,:,:].reshape(-1,self.y_dim,self.x_dim)              
      
             # mask values
             for time in range(self.histlen):
                 sample_input[time*(1+3*self.z_dim):(time+1)*(1+3*self.z_dim),:,:] = np.where( self.out_masks==1, 
                                                                                               sample_input[time*(1+3*self.z_dim):(time+1)*(1+3*self.z_dim),:,:],
                                                                                               np.expand_dims( self.land_values, axis=(1,2)) )
             for time in range(self.predlen):
                sample_output[time*(self.z_dim*3+1):(time+1)*(self.z_dim*3+1),:,:] = np.where( self.out_masks==1,
                                                                                     sample_output[time*(self.z_dim*3+1):(time+1)*(self.z_dim*3+1),:,:], 
                                                                                     np.expand_dims( self.land_values, axis=(1,2) ) )

          elif self.dim == '3d':
             # Dims are channels,z,y,x
             sample_input  = np.zeros(( 4*self.histlen, self.z_dim, self.y_dim, self.x_dim ))  # shape: (no channels, z, y, x)
             sample_output = np.zeros(( 4, self.z_dim, self.y_dim, self.x_dim ))  # shape: (no channels, z, y, x)
      
             for time in range(self.histlen):
                sample_input[time*4,:,:,:]   = da_T_in[time, :, :, :]
                sample_input[time*4+1,:,:,:] = da_U_in[time, :, :, :]
                sample_input[time*4+2,:,:,:] = da_V_in[time, :, :, :]
                sample_input[time*4+3,:,:,:] = np.broadcast_to(da_Eta_in[time, :, :], (1, self.z_dim, self.y_dim, self.x_dim))
      
             sample_output[0,:,:,:] = da_T_out[:, :, :, :]
             sample_output[1,:,:,:] = da_U_out[:, :, :, :]
             sample_output[2,:,:,:] = da_V_out[:, :, :, :]
             sample_output[3,:,:,:] = np.broadcast_to(da_Eta_out[:, :, :], (1, self.z_dim, self.y_dim, self.x_dim))
      
             # mask values
             for time in range(self.histlen):
                 sample_input[time*4:(time+1)*4,:,:,:] = np.where( self.out_masks==1,
                                                                   sample_input[time*4:(time+1)*4,:,:,:], 
                                                                   np.expand_dims( self.land_values, axis=(1,2,3)) )
             sample_output[:,:,:,:] = np.where( self.out_masks==1, sample_output[:,:,:], np.expand_dims( self.land_values, axis=(1,2,3) ) )

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
 
       return sample_input, sample_output, self.masks, self.out_masks



def CreateDataset( MITgcm_filename, subsample_rate, histlen, predlen, land, tic, bdyweight, landvalues, grid_filename, dim, 
                   modelstyle, predictionjump, z_dim, no_phys_channels, no_in_channels, no_out_channels, normmethod, model_name):

    batch_size = 256   # hard coded as set to max viable and no benefit to changing

    # Open file for dimension etc later
    MITgcm_ds = xr.open_dataset(MITgcm_filename, lock=False)
    #MITgcm_ds = xr.open_dataset(MITgcm_filename)
    MITgcm_ds.close()
  
    # Read in mean, std, range
    inputs_mean, inputs_std, inputs_range, targets_mean, targets_std, targets_range = ReadMeanStd(dim, z_dim, predictionjump)

    # Get data using pytorch dataloader
    MITgcm_Dataset = rr.MITgcm_Dataset( MITgcm_filename, 0.0, 1.0, subsample_rate,
                                       histlen, predlen, land, tic, bdyweight, landvalues, grid_filename, dim, modelstyle,
                                       transform = transforms.Compose( [ rr.RF_Normalise_sample(inputs_mean, inputs_std, inputs_range,
                                                                         targets_mean, targets_std, targets_range, histlen, predlen,
                                                                         no_phys_channels, dim, normmethod, modelstyle)] ) )

    MITgcm_loader = torch.utils.data.DataLoader(MITgcm_Dataset, batch_size=batch_size, shuffle=False,
                                                num_workers=1, pin_memory=True )

    no_samples = len(MITgcm_Dataset)
    X_size = MITgcm_ds['X'].shape[0]
    Y_size = MITgcm_ds['Y'].shape[0]
    Z_size = MITgcm_ds['Zmd000038'].shape[0]

    batch = 0

    if batch == 0:
       out_file = nc4.Dataset('../../../Channel_nn_Outputs/DATASETS/Dataset_'+model_name+'.nc','w', format='NETCDF4')
       # Create dimensions
       out_file.createDimension('samples', no_samples)
       out_file.createDimension('histlen', histlen)
       out_file.createDimension('phys_channels', no_phys_channels)
       out_file.createDimension('out_channels', no_out_channels)
       out_file.createDimension('X', X_size)
       out_file.createDimension('Y', Y_size)
       out_file.createDimension('Z', Z_size)
       # Create dimension variables
       nc_samples = out_file.createVariable( 'samples', 'i4', 'samples' )
       nc_histlen = out_file.createVariable( 'histlen', 'i4', 'histlen' )
       nc_phys_channels = out_file.createVariable( 'phys_channels', 'i4',  'phys_channels')
       nc_out_channels = out_file.createVariable( 'out_channels', 'i4', 'out_channels' )
       nc_X = out_file.createVariable( 'X', 'i4', 'X' )
       nc_Y = out_file.createVariable( 'Y', 'i4', 'Y' )
       nc_Z = out_file.createVariable( 'Z', 'i4', 'Z' )
       # Fill dimension variables
       nc_samples[:] = np.arange(no_samples)
       nc_histlen[:] = np.arange(histlen)
       nc_phys_channels[:] = np.arange(no_phys_channels)
       nc_out_channels[:] = np.arange(no_out_channels)
       nc_X[:] = MITgcm_ds['X'].values
       nc_Y[:] = MITgcm_ds['Y'].values
       nc_Z[:] = MITgcm_ds['Zmd000038'].values
       # Create variables
       if modelstyle == 'ConvLSTM' or modelstyle == 'UNetConvLSTM':
          nc_inputs    = out_file.createVariable( 'inputs',    'f4', ('samples', 'phys_channels', 'histlen', 'Y', 'X') )
          nc_targets   = out_file.createVariable( 'targets',   'f4', ('samples', 'out_channels', 'Y', 'X') )
          nc_masks     = out_file.createVariable( 'masks',     'f4', ('histlen', 'Z', 'Y', 'X') )
          nc_out_masks = out_file.createVariable( 'out_masks', 'f4', ('out_channels', 'Y', 'X') )
       elif dim == '2d':
          nc_inputs    = out_file.createVariable( 'inputs',    'f4', ('samples', 'phys_channels', 'Y', 'X') )
          nc_targets   = out_file.createVariable( 'targets',   'f4', ('samples', 'out_channels', 'Y', 'X') )
          nc_masks     = out_file.createVariable( 'masks',     'f4', ('Z', 'Y', 'X') )
          nc_out_masks = out_file.createVariable( 'out_masks', 'f4', ('out_channels', 'Y', 'X') )
       elif dim == '3d':
          nc_inputs    = out_file.createVariable( 'inputs',    'f4', ('samples', 'phys_channels', 'Z', 'Y', 'X') )
          nc_targets   = out_file.createVariable( 'targets',   'f4', ('samples', 'out_channels', 'Z', 'Y', 'X') )
          nc_masks     = out_file.createVariable( 'masks',     'f4', ('Z', 'Y', 'X') )
          nc_out_masks = out_file.createVariable( 'out_masks', 'f4', ('out_channels', 'Z', 'Y', 'X') )
       out_file.close()


    # Fill Dataset
    for input_batch, target_batch, masks, out_masks in MITgcm_loader:
       print('batch number : '+str(batch))
       out_file = nc4.Dataset('../../../Channel_nn_Outputs/DATASETS/Dataset_'+model_name+'.nc','r+', format='NETCDF4')
       nc_inputs[batch*batch_size:(batch+1)*batch_size] = input_batch
       nc_targets[batch*batch_size:(batch+1)*batch_size] = target_batch
       if batch == 0:
          nc_masks[:] = masks[0,]
          nc_out_masks[:] = out_masks[0,]
       batch = batch+1
       out_file.close()


class RF_Normalise_sample(object):

    def __init__(self, inputs_mean, inputs_std, inputs_range, targets_mean, targets_std, targets_range, 
                 histlen, predlen, no_phys_channels, dim, norm_method, model_style):
        self.inputs_mean   = inputs_mean
        self.inputs_std    = inputs_std
        self.inputs_range  = inputs_range
        self.targets_mean  = targets_mean
        self.targets_std   = targets_std
        self.targets_range = targets_range
        self.histlen       = histlen
        self.predlen       = predlen
        self.no_phys_channels = no_phys_channels
        self.dim           = dim
        self.norm_method   = norm_method
        self.model_style   = model_style

    def __call__(self, sample):

        # Using transforms.Normalize returns the function, rather than the normalised array - can't figure out how to avoid this...
        #sample_input  = transforms.Normalize(sample_input, self.inputs_inputs_mean, self.inputs_inputs_std)
        #sample_output = transforms.Normalize(sample_output,self.outputs_inputs_mean, self.outputs_inputs_std)

        # only normalise for channels with physical variables (= no_out channels*histlen)..
        # the input channels also have mask channels and lat info, but we don't want to normalise these. 

        if self.model_style == 'ConvLSTM' or self.model_style == 'UNetConvLSTM':
           for time in range(self.histlen):
              sample['input'][time,:self.no_phys_channels, :, :] =                   \
                                   RF_Normalise( sample['input'][time,:self.no_phys_channels, :, :],
                                                 self.inputs_mean, self.inputs_std, self.inputs_range, self.dim, self.norm_method )
           for time in range(self.predlen):
              sample['output'][time*self.no_phys_channels:(time+1)*self.no_phys_channels, :, :] =                   \
                                   RF_Normalise( sample['output'][time*self.no_phys_channels:(time+1)*self.no_phys_channels, :, :],
                                                 self.targets_mean, self.targets_mean, self.targets_range, self.dim, self.norm_method )
        elif self.dim == '2d':
           for time in range(self.histlen):
              sample['input'][time*self.no_phys_channels:(time+1)*self.no_phys_channels, :, :] =                   \
                                   RF_Normalise( sample['input'][time*self.no_phys_channels:(time+1)*self.no_phys_channels, :, :],
                                                 self.inputs_mean, self.inputs_std, self.inputs_range, self.dim, self.norm_method )
           for time in range(self.predlen):
              sample['output'][time*self.no_phys_channels:(time+1)*self.no_phys_channels, :, :] =                   \
                                   RF_Normalise( sample['output'][time*self.no_phys_channels:(time+1)*self.no_phys_channels, :, :],
                                                 self.targets_mean, self.targets_mean, self.targets_range, self.dim, self.norm_method )
        elif self.dim == '3d':
           for time in range(self.histlen):
              sample['input'][time*self.no_phys_channels:(time+1)*self.no_phys_channels, :, :, :] =                   \
                                   RF_Normalise( sample['input'][time*self.no_phys_channels:(time+1)*self.no_phys_channels, :, :, :],
                                                 self.inputs_mean, self.inputs_std, self.inputs_range,self.dim, self.norm_method )
           for time in range(self.predlen):
              sample['output'][time*self.no_phys_channels:(time+1)*self.no_phys_channels, :, :, :] =                   \
                                   RF_Normalise( sample['output'][time*self.no_phys_channels:(time+1)*self.no_phys_channels, :, :, :],
                                                 self.targets_mean, self.targets_mean, self.targets_range, self.dim, self.norm_method )


        return sample['input'], sample['output']

def RF_Normalise(data, d_mean, d_std, d_range, dim, norm_method ):
    """Normalise data based on training means and std (given)

       ReNormalises prediction for use in iteration
       dims of data: channels(, z), y, x

    Args:
       TBD
    """

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

def RF_DeNormalise(samples, data_mean, data_std, data_range, dim, norm_method):
    """de-Normalise data based on training means and std (given)

    Args:
       Array to de-norm (expected shape - no_samples, no_channels,( z,) y, x)
       Training output mean
       Training output std
    """

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

