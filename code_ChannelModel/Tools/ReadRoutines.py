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


# Create Dataset, which inherits the data.Dataset class
class MITGCM_Wholefield_Dataset(data.Dataset):

   '''
         MITGCM dataset
        
         Note data not normalised here - needs to be done in network training routine

         Note currently code is set up so that the data is 2-d with different channels for each level of 
         each variable. This solves issues with eta being 2-d and other inputs being 3-d. This matches 
         what is done in Scher paper.
         The alternative is to have 3-d fields, with each channel being separate variables. Potentially 
         worth assessing impact of this at some stage
   '''

   def __init__(self, MITGCM_filename, start_ratio, end_ratio, stride, dataset_end_index, transform=None):
       """
       Args:
          MITGCM_filename (string): Path to the MITGCM filename containing the data.
          start_ratio (float)     : Ratio point (between 0 and 1) at which to start sampling from MITGCM data
          end_ratio (float)       : Ratio point (between 0 and 1) at which to stop sampling from MITGCM data
          stride (integer)        : Rate at which to subsample in time when reading MITGCM data
          dataset_end_index       : Point of dataset file after which to ignore (i.e. cause model is no longer
                                    dynamically active
       """
 
       self.ds      = xr.open_dataset(MITGCM_filename)
       self.start   = int(start_ratio * dataset_end_index)
       self.end     = int(end_ratio * dataset_end_index)
       self.stride  = stride
       self.transform = transform
       
       self.ds_inputs  = self.ds.isel(T=slice(self.start, self.end, self.stride))
       self.ds_outputs = self.ds.isel(T=slice(self.start+1, self.end+1, self.stride))
  
   def __len__(self):
       return self.ds_outputs.sizes['T']

   def __getitem__(self, idx):

       da_T_in_tmp = self.ds_inputs['THETA'].values[idx,:,:,:] 
       da_T_in     = 0.5 * (da_T_in_tmp[:,:-1,:]+da_T_in_tmp[:,1:,:]) # average to get onto same grid as V points  
       da_U_in_tmp = self.ds_inputs['UVEL'].values[idx,:,:,:]
       da_U_in_tmp = 0.5 * (da_U_in_tmp[:,:,:-1]+da_U_in_tmp[:,:,1:]) # average x dir onto same grid as T points  
       da_U_in     = 0.5 * (da_U_in_tmp[:,:-1,:]+da_U_in_tmp[:,1:,:]) # average y dir onto same grid as V points  
       da_V_in     = self.ds_inputs['VVEL'].values[idx,:,1:-1,:]      # Ignore first and last points as both zero
       da_Eta_in_tmp = self.ds_inputs['ETAN'].values[idx,0,:,:]
       da_Eta_in   = 0.5 * (da_Eta_in_tmp[:-1,:]+da_Eta_in_tmp[1:,:]) # average to get onto same grid as V points  

       da_T_out_tmp = self.ds_outputs['THETA'].values[idx,:,:,:]
       da_T_out     = 0.5 * (da_T_out_tmp[:,:-1,:]+da_T_out_tmp[:,1:,:]) # average to get onto same grid as V points  
       da_U_out_tmp = self.ds_outputs['UVEL'].values[idx,:,:,:]
       da_U_out_tmp = 0.5 * (da_U_out_tmp[:,:,:-1]+da_U_out_tmp[:,:,1:])  # average to get onto same grid as T points
       da_U_out     = 0.5 * (da_U_out_tmp[:,:-1,:]+da_U_out_tmp[:,1:,:])  # average y dir onto same grid as V points  
       da_V_out     = self.ds_outputs['VVEL'].values[idx,:,1:-1,:]        # Ignore first and last points as both zero
       da_Eta_out_tmp = self.ds_outputs['ETAN'].values[idx,0,:,:]
       da_Eta_out   = 0.5 * (da_Eta_out_tmp[:-1,:]+da_Eta_out_tmp[1:,:]) # average to get onto same grid as V points  

       if np.isnan(da_T_in).any():
          print('da_T_in contains a NaN')
          print('nans at '+str(np.argwhere(np.isnan(da_T_in))) )
       #if np.isnan(da_S_in).any():
       #   print('da_S_in contains a NaN')
       #   print('nans at '+str(np.argwhere(np.isnan(da_S_in))) )
       if np.isnan(da_U_in).any():
          print('da_U_in contains a NaN')
          print('nans at '+str(np.argwhere(np.isnan(da_U_in))) )
       if np.isnan(da_V_in).any():
          print('da_V_in contains a NaN')
          print('nans at '+str(np.argwhere(np.isnan(da_V_in))) )
       if np.isnan(da_Eta_in).any():
          print('da_Eta_in contains a NaN')
          print('nans at '+str(np.argwhere(np.isnan(da_Kwz_in))) )

       if np.isnan(da_T_out).any():
          print('da_T_out contains a NaN')
          print('nans at '+str(np.argwhere(np.isnan(da_T_out))) )
       #if np.isnan(da_S_out).any():
       #   print('da_S_out contains a NaN')
       #   print('nans at '+str(np.argwhere(np.isnan(da_S_out))) )
       if np.isnan(da_U_in).any():
          print('da_U_in contains a NaN')
          print('nans at '+str(np.argwhere(np.isnan(da_U_in))) )
       if np.isnan(da_V_in).any():
          print('da_V_in contains a NaN')
          print('nans at '+str(np.argwhere(np.isnan(da_V_in))) )
       if np.isnan(da_Eta_out).any():
          print('da_Eta_out contains a NaN')
          print('nans at '+str(np.argwhere(np.isnan(da_Kwz_out))) )

       #if np.isnan(mask).any():
       #   print('mask contains a NaN')
       #   print('nans at '+str(np.argwhere(np.isnan(mask))) )

       sample_input  = np.zeros(( 0, da_T_in.shape[1],  da_T_in.shape[2]  ))  # shape: (no channels, y, x)
       sample_output = np.zeros(( 0, da_T_out.shape[1], da_T_out.shape[2]  ))  # shape: (no channels, y, x)

       for level in range(da_T_in.shape[0]):
           sample_input  = np.concatenate((sample_input ,
                                            da_T_in[level, :, :].reshape(1,da_T_in.shape[1],da_T_in.shape[2])),axis=0)              
           sample_output = np.concatenate((sample_output, 
                                           da_T_out[level, :, :].reshape(1,da_T_in.shape[1],da_T_in.shape[2])),axis=0)              

       #for level in range(da_S_in.shape[0]):
       #    sample_input  = np.concatenate((sample_input ,
       #                                     da_S_in[level, :, :].reshape(1,da_S_in.shape[1],da_S_in.shape[2])),axis=0)              
       #    sample_output = np.concatenate((sample_output, 
       #                                    da_S_out[level, :, :].reshape(1,da_S_in.shape[1],da_S_in.shape[2])),axis=0)              

       for level in range(da_U_in.shape[0]):
           sample_input  = np.concatenate((sample_input ,
                                            da_U_in[level, :, :].reshape(1,da_T_in.shape[1],da_T_in.shape[2])),axis=0)              
           sample_output = np.concatenate((sample_output, 
                                           da_U_out[level, :, :].reshape(1,da_T_in.shape[1],da_T_in.shape[2])),axis=0)              

       for level in range(da_V_in.shape[0]):
           sample_input  = np.concatenate((sample_input ,
                                            da_V_in[level, :, :].reshape(1,da_T_in.shape[1],da_T_in.shape[2])),axis=0)              
           sample_output = np.concatenate((sample_output, 
                                           da_V_out[level, :, :].reshape(1,da_T_in.shape[1],da_T_in.shape[2])),axis=0)              

       sample_input  = np.concatenate((sample_input , 
                                        da_Eta_in[:, :].reshape(1,da_T_in.shape[1],da_T_in.shape[2])),axis=0)              
       sample_output = np.concatenate((sample_output, 
                                       da_Eta_out[:, :].reshape(1,da_T_in.shape[1],da_T_in.shape[2])),axis=0)              

       #for level in range(mask.shape[0]):
       #    sample_input  = np.concatenate((sample_input , mask[level, :, :].reshape(1,da_T_in.shape[1],da_T_in.shape[2])),axis=0) 
       #    sample_output = np.concatenate((sample_output, mask[level, :, :].reshape(1,da_T_in.shape[1],da_T_in.shape[2])),axis=0) 
 
       sample_input = torch.from_numpy(sample_input)
       sample_output = torch.from_numpy(sample_output)
 
       sample = {'input':sample_input, 'output':sample_output}
       
       if self.transform:
          sample = self.transform(sample)
 
       if np.isnan(sample['input']).any():
          print('sample_input contains a NaN')
       if np.isnan(sample['output']).any():
          print('sample_output contains a NaN')
 
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

        for channel in range(sample_input.shape[0]):
           #if not (self.train_std[channel] == 0.0):
           #   sample_input[channel, :, :]  = (sample_input[channel, :, :] - self.train_mean[channel]) / self.train_std[channel]
           #   sample_output[channel, :, :] = (sample_output[channel, :, :] - self.train_mean[channel]) / self.train_std[channel]
           sample_input[channel, :, :]  = (sample_input[channel, :, :] - self.train_mean[channel]) / self.train_range[channel]
           sample_output[channel, :, :] = (sample_output[channel, :, :] - self.train_mean[channel]) / self.train_range[channel]

        return {'input':sample_input, 'output':sample_output}

def RF_DeNormalise(samples, train_mean, train_std, train_range):
    """de-Normalise data based on training means and std (given)

    Args:
       Array to de-norm (expected shape - no_samples, no_channels, y, x)
       Training output mean
       Training output std
    """

    for channel in range(samples.shape[1]):
       #if not (train_std[channel] == 0.0):
       #   samples[:,channel, :, :]  = (samples[:,channel, :, :] * train_std[channel] ) + train_mean[channel]
       samples[:,channel, :, :]  = (samples[:,channel, :, :] * train_range[channel] ) + train_mean[channel]

    return samples

