#!/usr/bin/env python
# coding: utf-8

# Code to train a network to learn the dynamic evolution of an MITgcm channel configuration
# Designed to take the entire field for all variables, apply a NN (with conv layers), and output
# the entire field for all variables one day later (i.e. the next iteration of the MITgcm netcdf 
# file here, although note this is 2 steps of the underlying model which runs a 12 hourly timestep.
# The data is subsampled in time to give quasi-independence

#from comet_ml import Experiment

import sys
sys.path.append('../Tools')
import ReadRoutines as rr
import Channel_Model_Plotting as ChnPlt

import numpy as np
import os
import glob
import xarray as xr
import pickle

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from torch.utils import data
import torch.nn as nn
from torch.autograd import Variable
import torch
from torchvision import transforms, utils

import netCDF4 as nc4

import time as time
import gc as gc

import logging

import multiprocessing as mp

from scipy.ndimage import gaussian_filter

import wandb
import gcm_filters as gcm_filters

def ReadMeanStd(dim, no_levels, pred_jump):
   mean_std_file = '../../../Channel_nn_Outputs/DATASETS/'+pred_jump+'_MeanStd.npz'
   mean_std_data = np.load(mean_std_file)
   inputs_mean  = mean_std_data['arr_0']
   inputs_std   = mean_std_data['arr_1']
   inputs_range = mean_std_data['arr_2']
   targets_mean  = mean_std_data['arr_3']
   targets_std   = mean_std_data['arr_4']
   targets_range = mean_std_data['arr_5']

   # Expand arrays to multiple levels for 2d approach
   if dim =='2d':
      tmp_inputs_mean  = np.zeros((3*no_levels+1))
      tmp_inputs_std   = np.zeros((3*no_levels+1))
      tmp_inputs_range = np.zeros((3*no_levels+1))
      tmp_targets_mean  = np.zeros((3*no_levels+1))
      tmp_targets_std   = np.zeros((3*no_levels+1))
      tmp_targets_range = np.zeros((3*no_levels+1))

      for var in range(3):
         tmp_inputs_mean[var*no_levels:(var+1)*no_levels] = inputs_mean[var]
         tmp_inputs_std[var*no_levels:(var+1)*no_levels] = inputs_std[var]
         tmp_inputs_range[var*no_levels:(var+1)*no_levels] = inputs_range[var]
         tmp_targets_mean[var*no_levels:(var+1)*no_levels] = targets_mean[var]
         tmp_targets_std[var*no_levels:(var+1)*no_levels] = targets_std[var]
         tmp_targets_range[var*no_levels:(var+1)*no_levels] = targets_range[var]
      tmp_inputs_mean[3*no_levels] = inputs_mean[3]
      tmp_inputs_std[3*no_levels] = inputs_std[3]
      tmp_inputs_range[3*no_levels] = inputs_range[3]
      tmp_targets_mean[3*no_levels] = targets_mean[3]
      tmp_targets_std[3*no_levels] = targets_std[3]
      tmp_targets_range[3*no_levels] = targets_range[3]

      inputs_mean  = tmp_inputs_mean
      inputs_std   = tmp_inputs_std 
      inputs_range = tmp_inputs_range
      targets_mean  = tmp_targets_mean
      targets_std   = tmp_targets_std 
      targets_range = tmp_targets_range

   return(inputs_mean, inputs_std, inputs_range, targets_mean, targets_std, targets_range)

def my_loss(output, target):
   loss = torch.mean( (output - target)**2 )
   return loss

def TrainModel(model_name, model_style, dimension, histlen, predlen, tic, TEST, no_tr_samples, no_val_samples, save_freq, train_loader, val_loader,
               h, optimizer, num_epochs, seed_value, losses, channel_dim, no_phys_channels, start_epoch=1, current_best_loss=0.1):

   device = 'cuda' if torch.cuda.is_available() else 'cpu'

   no_depth_levels = 38  # Hard coded...perhaps should change...?
   # Set variables to remove randomness and ensure reproducible results
   os.environ['PYTHONHASHSEED'] = str(seed_value)
   np.random.seed(seed_value)
   torch.manual_seed(seed_value)
   torch.cuda.manual_seed(seed_value)
   torch.cuda.manual_seed_all(seed_value)
   torch.backends.cudnn.enabled = False
   torch.backends.cudnn.deterministic = True
   torch.backends.cudnn.benchmark = False

   logging.info('###### TRAINING MODEL ######')
   for epoch in range( start_epoch, num_epochs+start_epoch ):

       # Training
       losses['train'].append(0.)
       losses['train_Temp'].append(0.)
       losses['train_U'].append(0.)
       losses['train_V'].append(0.)
       losses['train_Eta'].append(0.)
       losses['val'].append(0.)

       batch_no = 0

       h.train(True)

       #for input_batch, target_batch, masks, bdy_masks in train_loader:
       for input_batch, target_batch, masks, out_masks in train_loader:

           input_batch = input_batch.to(device, non_blocking=True, dtype=torch.float)
           masks = masks.to(device, non_blocking=True, dtype=torch.float)
           out_masks = out_masks.to(device, non_blocking=True, dtype=torch.float)
           target_batch = target_batch.to(device, non_blocking=True, dtype=torch.float)

           # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
           optimizer.zero_grad()
       
           # get prediction from the model, given the inputs
           # multiply by masks, to give 0 at land values, to circumvent issues with normalisation!
           if predlen == 1:
              predicted_batch = h( torch.cat((input_batch, masks), dim=channel_dim) ) * out_masks
           else:
              if dimension != '2d' and model_style != 'UNet2dTransp':
                 sys.exit('Cannot currently use predlen>1 with options other than 2d tansp network')
              predicted_batch = torch.zeros(target_batch.shape).to(device, non_blocking=True, dtype=torch.float)
              # If predlen>1 calc iteratively for 'roll out' loss function
              for time in range(predlen):
                 predicted_batch[:,time*no_phys_channels:(time+1)*no_phys_channels,:,:] = h( torch.cat((input_batch, masks), dim=channel_dim) ) * out_masks
                 for hist_time in range(histlen-1):
                    input_batch[:, hist_time*no_phys_channels:(hist_time+1)*no_phys_channels, :, :] =   \
                              input_batch[:, (hist_time+1)*no_phys_channels:(hist_time+2)*no_phys_channels, :, :]
                 input_batch[:, (histlen-1)*no_phys_channels:, :, :] =  (input_batch[:,(histlen-1)*no_phys_channels:histlen*no_phys_channels,:,:] +
                                                                         predicted_batch[:,time*no_phys_channels:(time+1)*no_phys_channels,:,:])
	      
           logging.debug('For test run check if output is correct shape, the following two shapes should match!\n')
           logging.debug('target_batch.shape for training data: '+str(target_batch.shape)+'\n')
           logging.debug('predicted_batch.shape for training data: '+str(predicted_batch.shape)+'\n')
           logging.debug('masks.shape : '+str(masks.shape)+'\n')
    
           # Calculate and update loss values
           
           loss = my_loss(predicted_batch, target_batch).double()
           losses['train'][-1] = losses['train'][-1] + loss.item() * input_batch.shape[0]

           if model_style == 'ConvLSTM' or model_style == 'UNetConvLSTM':
              if dimension == '2d': 
                 losses['train_Temp'][-1] = losses['train_Temp'][-1] + \
                                            my_loss( predicted_batch[:,:no_depth_levels,:,:],
						    target_batch[:,:no_depth_levels,:,:],
                                            ).item() * input_batch.shape[0]
                 losses['train_U'][-1] = losses['train_U'][-1] + \
                                         my_loss( predicted_batch[:,no_depth_levels:2*no_depth_levels,:,:],
                                         target_batch[:,no_depth_levels:2*no_depth_levels,:,:],
                                         ).item() * input_batch.shape[0]
                 losses['train_V'][-1] = losses['train_V'][-1] + \
                                         my_loss( predicted_batch[:,2*no_depth_levels:3*no_depth_levels,:,:],
                                         target_batch[:,2*no_depth_levels:3*no_depth_levels,:,:],
                                         ).item() * input_batch.shape[0]
                 losses['train_Eta'][-1] = losses['train_Eta'][-1] + \
                                           my_loss( predicted_batch[:,3*no_depth_levels,:,:],
                                           target_batch[:,3*no_depth_levels,:,:],
                                           ).item() * input_batch.shape[0]
           else:  
              if dimension == '2d': 
                 losses['train_Temp'][-1] = losses['train_Temp'][-1] + \
                                            my_loss( predicted_batch[:,:no_depth_levels,:,:],
                                            target_batch[:,:no_depth_levels,:,:],
                                            ).item() * input_batch.shape[0]
                 losses['train_U'][-1] = losses['train_U'][-1] + \
                                         my_loss( predicted_batch[:,no_depth_levels:2*no_depth_levels,:,:],
                                         target_batch[:,no_depth_levels:2*no_depth_levels,:,:],
                                         ).item() * input_batch.shape[0]
                 losses['train_V'][-1] = losses['train_V'][-1] + \
                                         my_loss( predicted_batch[:,2*no_depth_levels:3*no_depth_levels,:,:],
                                         target_batch[:,2*no_depth_levels:3*no_depth_levels,:,:],
                                         ).item() * input_batch.shape[0]
                 losses['train_Eta'][-1] = losses['train_Eta'][-1] + \
                                           my_loss( predicted_batch[:,3*no_depth_levels,:,:],
                                           target_batch[:,3*no_depth_levels,:,:],
                                           ).item() * input_batch.shape[0]
              elif dimension == '3d': 
                 losses['train_Temp'][-1] = losses['train_Temp'][-1] + \
                                            my_loss( predicted_batch[:,0,:,:,:], target_batch[:,0,:,:,:] ).item() * input_batch.shape[0]
                 losses['train_U'][-1]    = losses['train_U'][-1] + \
                                            my_loss( predicted_batch[:,1,:,:,:], target_batch[:,1,:,:,:] ).item() * input_batch.shape[0]
                 losses['train_V'][-1]    = losses['train_V'][-1] + \
                                            my_loss( predicted_batch[:,2,:,:,:], target_batch[:,2,:,:,:] ).item() * input_batch.shape[0]
                 losses['train_Eta'][-1]  = losses['train_Eta'][-1] + \
                                            my_loss( predicted_batch[:,3,:,:,:], target_batch[:,3,:,:,:] ).item() * input_batch.shape[0]
  
           # get gradients w.r.t to parameters
           loss.backward()
        
           # update parameters
           optimizer.step()
   
           del input_batch
           del target_batch
           del predicted_batch
           gc.collect()
           torch.cuda.empty_cache()

           batch_no = batch_no + 1

       wandb.log({"train loss" : losses['train'][-1]/no_tr_samples})
       losses['train'][-1] = losses['train'][-1] / no_tr_samples
       losses['train_Temp'][-1] = losses['train_Temp'][-1] / no_tr_samples
       losses['train_U'][-1] = losses['train_U'][-1] / no_tr_samples
       losses['train_V'][-1] = losses['train_V'][-1] / no_tr_samples
       losses['train_Eta'][-1] = losses['train_Eta'][-1] / no_tr_samples
  
       logging.info('epoch {}, training loss {}'.format(epoch, losses['train'][-1])+'\n')

       #### Validation ######
   
       batch_no = 0
       
       h.train(False)

       #for input_batch, target_batch, masks, bdy_masks in val_loader:
       for input_batch, target_batch, masks, out_masks in val_loader:
   
           # Send inputs and labels to GPU if available
           input_batch = input_batch.to(device, non_blocking=True, dtype=torch.float)
           masks = masks.to(device, non_blocking=True, dtype=torch.float)
           out_masks = out_masks.to(device, non_blocking=True, dtype=torch.float)
           target_batch = target_batch.to(device, non_blocking=True, dtype=torch.float)

           # get prediction from the model, given the inputs
           if predlen == 1:
              predicted_batch = h( torch.cat((input_batch, masks), dim=channel_dim) ) * out_masks
           else:
              if dimension != '2d' and model_style != 'UNet2dTransp':
                 sys.exit('Cannot currently use predlen>1 with options other than 2d tansp network')
              predicted_batch = torch.zeros(target_batch.shape).to(device, non_blocking=True, dtype=torch.float)
              for time in range(predlen):
                 predicted_batch[:,time*no_phys_channels:(time+1)*no_phys_channels,:,:] = h( torch.cat((input_batch, masks), dim=channel_dim) ) * out_masks
                 for hist_time in range(histlen-1):
                    input_batch[:, hist_time*no_phys_channels:(hist_time+1)*no_phys_channels, :, :] =   \
                              input_batch[:, (hist_time+1)*no_phys_channels:(hist_time+2)*no_phys_channels, :, :]
                 input_batch[:, (histlen-1)*no_phys_channels:, :, :] =  (input_batch[:,(histlen-1)*no_phys_channels:histlen*no_phys_channels,:,:] +
                                                                         predicted_batch[:,time*no_phys_channels:(time+1)*no_phys_channels,:,:])
       
           # get loss for the predicted_batch output
           losses['val'][-1] = losses['val'][-1] +  \
                               my_loss( predicted_batch, target_batch ).item() * input_batch.shape[0]
   
           del input_batch
           del target_batch
           del predicted_batch
           gc.collect()
           torch.cuda.empty_cache()

           batch_no = batch_no + 1
   
       wandb.log({"val loss" : losses['val'][-1]/no_tr_samples})
       losses['val'][-1] = losses['val'][-1] / no_val_samples
       logging.info('epoch {}, validation loss {}'.format(epoch, losses['val'][-1])+'\n')
  
       # Save model if this is the best version of the model 
       if losses['val'][-1] < current_best_loss :
           for model_file in glob.glob('../../../Channel_nn_Outputs/'+model_name+'/MODELS/'+model_name+'_epoch*_SavedBESTModel.pt'):
              os.remove(model_file)
           pkl_filename = '../../../Channel_nn_Outputs/'+model_name+'/MODELS/'+model_name+'_epoch'+str(epoch)+'_SavedBESTModel.pt'
           torch.save({
                       'epoch': epoch,
                       'model_state_dict': h.state_dict(),
                       'optimizer_state_dict': optimizer.state_dict(),
                       'loss': loss,
                       'losses': losses,
                       'best_loss': current_best_loss,
                       }, pkl_filename)
           current_best_loss = losses['val'][-1]

       # Save model if its been a while 
       if ( epoch%save_freq == 0 or epoch == num_epochs+start_epoch-1 ) and epoch != start_epoch :
           pkl_filename = '../../../Channel_nn_Outputs/'+model_name+'/MODELS/'+model_name+'_epoch'+str(epoch)+'_SavedModel.pt'
           torch.save({
                       'epoch': epoch,
                       'model_state_dict': h.state_dict(),
                       'optimizer_state_dict': optimizer.state_dict(),
                       'loss': loss,
                       'losses': losses,
                       'best_loss': current_best_loss,
                       }, pkl_filename)
  
       #TimeCheck(tic, 'Epoch '+str(epoch))

   logging.info('End of train model')
   return(losses)

def PlotScatter(model_name, dimension, data_loader, h, epoch, title, MeanStd_prefix, norm_method, channel_dim, pred_jump, no_phys_channels):

   logging.info('Making scatter plots')
   plt.rcParams.update({'font.size': 14})

   no_depth_levels = 38  # Hard coded...perhaps should change...?

   inputs_mean, inputs_std, inputs_range, targets_mean, targets_std, targets_range = ReadMeanStd(dimension, no_depth_levels, pred_jump)

   device = 'cuda' if torch.cuda.is_available() else 'cpu'

   h.train(False)

   with torch.no_grad():
      #input_batch, target_batch, masks, bdy_masks = next(iter(data_loader))
      input_batch, target_batch, masks, out_masks = next(iter(data_loader))
      input_batch = input_batch.to(device, non_blocking=True, dtype=torch.float)
      target_batch = target_batch.double()
      masks  = masks.to(device, non_blocking=True, dtype=torch.float)
      predicted_batch = h( torch.cat( (input_batch, masks), dim=channel_dim )
                         ).cpu().detach().double()

   # Denormalise and mask, if predlen>1 (loss function calc over multiple steps) only take first set of predictions
   target_batch = torch.where(out_masks==1, rr.RF_DeNormalise(target_batch[:, :no_phys_channels, :, :],
                                                              targets_mean, targets_std, targets_range, dimension, norm_method), np.nan)
   predicted_batch = torch.where(out_masks==1, rr.RF_DeNormalise(predicted_batch, targets_mean, targets_std, targets_range, dimension, norm_method), np.nan)

   if dimension == '2d':
      # reshape to give each variable its own dimension
      targets = np.zeros((predicted_batch.shape[0], 4, no_depth_levels, target_batch.shape[2], target_batch.shape[3] ))
      predictions = np.zeros((predicted_batch.shape[0], 4, no_depth_levels, target_batch.shape[2], target_batch.shape[3] ))
      for i in range(3):
         targets[:,i,:,:,:]  = target_batch[:,i*no_depth_levels:(i+1)*no_depth_levels,:,:] 
         predictions[:,i,:,:,:] = predicted_batch[:,i*no_depth_levels:(i+1)*no_depth_levels,:,:] 
      targets[:,3,:,:,:] = target_batch[:,3*no_depth_levels:4*no_depth_levels,:,:]
      predictions[:,3,:,:,:] = predicted_batch[:,3*no_depth_levels:4*no_depth_levels,:,:]

   elif dimension == '3d':
      targets = target_batch[:,:,:,:,:].cpu().detach().numpy()
      predictions = predicted_batch[:,:,:,:,:].cpu().detach().numpy()

   fig_main = plt.figure(figsize=(9,9))
   ax_temp  = fig_main.add_subplot(221)
   ax_U     = fig_main.add_subplot(223)
   ax_V     = fig_main.add_subplot(224)
   ax_Eta   = fig_main.add_subplot(222)

   ax_temp.scatter(targets[:,0,:,:,:],
                   predictions[:,0,:,:,:],
                   edgecolors=(0, 0, 0), alpha=0.15, color='blue', label='Temperature')
   ax_U.scatter(targets[:,1,:,:,:],
                predictions[:,1,:,:,:],
                edgecolors=(0, 0, 0), alpha=0.15, color='red', label='East-West Velocity')
   ax_V.scatter(targets[:,2,:,:,:], 
                predictions[:,2,:,:,:],
                edgecolors=(0, 0, 0), alpha=0.15, color='orange', label='North-South Velocity')
   ax_Eta.scatter(targets[:,3,0,:,:], 
                  predictions[:,3,0,:,:],
                  edgecolors=(0, 0, 0), alpha=0.15, color='purple', label='Sea Surface Height')
   bottom = min( np.nanmin(targets), np.nanmin(predictions) )
   top    = max( np.nanmax(targets), np.nanmax(predictions) )  
   
   ax_temp.set_xlabel('Truth')
   ax_temp.set_ylabel('Predictions')
   ax_temp.set_title('Temperature')
   ax_temp.set_xlim( min( np.nanmin(targets[:,0,:,:,:]), np.nanmin(predictions[:,0,:,:,:]) ),
                     max( np.nanmax(targets[:,0,:,:,:]), np.nanmax(predictions[:,0,:,:,:]) ) )
   ax_temp.set_ylim( min( np.nanmin(targets[:,0,:,:,:]), np.nanmin(predictions[:,0,:,:,:]) ),
                     max( np.nanmax(targets[:,0,:,:,:]), np.nanmax(predictions[:,0,:,:,:]) ) )
   ax_temp.plot([bottom, top], [bottom, top], 'k--', lw=1, color='black')

   ax_U.set_xlabel('Truth')
   ax_U.set_ylabel('Predictions')
   ax_U.set_title('U velocity')
   ax_U.set_xlim( min( np.nanmin(targets[:,1,:,:,:]), np.nanmin(predictions[:,1,:,:,:]) ),
                  max( np.nanmax(targets[:,1,:,:,:]), np.nanmax(predictions[:,1,:,:,:]) ) )
   ax_U.set_ylim( min( np.nanmin(targets[:,1,:,:,:]), np.nanmin(predictions[:,1,:,:,:]) ),
                  max( np.nanmax(targets[:,1,:,:,:]), np.nanmax(predictions[:,1,:,:,:]) ) )
   ax_U.plot([bottom, top], [bottom, top], 'k--', lw=1, color='black')

   ax_V.set_xlabel('Truth')
   ax_V.set_ylabel('Predictions')
   ax_V.set_title('V velocity')
   ax_V.set_xlim( min( np.nanmin(targets[:,2,:,:,:]), np.nanmin(predictions[:,2,:,:,:]) ),
                  max( np.nanmax(targets[:,2,:,:,:]), np.nanmax(predictions[:,2,:,:,:]) ) )
   ax_V.set_ylim( min( np.nanmin(targets[:,2,:,:,:]), np.nanmin(predictions[:,2,:,:,:]) ),
                  max( np.nanmax(targets[:,2,:,:,:]), np.nanmax(predictions[:,2,:,:,:]) ) )
   ax_V.plot([bottom, top], [bottom, top], 'k--', lw=1, color='black')

   ax_Eta.set_xlabel('Truth')
   ax_Eta.set_ylabel('Predictions')
   ax_Eta.set_title('Sea Surface Height')
   ax_Eta.set_xlim( min( np.nanmin(targets[:,3,:,:,:]), np.nanmin(predictions[:,3,:,:,:]) ),
                    max( np.nanmax(targets[:,3,:,:,:]), np.nanmax(predictions[:,3,:,:,:]) ) )
   ax_Eta.set_ylim( min( np.nanmin(targets[:,3,:,:,:]), np.nanmin(predictions[:,3,:,:,:]) ),
                    max( np.nanmax(targets[:,3,:,:,:]), np.nanmax(predictions[:,3,:,:,:]) ) )
   ax_Eta.plot([bottom, top], [bottom, top], 'k--', lw=1, color='black')

   plt.tight_layout()

   plt.savefig('../../../Channel_nn_Outputs/'+model_name+'/PLOTS/'+model_name+'_scatter_epoch'+str(epoch).rjust(3,'0')+'_'+title+'.png',
               bbox_inches = 'tight', pad_inches = 0.1)
   plt.close()


def LoadModel(model_name, h, optimizer, saved_epoch, tr_inf, losses, best):
   if best:
      pkl_filename = '../../../Channel_nn_Outputs/'+model_name+'/MODELS/'+model_name+'_epoch'+str(saved_epoch)+'_SavedBESTModel.pt'
   else:
      pkl_filename = '../../../Channel_nn_Outputs/'+model_name+'/MODELS/'+model_name+'_epoch'+str(saved_epoch)+'_SavedModel.pt'
   print(pkl_filename)
   checkpoint = torch.load(pkl_filename)
   h.load_state_dict(checkpoint['model_state_dict'])
   optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
   epoch = checkpoint['epoch']
   loss = checkpoint['loss']
   losses = checkpoint['losses']
   current_best_loss = checkpoint['best_loss']

   if tr_inf == 'tr':
      h.train()
   elif tr_inf == 'inf':
      h.eval()

   return losses, h, optimizer, current_best_loss

def plot_training_output(model_name, start_epoch, total_epochs, plot_freq, losses):
   plot_dir = '../../../Channel_nn_Outputs/'+model_name+'/PLOTS/'
   if not os.path.isdir(plot_dir):
      os.system("mkdir %s" % (plot_dir))
 
   ## Plot training loss over time
   fig = plt.figure()
   ax1 = fig.add_subplot(111)
   ax1.plot(range(0, len(losses['train'])), losses['train'],       color='black',  label='Training Loss')
   ax1.plot(range(0, len(losses['train'])), losses['train_Temp'],  color='blue',   label='Temperature Training Loss')
   ax1.plot(range(0, len(losses['train'])), losses['train_U'],     color='red',    label='U Training Loss')
   ax1.plot(range(0, len(losses['train'])), losses['train_V'],     color='orange', label='V Training Loss')
   ax1.plot(range(0, len(losses['train'])), losses['train_Eta'],   color='purple', label='Eta Training Loss')
   ax1.plot(range(0, len(losses['train'])), losses['val'],         color='grey',   label='Validation Loss')
   ax1.set_xlabel('Epochs')
   ax1.set_ylabel('Loss')
   ax1.set_yscale('log')
   ax1.legend()
   plt.savefig(plot_dir+'/'+model_name+'_TrainingValLossPerEpoch_epoch'+str(total_epochs)+'.png',
               bbox_inches = 'tight', pad_inches = 0.1)
   plt.close()
 
def OutputStats(model_name, model_style, MeanStd_prefix, MITgcm_filename, data_loader, h, no_epochs, y_dim_used, dimension, 
                histlen, land, file_append, norm_method, channel_dim, pred_jump, no_phys_channels, MITgcm_stats_filename):
   #-------------------------------------------------------------
   # Output the rms and correlation coefficients over a dataset
   #-------------------------------------------------------------
   #  1. Make predictions over entire dataset (training, val or test), summing Mean Squared Error 
   #     as we go
   #  2. Store predictions and targets for a small number of smaples in an array
   #  3. Calculate the RMS of the dataset from the summed squared error
   #  4. Store the RMS and subset of samples in a netcdf file
   logging.info('Outputting stats')

   device = 'cuda' if torch.cuda.is_available() else 'cpu'

   # Read in grid data from MITgcm file
   MITgcm_ds = xr.open_dataset(MITgcm_filename)
   da_X = MITgcm_ds['X']
   da_Y = MITgcm_ds['Yp1']
   da_Z = MITgcm_ds['Zmd000038'] 

   MITgcm_stats_ds = xr.open_dataset(MITgcm_stats_filename)

   inputs_mean, inputs_std, inputs_range, targets_mean, targets_std, targets_range = ReadMeanStd(dimension, da_Z.shape[0], pred_jump)

   if land == 'ExcLand':
      da_Y = da_Y[3:]
  
   no_samples = 50   # We calculate stats over the entire dataset, but only output a subset of the samples based on no_samples
   no_depth_levels = da_Z.shape[0]

   batch_no = 0
   count = 0

   if isinstance(h, list):
      for i in range(len(h)):
         h[i].train(False)
         if torch.cuda.is_available():
            h[i] = h[i].cuda()
   else:
      h.train(False)
      if torch.cuda.is_available():
         h = h.cuda()

   with torch.no_grad():
      for input_batch, target_batch, masks, out_masks in data_loader:
          input_batch = input_batch.to(device, non_blocking=True, dtype=torch.float)
          target_batch = target_batch.cpu().detach().numpy()
          masks = masks.to(device, non_blocking=True, dtype=torch.float)

          # get prediction from the model, given the inputs
          if isinstance(h, list):
             predicted_batch = h[0]( torch.cat( (input_batch, masks), dim=channel_dim) ) 
             for i in range(1, len(h)):
                predicted_batch = predicted_batch + h[i]( torch.cat( (input_batch, masks ), dim=channel_dim) )
             predicted_batch = predicted_batch / len(h)
          else:
             predicted_batch = h( torch.cat( (input_batch, masks), dim=channel_dim) ) 
          predicted_batch = predicted_batch.cpu().detach().numpy() 

          # Denormalise, if predlen>1 (loss function calc over multiple steps) only take first step, and convert from increments to fields for outputting
          target_batch = rr.RF_DeNormalise(target_batch[:, :no_phys_channels, :, :], targets_mean, targets_std, targets_range,
                                           dimension, norm_method)
          predicted_batch = rr.RF_DeNormalise(predicted_batch, targets_mean, targets_std, targets_range, dimension, norm_method)
          input_batch = input_batch.cpu().detach().numpy()

          if model_style == 'ConvLSTM' or model_style == 'UNetConvLSTM':
             input_batch[:,:,:no_phys_channels,:,:] = rr.RF_DeNormalise( input_batch[:,:,:no_phys_channels,:,:],
                                                                    inputs_mean, inputs_std, inputs_range, dimension, norm_method )
             target_batch = input_batch[:,-1,:no_phys_channels,:,:] + target_batch
             predicted_batch = input_batch[:,-1,:no_phys_channels,:,:] + predicted_batch
          elif dimension == '2d':
             input_batch[:,:no_phys_channels,:,:] = rr.RF_DeNormalise( input_batch[:,:no_phys_channels,:,:],
                                                                       inputs_mean, inputs_std, inputs_range, dimension, norm_method )
             #target_batch = input_batch[:,-(3*no_depth_levels+1):,:,:] + target_batch
             #predicted_batch = input_batch[:,-(3*no_depth_levels+1):,:,:] + predicted_batch
             target_batch = input_batch[:,(histlen-1)*no_phys_channels:histlen*no_phys_channels,:,:] + target_batch
             predicted_batch = input_batch[:,(histlen-1)*no_phys_channels:histlen*no_phys_channels,:,:] + predicted_batch
          elif dimension == '3d':
             input_batch[:,:no_phys_channels,:,:,:] = rr.RF_DeNormalise( input_batch[:,:no_phys_channels,:,:,:],
                                                                         inputs_mean, inputs_std, inputs_range, dimension, norm_method )
             target_batch = input_batch[:,(histlen-1)*no_phys_channels:histlen*no_phys_channels,:,:,:] + target_batch
             predicted_batch = input_batch[:,(histlen-1)*no_phys_channels:histlen*no_phys_channels,:,:,:] + predicted_batch

          # Add summed error of this batch
          if batch_no==0:
             targets     = target_batch
             predictions = predicted_batch 
             summed_sq_error = np.sum( np.square(predicted_batch-target_batch), axis=0 ) 
          else:
             if count < no_samples:
                targets     = np.concatenate( ( targets, target_batch ), axis=0)
                predictions = np.concatenate( ( predictions, predicted_batch ), axis=0)
             summed_sq_error = summed_sq_error + np.sum( np.square(predicted_batch-target_batch), axis=0 )

          batch_no = batch_no+1
          count = count + input_batch.shape[0]

          del input_batch
          del target_batch
          del predicted_batch
          gc.collect()
          torch.cuda.empty_cache()

   no_samples = min(no_samples, count) # Reduce no_samples if this is bigger than total samples!
  
   rms_error = np.sqrt( np.divide(summed_sq_error, count) )
   # mask rms error (for ease of viewing plots), leave othere variables so we can see what's happening
   rms_error = np.where( out_masks[0]==1, rms_error, np.nan)

   # Set up netcdf files
   nc_filename = '../../../Channel_nn_Outputs/'+model_name+'/STATS/'+model_name+'_'+str(no_epochs)+'epochs_StatsOutput_'+file_append+'.nc'
   nc_file = nc4.Dataset(nc_filename,'w', format='NETCDF4') #'w' stands for write
   # Create Dimensions
   no_depth_levels = 38  # Hard coded...perhaps should change...?
   nc_file.createDimension('T', no_samples)
   nc_file.createDimension('Z', da_Z.values.shape[0])
   nc_file.createDimension('Y', y_dim_used) 
   nc_file.createDimension('X', da_X.values.shape[0])
   # Create variables
   nc_T = nc_file.createVariable('T', 'i4', 'T')
   nc_Z = nc_file.createVariable('Z', 'i4', 'Z')
   nc_Y = nc_file.createVariable('Y', 'i4', 'Y')  
   nc_X = nc_file.createVariable('X', 'i4', 'X')

   nc_TrueTemp   = nc_file.createVariable( 'True_Temp'  , 'f4', ('T', 'Z', 'Y', 'X') )
   nc_TrueU      = nc_file.createVariable( 'True_U'     , 'f4', ('T', 'Z', 'Y', 'X') )
   nc_TrueV      = nc_file.createVariable( 'True_V'     , 'f4', ('T', 'Z', 'Y', 'X') )
   nc_TrueEta    = nc_file.createVariable( 'True_Eta'   , 'f4', ('T', 'Y', 'X')      )

   nc_Mask       = nc_file.createVariable( 'Mask'       , 'f4', ('Z', 'Y', 'X') )
   nc_TempMask   = nc_file.createVariable( 'Temp_Mask'  , 'f4', ('Z', 'Y', 'X') )
   nc_UMask      = nc_file.createVariable( 'U_Mask'     , 'f4', ('Z', 'Y', 'X') )
   nc_VMask      = nc_file.createVariable( 'V_Mask'     , 'f4', ('Z', 'Y', 'X') )
   nc_EtaMask    = nc_file.createVariable( 'Eta_Mask'   , 'f4', ('Y', 'X')      )

   nc_PredTemp   = nc_file.createVariable( 'Pred_Temp'  , 'f4', ('T', 'Z', 'Y', 'X') )
   nc_PredU      = nc_file.createVariable( 'Pred_U'     , 'f4', ('T', 'Z', 'Y', 'X') )
   nc_PredV      = nc_file.createVariable( 'Pred_V'     , 'f4', ('T', 'Z', 'Y', 'X') ) 
   nc_PredEta    = nc_file.createVariable( 'Pred_Eta'   , 'f4', ('T', 'Y', 'X')      )

   nc_TempErrors = nc_file.createVariable( 'Temp_Errors', 'f4', ('T', 'Z', 'Y', 'X') )
   nc_UErrors    = nc_file.createVariable( 'U_Errors'   , 'f4', ('T', 'Z', 'Y', 'X') )
   nc_VErrors    = nc_file.createVariable( 'V_Errors'   , 'f4', ('T', 'Z', 'Y', 'X') )
   nc_EtaErrors  = nc_file.createVariable( 'Eta_Errors' , 'f4', ('T', 'Y', 'X')      )

   nc_TempRMS    = nc_file.createVariable( 'Temp_RMS'   , 'f4', ('Z', 'Y', 'X')      )
   nc_U_RMS      = nc_file.createVariable( 'U_RMS'      , 'f4', ('Z', 'Y', 'X')      )
   nc_V_RMS      = nc_file.createVariable( 'V_RMS'      , 'f4', ('Z', 'Y', 'X')      )
   nc_EtaRMS     = nc_file.createVariable( 'Eta_RMS'    , 'f4', ('Y', 'X')           )

   nc_Scaled_TempRMS = nc_file.createVariable( 'ScaledTempRMS' , 'f4', ('Z', 'Y', 'X')      )
   nc_Scaled_U_RMS   = nc_file.createVariable( 'Scaled_U_RMS'  , 'f4', ('Z', 'Y', 'X')      )
   nc_Scaled_V_RMS   = nc_file.createVariable( 'Scaled_V_RMS'  , 'f4', ('Z', 'Y', 'X')      )
   nc_Scaled_EtaRMS  = nc_file.createVariable( 'ScaledEtaRMS'  , 'f4', ('Y', 'X')           )

   # Fill variables
   nc_T[:] = np.arange(no_samples)
   nc_Z[:] = da_Z.values
   nc_Y[:] = da_Y.values[:y_dim_used]
   nc_X[:] = da_X.values

   if dimension=='2d':
      nc_TrueTemp[:,:,:,:] = targets[:no_samples,0:no_depth_levels,:,:] 
      nc_TrueU[:,:,:,:]    = targets[:no_samples,1*no_depth_levels:2*no_depth_levels,:,:]
      nc_TrueV[:,:,:,:]    = targets[:no_samples,2*no_depth_levels:3*no_depth_levels,:,:]
      nc_TrueEta[:,:,:]    = targets[:no_samples,3*no_depth_levels,:,:]

      #nc_Mask[:,:,:]     = masks[0,:,:,:].cpu()
      nc_TempMask[:,:,:] = out_masks[0,0:no_depth_levels,:,:]
      nc_UMask[:,:,:]    = out_masks[0,1*no_depth_levels:2*no_depth_levels,:,:]
      nc_VMask[:,:,:]    = out_masks[0,2*no_depth_levels:3*no_depth_levels,:,:]
      nc_EtaMask[:,:]    = out_masks[0,3*no_depth_levels,:,:]

      nc_PredTemp[:,:,:,:] = predictions[:no_samples,0:no_depth_levels,:,:]
      nc_PredU[:,:,:,:]    = predictions[:no_samples,1*no_depth_levels:2*no_depth_levels,:,:]
      nc_PredV[:,:,:,:]    = predictions[:no_samples,2*no_depth_levels:3*no_depth_levels,:,:]
      nc_PredEta[:,:,:]    = predictions[:no_samples,3*no_depth_levels,:,:]
 
      nc_TempErrors[:,:,:,:] = ( predictions[:no_samples,0:no_depth_levels,:,:]
                                 - targets[:no_samples,0:no_depth_levels,:,:] )
      nc_UErrors[:,:,:,:]    = ( predictions[:no_samples,1*no_depth_levels:2*no_depth_levels,:,:]
                                 - targets[:no_samples,1*no_depth_levels:2*no_depth_levels,:,:] )
      nc_VErrors[:,:,:,:]    = ( predictions[:no_samples,2*no_depth_levels:3*no_depth_levels,:,:] 
                                 - targets[:no_samples,2*no_depth_levels:3*no_depth_levels,:,:] )
      nc_EtaErrors[:,:,:]    = ( predictions[:no_samples,3*no_depth_levels,:,:] 
                                 - targets[:no_samples,3*no_depth_levels,:,:] )

      nc_TempRMS[:,:,:] = rms_error[0:no_depth_levels,:,:]
      nc_U_RMS[:,:,:]   = rms_error[1*no_depth_levels:2*no_depth_levels,:,:] 
      nc_V_RMS[:,:,:]   = rms_error[2*no_depth_levels:3*no_depth_levels,:,:]
      nc_EtaRMS[:,:]    = rms_error[3*no_depth_levels,:,:]

      nc_Scaled_TempRMS[:,:,:] = rms_error[0:no_depth_levels,:,:] / MITgcm_stats_ds['StdTemp'].values
      nc_Scaled_U_RMS[:,:,:]   = rms_error[1*no_depth_levels:2*no_depth_levels,:,:] / MITgcm_stats_ds['StdUVel'].values
      nc_Scaled_V_RMS[:,:,:]   = rms_error[2*no_depth_levels:3*no_depth_levels,:,:] / MITgcm_stats_ds['StdVVel'].values
      nc_Scaled_EtaRMS[:,:]    = rms_error[3*no_depth_levels,:,:] / MITgcm_stats_ds['StdEta'].values

   elif dimension=='3d':
      nc_TrueTemp[:,:,:,:] = targets[:no_samples,0,:,:,:] 
      nc_TrueU[:,:,:,:]    = targets[:no_samples,1,:,:,:]
      nc_TrueV[:,:,:,:]    = targets[:no_samples,2,:,:,:]
      nc_TrueEta[:,:,:]    = targets[:no_samples,3,0,:,:]

      nc_Mask[:,:,:]     = masks[0,0,:,:,:].cpu()
      nc_TempMask[:,:,:] = out_masks[0,0,:,:,:]
      nc_UMask[:,:,:]    = out_masks[0,1,:,:,:]
      nc_VMask[:,:,:]    = out_masks[0,2,:,:,:]
      nc_EtaMask[:,:]    = out_masks[0,3,0,:,:]

      nc_PredTemp[:,:,:,:] = predictions[:no_samples,0,:,:,:]
      nc_PredU[:,:,:,:]    = predictions[:no_samples,1,:,:,:]
      nc_PredV[:,:,:,:]    = predictions[:no_samples,2,:,:,:]
      nc_PredEta[:,:,:]    = predictions[:no_samples,3,0,:,:]
 
      nc_TempErrors[:,:,:,:] = ( predictions[:no_samples,0,:,:,:]
                                 - targets[:no_samples,0,:,:,:] )
      nc_UErrors[:,:,:,:]    = ( predictions[:no_samples,1,:,:,:]
                                 - targets[:no_samples,1,:,:,:] )
      nc_VErrors[:,:,:,:]    = ( predictions[:no_samples,2,:,:,:] 
                                 - targets[:no_samples,2,:,:,:] )
      nc_EtaErrors[:,:,:]    = ( predictions[:no_samples,3,0,:,:] 
                                 - targets[:no_samples,3,0,:,:] )

      nc_TempRMS[:,:,:] = rms_error[0,:,:,:]
      nc_U_RMS[:,:,:]   = rms_error[1,:,:,:] 
      nc_V_RMS[:,:,:]   = rms_error[2,:,:,:]
      nc_EtaRMS[:,:]    = rms_error[3,0,:,:]


def IterativelyPredict(model_name, model_style, MeanStd_prefix, MITgcm_filename, Iterate_Dataset, h, start, for_len, no_epochs,
                       y_dim_used, land, dimension, histlen, landvalues, iterate_method, iterate_smooth, norm_method, channel_dim, pred_jump, for_subsample):
   logging.info('Iterating')

   landvalues = torch.tensor(landvalues)
  
   device = 'cuda' if torch.cuda.is_available() else 'cpu'
   if isinstance(h, list):
      for i in range(len(h)):
         h[i].train(False)
         if torch.cuda.is_available():
            h[i] = h[i].cuda()
   else:
      h.train(False)
      if torch.cuda.is_available():
         h = h.cuda()

   # Read in grid data from MITgcm file
   MITgcm_ds = xr.open_dataset(MITgcm_filename)
   da_X = MITgcm_ds['X']
   da_Y = MITgcm_ds['Yp1']
   if land == 'ExcLand':
      da_Y = da_Y[3:]
   da_Z = MITgcm_ds['Zmd000038'] 

   no_depth_levels = 38   #Unesseccary duplication, could use with da_Z shape , should change...

   inputs_mean, inputs_std, inputs_range, targets_mean, targets_std, targets_range = ReadMeanStd(dimension, da_Z.shape[0], pred_jump)

   # Set up netcdf files to write to
   nc_filename = '../../../Channel_nn_Outputs/'+model_name+'/ITERATED_FORECAST/'+model_name+'_'+str(no_epochs)+'epochs_'+iterate_method+'_smooth'+str(iterate_smooth)+'_Forecast'+str(for_len)+'.nc'
   nc_file = nc4.Dataset(nc_filename,'w', format='NETCDF4') #'w' stands for write
   # Create Dimensions
   nc_file.createDimension('T', for_len/for_subsample+histlen)
   nc_file.createDimension('Z', da_Z.values.shape[0])
   nc_file.createDimension('Y', y_dim_used)
   nc_file.createDimension('X', da_X.values.shape[0])

   # Create variables
   nc_T = nc_file.createVariable('T', 'i4', 'T')
   nc_Z = nc_file.createVariable('Z', 'i4', 'Z')
   nc_Y = nc_file.createVariable('Y', 'i4', 'Y')  
   nc_X = nc_file.createVariable('X', 'i4', 'X')

   nc_TrueTemp   = nc_file.createVariable( 'True_Temp'  , 'f4', ('T', 'Z', 'Y', 'X') )
   nc_TrueU      = nc_file.createVariable( 'True_U'     , 'f4', ('T', 'Z', 'Y', 'X') )
   nc_TrueV      = nc_file.createVariable( 'True_V'     , 'f4', ('T', 'Z', 'Y', 'X') )
   nc_TrueEta    = nc_file.createVariable( 'True_Eta'   , 'f4', ('T', 'Y', 'X')      )

   nc_TrueTempInc = nc_file.createVariable( 'Tr_TInc'  , 'f4', ('T', 'Z', 'Y', 'X') )
   nc_TrueUInc    = nc_file.createVariable( 'Tr_UInc'     , 'f4', ('T', 'Z', 'Y', 'X') )
   nc_TrueVInc    = nc_file.createVariable( 'Tr_VInc'     , 'f4', ('T', 'Z', 'Y', 'X') )
   nc_TrueEtaInc  = nc_file.createVariable( 'Tr_EInc'   , 'f4', ('T', 'Y', 'X')      )

   nc_PredTempInc= nc_file.createVariable( 'Pr_T_Inc'  , 'f4', ('T', 'Z', 'Y', 'X') )
   nc_PredUInc   = nc_file.createVariable( 'Pr_U_Inc'     , 'f4', ('T', 'Z', 'Y', 'X') )
   nc_PredVInc   = nc_file.createVariable( 'Pr_V_Inc'     , 'f4', ('T', 'Z', 'Y', 'X') ) 
   nc_PredEtaInc = nc_file.createVariable( 'Pr_E_Inc'   , 'f4', ('T', 'Y', 'X')      )

   nc_PredTempFld= nc_file.createVariable( 'Pred_Temp'  , 'f4', ('T', 'Z', 'Y', 'X') )
   nc_PredUFld   = nc_file.createVariable( 'Pred_U'     , 'f4', ('T', 'Z', 'Y', 'X') )
   nc_PredVFld   = nc_file.createVariable( 'Pred_V'     , 'f4', ('T', 'Z', 'Y', 'X') ) 
   nc_PredEtaFld = nc_file.createVariable( 'Pred_Eta'   , 'f4', ('T', 'Y', 'X')      )

   nc_TempErrors = nc_file.createVariable( 'Temp_Errors', 'f4', ('T', 'Z', 'Y', 'X') )
   nc_UErrors    = nc_file.createVariable( 'U_Errors'   , 'f4', ('T', 'Z', 'Y', 'X') )
   nc_VErrors    = nc_file.createVariable( 'V_Errors'   , 'f4', ('T', 'Z', 'Y', 'X') )
   nc_EtaErrors  = nc_file.createVariable( 'Eta_Errors' , 'f4', ('T', 'Y', 'X')      )

   nc_TempMask   = nc_file.createVariable( 'Temp_Mask'  , 'f4', ('Z', 'Y', 'X') )
   nc_UMask      = nc_file.createVariable( 'U_Mask'     , 'f4', ('Z', 'Y', 'X') )
   nc_VMask      = nc_file.createVariable( 'V_Mask'     , 'f4', ('Z', 'Y', 'X') )
   nc_EtaMask    = nc_file.createVariable( 'Eta_Mask'   , 'f4', ('Y', 'X')      )

   # Fill netcdf coordinate variables
   nc_T[:] = np.arange(for_len/for_subsample+histlen)
   nc_Z[:] = da_Z.values
   nc_Y[:] = da_Y.values[:y_dim_used]
   nc_X[:] = da_X.values

   # Read in data from MITgcm dataset (take 0th entry, as others are target, masks etc) and save as first entry in both arrays
   input_sample, target_sample, masks, out_masks = Iterate_Dataset.__getitem__(start)

   # Give extra dimension at front (number of samples - here 1)
   input_sample = input_sample.unsqueeze(0)
   masks = masks.unsqueeze(0)
   if model_style == 'ConvLSTM' or model_style == 'UNetConvLSTM':
      iterated_fields = rr.RF_DeNormalise(Iterate_Dataset.__getitem__(start)[0][:,:histlen*(no_depth_levels*3+1),:,:],
                                          inputs_mean, inputs_std, inputs_range, dimension, norm_method)
      MITgcm_data = Iterate_Dataset.__getitem__(start)[0][:,:histlen*(no_depth_levels*3+1),:,:]
   elif dimension == '2d':
      iterated_fields = rr.RF_DeNormalise(Iterate_Dataset.__getitem__(start)[0][:(no_depth_levels*3+1),:,:].unsqueeze(0),
                                          inputs_mean, inputs_std, inputs_range, dimension, norm_method)
      MITgcm_data = Iterate_Dataset.__getitem__(start)[0].unsqueeze(0)[:,:no_depth_levels*3+1,:,:]
      for time in range(1,histlen):
          iterated_fields = torch.cat( (iterated_fields,
                                       rr.RF_DeNormalise(Iterate_Dataset.__getitem__(start)[0]
                                                         [time*(no_depth_levels*3+1):(time+1)*(no_depth_levels*3+1),:,:].unsqueeze(0),
                                                         inputs_mean, inputs_std, inputs_range, dimension, norm_method) ),
                                     axis = 0)
          MITgcm_data = torch.cat( ( MITgcm_data,
                                     Iterate_Dataset.__getitem__(start)[0].unsqueeze(0)[:,time*(no_depth_levels*3+1):(time+1)*(no_depth_levels*3+1),:,:] ),
                                   axis=0)
   elif dimension == '3d':
      iterated_fields = rr.RF_DeNormalise(Iterate_Dataset.__getitem__(start)[0][:4,:,:,:].unsqueeze(0),
                                          inputs_mean, inputs_std, inputs_range, dimension, norm_method)
      MITgcm_data = Iterate_Dataset.__getitem__(start)[0][:4,:,:,:].unsqueeze(0)
      for time in range(1,histlen):
          iterated_fields = torch.cat( (iterated_fields,
                                       rr.RF_DeNormalise(Iterate_Dataset.__getitem__(start)[0]
                                                         [time*4:(time+1)*4,:,:,:].unsqueeze(0),
                                                         inputs_mean, inputs_std, inputs_range, dimension, norm_method) ),
                                     axis = 0)
          MITgcm_data = torch.cat( (MITgcm_data, 
                                   Iterate_Dataset.__getitem__(start)[0][time*4:(time+1)*4,:,:].unsqueeze(0) ),
                                   axis=0)
   predicted_increment = torch.zeros(MITgcm_data.shape)
   true_increment = torch.zeros(MITgcm_data.shape)

   # Make iterative forecast, saving predictions as we go, and pull out MITgcm data at correct time steps
   with torch.no_grad():
      for time in range(for_len):
         # Make prediction, for both multi model ensemble, or single model predictions
         if isinstance(h, list):
            predicted = h[0]( torch.cat((input_sample, masks), axis=channel_dim)
                              .to(device, non_blocking=True, dtype=torch.float) ).cpu().detach()
            for i in range(1, len(h)):
               predicted = predicted + \
                           h[i]( torch.cat((input_sample, masks), axis=channel_dim)
                                 .to(device, non_blocking=True, dtype=torch.float) ).cpu().detach()
            predicted = predicted / len(h)
         else:
            predicted = h( torch.cat( (input_sample, masks),axis=channel_dim ).to(device, non_blocking=True, dtype=torch.float)
                         ).cpu().detach()

         # Denormalise 
         predicted = rr.RF_DeNormalise(predicted, targets_mean, targets_std, targets_range, dimension, norm_method)
         if ( time%for_subsample == 0 ):
            print('    '+str(time))
            predicted_increment = torch.cat( (predicted_increment, predicted), axis=0)

         # Calculate next field, by combining prediction (increment) with field
         if iterate_method == 'simple':
            next_field = iterated_fields[-1] + predicted
         elif iterate_method == 'AB2':
            if time == 0:
               if dimension == '2d':
                  next_field = rr.RF_DeNormalise(Iterate_Dataset.__getitem__(start+time+1)[0]
                                                 [(histlen-1)*(no_depth_levels*3+1):histlen*(no_depth_levels*3+1),:,:].unsqueeze(0),
                                                 inputs_mean, inputs_std, inputs_range, dimension, norm_method)
               elif dimension == '3d':
                  next_field = rr.RF_DeNormalise(Iterate_Dataset.__getitem__(start+time+1)[0]
                                                 [(histlen-1)*4:histlen*4,:,:,:].unsqueeze(0),
                                                 inputs_mean, inputs_std, inputs_range, dimension, norm_method)
            else: 
               next_field = iterated_fields[-1] + 3./2. * predicted - 1./2. * old_predicted
            old_predicted = predicted
         elif iterate_method == 'RK4':
            if dimension == '2d':
               k1 = predicted
               # (k1)/2 added to state, masked, then normalised, and finally masks catted on. And then passed through neural net)
               k2 = h( torch.cat( (
                          rr.RF_Normalise( 
                              torch.where( out_masks==1, ( iterated_fields[-1,:,:,:] + (k1)/2. ), landvalues.unsqueeze(1).unsqueeze(2) ),
                              inputs_mean, inputs_std, inputs_range, dimension, norm_method), 
                          masks),axis=channel_dim ).to(device, non_blocking=True, dtype=torch.float)
                     ).cpu().detach()
               k3 = h( torch.cat( (
                          rr.RF_Normalise( 
                              torch.where( out_masks==1, ( iterated_fields[-1,:,:,:] + (k2)/2. ), landvalues.unsqueeze(1).unsqueeze(2) ),
                              inputs_mean, inputs_std, inputs_range, dimension, norm_method), 
                          masks),axis=channel_dim ).to(device, non_blocking=True, dtype=torch.float)
                     ).cpu().detach()
               k4 = h( torch.cat( (
                          rr.RF_Normalise( 
                             torch.where( out_masks==1, ( iterated_fields[-1,:,:,:] + (k3)    ), landvalues.unsqueeze(1).unsqueeze(2) ),
                             inputs_mean, inputs_std, inputs_range, dimension, norm_method), 
                          masks),axis=channel_dim ).to(device, non_blocking=True, dtype=torch.float)
                     ).cpu().detach()
               next_field = iterated_fields[-1,:,:,:] + (1./6.) * ( k1 + 2*k2 +2*k3 + k4 )         
            elif dimension == '3d':
               k1 = predicted
               # (k1)/2 added to state, masked, then normalised, and finally masks catted on. And then passed through neural net)
               k2 = h( torch.cat( (
                          rr.RF_Normalise( 
                             torch.where( out_masks==1, ( iterated_fields[-1,:,:,:,:] + (k1)/2. ), landvalues.unsqueeze(1).unsqueeze(2).unsqueeze(3) ),
                             inputs_mean, inputs_std, inputs_range, dimension, norm_method), 
                          masks),axis=channel_dim ).to(device, non_blocking=True, dtype=torch.float)
                     ).cpu().detach()
               k3 = h( torch.cat( (
                          rr.RF_Normalise( 
                             torch.where( out_masks==1, ( iterated_fields[-1,:,:,:,:] + (k2)/2. ), landvalues.unsqueeze(1).unsqueeze(2).unsqueeze(3) ),
                             inputs_mean, inputs_std, inputs_range, dimension, norm_method), 
                          masks),axis=channel_dim ).to(device, non_blocking=True, dtype=torch.float)
                     ).cpu().detach()
               k4 = h( torch.cat( (
                          rr.RF_Normalise( 
                             torch.where( out_masks==1, ( iterated_fields[-1,:,:,:,:] + (k3)    ), landvalues.unsqueeze(1).unsqueeze(2).unsqueeze(3) ),
                             inputs_mean, inputs_std, inputs_range, dimension, norm_method), 
                          masks),axis=channel_dim ).to(device, non_blocking=True, dtype=torch.float)
                     ).cpu().detach()
               next_field = iterated_fields[-1,:,:,:,:] + (1./6.) * ( k1 + 2*k2 +2*k3 + k4 )         
         else:
            sys.exit('no suitable iteration method chosen')

         # Mask field
         if dimension == '2d':
            next_field = torch.where(out_masks==1, next_field, landvalues.unsqueeze(1).unsqueeze(2) )
         if dimension == '3d':
            next_field = torch.where( out_masks==1, next_field, landvalues.unsqueeze(1).unsqueeze(2).unsqueeze(3) )

         # Smooth field 
         if iterate_smooth != 0:
            for channel in range(next_field.shape[1]):
              filter = gcm_filters.Filter( filter_scale=iterate_smooth, dx_min=1, filter_shape=gcm_filters.FilterShape.TAPER,
                                           grid_type=gcm_filters.GridType.REGULAR_WITH_LAND,
                                           grid_vars={'wet_mask': xr.DataArray(out_masks[channel,:,:], dims=['y','x'])} )
              next_field[0,channel,:,:] = torch.from_numpy( filter.apply( xr.DataArray(next_field[0,channel,:,:], dims=['y', 'x'] ),
                                                            dims=['y', 'x']).values )

         if ( time%for_subsample == 0 ):
            print('    '+str(time))
            # Cat prediction onto existing fields
            iterated_fields = torch.cat( ( iterated_fields, next_field ), axis=0 )
            # Get MITgcm data for relevant step
            # Plus one to time dimension as we want the next step, to match time step of prediction
            if model_style == 'ConvLSTM' or model_style == 'UNetConvLSTM':
               MITgcm_data = torch.cat( ( MITgcm_data, Iterate_Dataset.__getitem__(start+time+1)[0][histlen-1:histlen,:3*no_depth_levels+1,:,:] ), axis=0)
               true_increment = torch.cat( ( true_increment, 
                                             Iterate_Dataset.__getitem__(start+time+1)[0][histlen-1:histlen,:3*no_depth_levels+1,:,:]
                                             - Iterate_Dataset.__getitem__(start+time)[0][histlen-1:histlen,:3*no_depth_levels+1,:,:] ), axis=0)
            elif dimension == '2d':
               MITgcm_data = torch.cat( ( MITgcm_data, Iterate_Dataset.__getitem__(start+time+1)[0]
                                          [(histlen-1)*(no_depth_levels*3+1):histlen*(no_depth_levels*3+1),:,:].unsqueeze(0) ),
                                          axis=0 )
               true_increment = torch.cat( ( true_increment,
                                             Iterate_Dataset.__getitem__(start+time+1)[0][(histlen-1)*(no_depth_levels*3+1):histlen*(no_depth_levels*3+1),:,:].unsqueeze(0)
                                             - Iterate_Dataset.__getitem__(start+time)[0][(histlen-1)*(no_depth_levels*3+1):histlen*(no_depth_levels*3+1),:,:].unsqueeze(0) ),
                                          axis=0 )
            elif dimension == '3d':
               MITgcm_data = torch.cat( ( MITgcm_data, Iterate_Dataset.__getitem__(start+time+1)[0]
                                          [(histlen-1)*4:histlen*4,:,:,:].unsqueeze(0) ),
                                          axis=0 )
               true_increment = torch.cat( ( true_increment,
                                             Iterate_Dataset.__getitem__(start+time+1)[0][(histlen-1)*4:histlen*4,:,:,:].unsqueeze(0)
                                             - Iterate_Dataset.__getitem__(start+time)[0][(histlen-1)*4:histlen*4,:,:,:].unsqueeze(0) ),
                                          axis=0 )

         # Re-Normalise next field ready to use as inputs
         next_field = rr.RF_Normalise(next_field, inputs_mean, inputs_std, inputs_range, dimension, norm_method)

         # Prep new input sample
         if model_style == 'ConvLSTM' or model_style == 'UNetConvLSTM':
            for time in range(histlen-1):
               input_sample[0,time,:no_depth_levels*3+1,:,:] = \
                                 input_sample[0,time+1,:no_depth_levels*3+1,:,:]
            input_sample[0,histlen-1,:no_depth_levels*3+1,:,:] = next_field[0,:,:,:]
         elif dimension == '2d':
            for time in range(histlen-1):
               input_sample[0,time*(3*no_depth_levels+1):(time+1)*(3*no_depth_levels+1),:,:] = \
                                 input_sample[0,(time+1)*(3*no_depth_levels+1):(time+2)*(3*no_depth_levels+1),:,:]
            input_sample[0,(histlen-1)*(3*no_depth_levels+1):(histlen)*(3*no_depth_levels+1),:,:] = next_field[0,:,:,:]
         elif dimension == '3d':
            for time in range(histlen-1):
               input_sample[0,time*4:(time+1)*4,:,:] = \
                                 input_sample[0,(time+1)*4:(time+2)*4,:,:]
            input_sample[0,(histlen-1)*4:(histlen)*4,:,:] = next_field
           
         del next_field
         del predicted 
         gc.collect()
         torch.cuda.empty_cache()

   ## Denormalise MITgcm Data 
   MITgcm_data          = rr.RF_DeNormalise(MITgcm_data, inputs_mean, inputs_std, inputs_range, dimension, norm_method) 
 
   # Fill variables
   if dimension == '2d':
      nc_TrueTemp[:,:,:,:] = MITgcm_data[:,0:no_depth_levels,:,:] 
      nc_TrueU[:,:,:,:]    = MITgcm_data[:,1*no_depth_levels:2*no_depth_levels,:,:]
      nc_TrueV[:,:,:,:]    = MITgcm_data[:,2*no_depth_levels:3*no_depth_levels,:,:]
      nc_TrueEta[:,:,:]    = MITgcm_data[:,3*no_depth_levels,:,:]

      nc_PredTempInc[:,:,:,:] = predicted_increment[:,0:no_depth_levels,:,:]
      nc_PredUInc[:,:,:,:]    = predicted_increment[:,1*no_depth_levels:2*no_depth_levels,:,:]
      nc_PredVInc[:,:,:,:]    = predicted_increment[:,2*no_depth_levels:3*no_depth_levels,:,:]
      nc_PredEtaInc[:,:,:]    = predicted_increment[:,3*no_depth_levels,:,:]
      
      nc_PredTempFld[:,:,:,:] = iterated_fields[:,0:no_depth_levels,:,:]
      nc_PredUFld[:,:,:,:]    = iterated_fields[:,1*no_depth_levels:2*no_depth_levels,:,:]
      nc_PredVFld[:,:,:,:]    = iterated_fields[:,2*no_depth_levels:3*no_depth_levels,:,:]
      nc_PredEtaFld[:,:,:]    = iterated_fields[:,3*no_depth_levels,:,:]
      
      nc_TrueTempInc[1:,:,:,:] = MITgcm_data[1:,0:no_depth_levels,:,:] - MITgcm_data[:-1,0:no_depth_levels,:,:] 
      nc_TrueUInc[1:,:,:,:]    = MITgcm_data[1:,1*no_depth_levels:2*no_depth_levels,:,:] - MITgcm_data[:-1,1*no_depth_levels:2*no_depth_levels,:,:]
      nc_TrueVInc[1:,:,:,:]    = MITgcm_data[1:,2*no_depth_levels:3*no_depth_levels,:,:] - MITgcm_data[:-1,2*no_depth_levels:3*no_depth_levels,:,:]
      nc_TrueEtaInc[1:,:,:]    = MITgcm_data[1:,3*no_depth_levels,:,:] - MITgcm_data[:-1,3*no_depth_levels,:,:]

      nc_TempErrors[:,:,:,:] = iterated_fields[:,0:no_depth_levels,:,:] - MITgcm_data[:,0:no_depth_levels,:,:]
      nc_UErrors[:,:,:,:]    = iterated_fields[:,1*no_depth_levels:2*no_depth_levels,:,:] - MITgcm_data[:,1*no_depth_levels:2*no_depth_levels,:,:]
      nc_VErrors[:,:,:,:]    = iterated_fields[:,2*no_depth_levels:3*no_depth_levels,:,:] - MITgcm_data[:,2*no_depth_levels:3*no_depth_levels,:,:]
      nc_EtaErrors[:,:,:]    = iterated_fields[:,3*no_depth_levels,:,:] - MITgcm_data[:,3*no_depth_levels,:,:]

      nc_TempMask[:,:,:] = out_masks[0:no_depth_levels,:,:]
      nc_UMask[:,:,:]    = out_masks[1*no_depth_levels:2*no_depth_levels,:,:]
      nc_VMask[:,:,:]    = out_masks[2*no_depth_levels:3*no_depth_levels,:,:]
      nc_EtaMask[:,:]    = out_masks[3*no_depth_levels,:,:]

   elif dimension == '3d':
      nc_TrueTemp[:,:,:,:] = MITgcm_data[:,0,:,:,:] 
      nc_TrueU[:,:,:,:]    = MITgcm_data[:,1,:,:,:]
      nc_TrueV[:,:,:,:]    = MITgcm_data[:,2,:,:,:]
      nc_TrueEta[:,:,:]    = MITgcm_data[:,3,0,:,:]

      nc_PredTempFld[:,:,:,:] = iterated_fields[:,0,:,:,:]
      nc_PredUFld[:,:,:,:]    = iterated_fields[:,1,:,:,:]
      nc_PredVFld[:,:,:,:]    = iterated_fields[:,2,:,:,:]
      nc_PredEtaFld[:,:,:]    = iterated_fields[:,3,0,:,:]
      
      nc_TempErrors[:,:,:,:] = iterated_fields[:,0,:,:,:] - MITgcm_data[:,0,:,:,:]
      nc_UErrors[:,:,:,:]    = iterated_fields[:,1,:,:,:] - MITgcm_data[:,1,:,:,:]
      nc_VErrors[:,:,:,:]    = iterated_fields[:,2,:,:,:] - MITgcm_data[:,2,:,:,:]
      nc_EtaErrors[:,:,:]    = iterated_fields[:,3,0,:,:] - MITgcm_data[:,3,0,:,:]

      nc_TempMask[:,:,:] = out_masks[0,:,:,:]
      nc_UMask[:,:,:]    = out_masks[1,:,:,:]
      nc_VMask[:,:,:]    = out_masks[2,:,:,:]
      nc_EtaMask[:,:]    = out_masks[3,0,:,:]

def PlotTrainingEvolution(model_name, model_style, MeanStd_prefix, MITgcm_filename, Evolution_Dataset, h, optimizer, epochs_list,
                          dimension, histlen, landvalues, norm_method, channel_dim, pred_jump, no_phys_channels):

   logging.info('Plotting Training Evolution')

   level = 2
   time = 5

   rootdir = '../../../Channel_nn_Outputs/'+model_name
   
   device = 'cuda' if torch.cuda.is_available() else 'cpu'
   MITgcm_ds = xr.open_dataset(MITgcm_filename)
   da_X = MITgcm_ds['X']
   da_Y = MITgcm_ds['Y']
   da_Z = MITgcm_ds['Zmd000038']
   no_levels = da_Z.shape[0]
   
   inputs_mean, inputs_std, inputs_range, targets_mean, targets_std, targets_range = ReadMeanStd(dimension, no_levels, pred_jump)

   input_sample, target_sample, masks, out_masks = Evolution_Dataset.__getitem__(time)
   input_sample = input_sample.unsqueeze(0)
   target_sample = target_sample.unsqueeze(0)

   if model_style == 'ConvLSTM' or model_style == 'UNetConvLSTM':
      prev_field = rr.RF_DeNormalise(input_sample[:,-1:,:no_phys_channels,:,:], inputs_mean, inputs_std, inputs_range, dimension, norm_method)[0,:,:,:,:]
   elif dimension == '2d':
      prev_field = rr.RF_DeNormalise(input_sample[:,(histlen-1)*no_phys_channels:histlen*no_phys_channels,:,:],
                                                  inputs_mean, inputs_std, inputs_range, dimension, norm_method)[0,:,:,:]
   elif dimension == '3d':
      prev_field = rr.RF_DeNormalise(input_sample[:,(histlen-1)*no_phys_channels:histlen*no_phys_channels,:,:,:],
                                                  inputs_mean, inputs_std, inputs_range, dimension, norm_method)[0,:,:,:,:]
   target_tend = rr.RF_DeNormalise(target_sample[:,:no_phys_channels,:,:], targets_mean, targets_std, targets_range, dimension, norm_method)[0,:,:,:]
   target_field = prev_field + target_tend
   target_tend = np.where( out_masks==1., target_tend, np.nan )
   target_field = np.where( out_masks==1., target_field, np.nan )

   for epochs in epochs_list:
   
      # Load model
      losses = {'train' : [], 'train_Temp': [], 'train_U' : [], 'train_V' : [], 'train_Eta' : [], 'val' : [] }
      losses, h, optimizer, current_best_loss = LoadModel(model_name, h, optimizer, epochs, 'inf', losses, False)  
      h.train(False)
      if torch.cuda.is_available():
         h = h.cuda()
   
      # Make prediction
      with torch.no_grad():
         predicted_tend = h( torch.cat( (input_sample, masks.unsqueeze(0)), dim=channel_dim ).to(
                                                                  device, non_blocking=True, dtype=torch.float) ).cpu().detach()
         predicted_tend = rr.RF_DeNormalise(predicted_tend, targets_mean, targets_std, targets_range, dimension, norm_method)[0]
         predicted_field = prev_field + predicted_tend
   
      # Make plot
      predicted_field = np.where( out_masks==1., predicted_field, np.nan )
      predicted_tend = np.where( out_masks==1., predicted_tend, np.nan )
      # Need to add code for 3d options here and make sure it works for histlen>1
      if model_style == 'ConvLSTM' or model_style == 'UNetConvLSTM':
         print(target_field.shape)
         print(predicted_field.shape)
         fig = ChnPlt.plot_depth_fld_diff(target_field[0,level,:,:], 'True Temperature',
                                          predicted_field[0,level,:,:], 'Predicted Temperature',
                                          level, da_X.values, da_Y.values, da_Z.values, title=None)
         plt.savefig(rootdir+'/TRAIN_EVOLUTION/'+model_name+'_'+str(epochs)+'epochs_Temp_diff_z'+str(level)+'.png', bbox_inches = 'tight', pad_inches = 0.1)
         
         fig = ChnPlt.plot_depth_fld_diff(target_field[0,no_levels+level,:,:], 'True East-West Velocity', 
                                          predicted_field[0,no_levels+level,:,:], 'Predicted East-West Velocity',
                                          level, da_X.values, da_Y.values, da_Z.values)
         plt.savefig(rootdir+'/TRAIN_EVOLUTION/'+model_name+'_'+str(epochs)+'epochs_U_diff_z'+str(level)+'.png', bbox_inches = 'tight', pad_inches = 0.1)
         
         fig = ChnPlt.plot_depth_fld_diff(target_field[0,2*no_levels+level,:,:], 'True North-South Velocity',
                                          predicted_field[0,2*no_levels+level,:,:], 'Predicted North-South Velocity',
                                          level, da_X.values, da_Y.values, da_Z.values, title=None)
         plt.savefig(rootdir+'/TRAIN_EVOLUTION/'+model_name+'_'+str(epochs)+'epochs_V_diff_z'+str(level)+'.png', bbox_inches = 'tight', pad_inches = 0.1)
         
         fig = ChnPlt.plot_depth_fld_diff(target_field[0,2*no_levels,:,:], 'True Sea Surface Height',
                                          predicted_field[0,2*no_levels,:,:], 'Predicted Sea Surface Height',
                                          0, da_X.values, da_Y.values, da_Z.values, title=None)
         plt.savefig(rootdir+'/TRAIN_EVOLUTION/'+model_name+'_'+str(epochs)+'epochs_Eta_diff_z'+str(level)+'.png', bbox_inches = 'tight', pad_inches = 0.1)

         fig = ChnPlt.plot_depth_fld_diff(target_tend[level,:,:], 'True Temperature Change',
                                          predicted_tend[level,:,:], 'Predicted Temperature Change',
                                          level, da_X.values, da_Y.values, da_Z.values, title=None)
         plt.savefig(rootdir+'/TRAIN_EVOLUTION/'+model_name+'_'+str(epochs)+'epochs_Temp_diff_z'+str(level)+'_tend.png', bbox_inches = 'tight', pad_inches = 0.1)
         
         fig = ChnPlt.plot_depth_fld_diff(target_tend[no_levels+level,:,:], 'True East-West Velocity Change', 
                                          predicted_tend[no_levels+level,:,:], 'Predicted East-West Velocity Change',
                                          level, da_X.values, da_Y.values, da_Z.values)
         plt.savefig(rootdir+'/TRAIN_EVOLUTION/'+model_name+'_'+str(epochs)+'epochs_U_diff_z'+str(level)+'_tend.png', bbox_inches = 'tight', pad_inches = 0.1)
         
         fig = ChnPlt.plot_depth_fld_diff(target_tend[2*no_levels+level,:,:], 'True North-South Velocity Change',
                                          predicted_tend[2*no_levels+level,:,:], 'Predicted North-South Velocity Change',
                                          level, da_X.values, da_Y.values, da_Z.values, title=None)
         plt.savefig(rootdir+'/TRAIN_EVOLUTION/'+model_name+'_'+str(epochs)+'epochs_V_diff_z'+str(level)+'_tend.png', bbox_inches = 'tight', pad_inches = 0.1)
         
         fig = ChnPlt.plot_depth_fld_diff(target_tend[2*no_levels,:,:], 'True Sea Surface Height Change',
                                          predicted_tend[2*no_levels,:,:], 'Predicted Sea Surface Height Change',
                                          0, da_X.values, da_Y.values, da_Z.values, title=None)
         plt.savefig(rootdir+'/TRAIN_EVOLUTION/'+model_name+'_'+str(epochs)+'epochs_Eta_diff_z'+str(level)+'_tend.png', bbox_inches = 'tight', pad_inches = 0.1)
      elif dimension == '2d':
         fig = ChnPlt.plot_depth_fld_diff(target_field[level,:,:], 'True Temperature',
                                          predicted_field[level,:,:], 'Predicted Temperature',
                                          level, da_X.values, da_Y.values, da_Z.values, title=None)
         plt.savefig(rootdir+'/TRAIN_EVOLUTION/'+model_name+'_'+str(epochs)+'epochs_Temp_diff_z'+str(level)+'.png', bbox_inches = 'tight', pad_inches = 0.1)
         
         fig = ChnPlt.plot_depth_fld_diff(target_field[no_levels+level,:,:], 'True East-West Velocity', 
                                          predicted_field[no_levels+level,:,:], 'Predicted East-West Velocity',
                                          level, da_X.values, da_Y.values, da_Z.values)
         plt.savefig(rootdir+'/TRAIN_EVOLUTION/'+model_name+'_'+str(epochs)+'epochs_U_diff_z'+str(level)+'.png', bbox_inches = 'tight', pad_inches = 0.1)
         
         fig = ChnPlt.plot_depth_fld_diff(target_field[2*no_levels+level,:,:], 'True North-South Velocity',
                                          predicted_field[2*no_levels+level,:,:], 'Predicted North-South Velocity',
                                          level, da_X.values, da_Y.values, da_Z.values, title=None)
         plt.savefig(rootdir+'/TRAIN_EVOLUTION/'+model_name+'_'+str(epochs)+'epochs_V_diff_z'+str(level)+'.png', bbox_inches = 'tight', pad_inches = 0.1)
         
         fig = ChnPlt.plot_depth_fld_diff(target_field[2*no_levels,:,:], 'True Sea Surface Height',
                                          predicted_field[2*no_levels,:,:], 'Predicted Sea Surface Height',
                                          0, da_X.values, da_Y.values, da_Z.values, title=None)
         plt.savefig(rootdir+'/TRAIN_EVOLUTION/'+model_name+'_'+str(epochs)+'epochs_Eta_diff_z'+str(level)+'.png', bbox_inches = 'tight', pad_inches = 0.1)

         fig = ChnPlt.plot_depth_fld_diff(target_tend[level,:,:], 'True Temperature Change',
                                          predicted_tend[level,:,:], 'Predicted Temperature Change',
                                          level, da_X.values, da_Y.values, da_Z.values, title=None)
         plt.savefig(rootdir+'/TRAIN_EVOLUTION/'+model_name+'_'+str(epochs)+'epochs_Temp_diff_z'+str(level)+'_tend.png', bbox_inches = 'tight', pad_inches = 0.1)
         
         fig = ChnPlt.plot_depth_fld_diff(target_tend[no_levels+level,:,:], 'True East-West Velocity Change', 
                                          predicted_tend[no_levels+level,:,:], 'Predicted East-West Velocity Change',
                                          level, da_X.values, da_Y.values, da_Z.values)
         plt.savefig(rootdir+'/TRAIN_EVOLUTION/'+model_name+'_'+str(epochs)+'epochs_U_diff_z'+str(level)+'_tend.png', bbox_inches = 'tight', pad_inches = 0.1)
         
         fig = ChnPlt.plot_depth_fld_diff(target_tend[2*no_levels+level,:,:], 'True North-South Velocity Change',
                                          predicted_tend[2*no_levels+level,:,:], 'Predicted North-South Velocity Change',
                                          level, da_X.values, da_Y.values, da_Z.values, title=None)
         plt.savefig(rootdir+'/TRAIN_EVOLUTION/'+model_name+'_'+str(epochs)+'epochs_V_diff_z'+str(level)+'_tend.png', bbox_inches = 'tight', pad_inches = 0.1)
         
         fig = ChnPlt.plot_depth_fld_diff(target_tend[2*no_levels,:,:], 'True Sea Surface Height Change',
                                          predicted_tend[2*no_levels,:,:], 'Predicted Sea Surface Height Change',
                                          0, da_X.values, da_Y.values, da_Z.values, title=None)
         plt.savefig(rootdir+'/TRAIN_EVOLUTION/'+model_name+'_'+str(epochs)+'epochs_Eta_diff_z'+str(level)+'_tend.png', bbox_inches = 'tight', pad_inches = 0.1)
