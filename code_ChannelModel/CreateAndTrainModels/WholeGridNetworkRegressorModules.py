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
#import torch.multiprocessing
#torch.multiprocessing.set_sharing_strategy('file_system')

#add this routine for memory profiling
def sizeof_fmt(num, suffix='B'):
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)


#def TimeCheck(tic, task):
#   toc = time.time()
#   logging.info('Finished '+task+' at {:0.4f} seconds'.format(toc - tic)+'\n')

def CalcMeanStd(MeanStd_prefix, MITGCM_filename, train_end_ratio, subsample_rate,
                batch_size, land, dimension, tic):

   no_depth_levels = 38 ## Note this is hard coded!!

   if dimension == '2d':
      mean_std_Dataset = rr.MITGCM_Dataset_2d( MITGCM_filename, 0.0, train_end_ratio, subsample_rate, 1, land, tic,
                                               transform=None)    # Use histlen=1 for this, as doesn't make much impact
      mean_std_loader = data.DataLoader(mean_std_Dataset, batch_size=batch_size, shuffle=False, num_workers=0)
      #------------------------------------
      # First calculate the mean and range
      #------------------------------------
      batch_no = 0
      count = 0.
      for input_batch, target_batch, masks, bdy_masks in mean_std_loader:
          
          # SHAPE: (batch_size, channels, y, x)

          # Calc mean of this batch
          batch_inputs_mean  = np.mean(input_batch.numpy(), axis = (0,2,3))
 
          #combine with previous mean and calc min and max
          if batch_no==0:
             inputs_mean = batch_inputs_mean  
             inputs_min  = np.amin( input_batch.numpy(), axis = (0,2,3) )
             inputs_max  = np.amax( input_batch.numpy(), axis = (0,2,3) )
          else:
             inputs_mean = 1./(count+input_batch.shape[0]) * ( count*inputs_mean + input_batch.shape[0]*batch_inputs_mean )
             inputs_min  = np.minimum( inputs_min, np.amin(input_batch.numpy()) )
             inputs_max  = np.maximum( inputs_max, np.amax(input_batch.numpy()) )

          batch_no = batch_no+1
          count = count + input_batch.shape[0]

      ## combine across all depth levels in each field (samples sizes are the same across all depths, so don't need to account for this)
      for i in range(3):  # Temp, U, V. No need to do for Eta as already 2-d/1 channel. Sal not included in model as const.
         inputs_mean[i*no_depth_levels:(i+1)*no_depth_levels]  = 1./no_depth_levels * np.sum(inputs_mean[i*no_depth_levels:(i+1)*no_depth_levels])
         inputs_min[ i*no_depth_levels:(i+1)*no_depth_levels]  = np.amin(inputs_min[i*no_depth_levels:(i+1)*no_depth_levels])
         inputs_max[ i*no_depth_levels:(i+1)*no_depth_levels]  = np.amax(inputs_max[i*no_depth_levels:(i+1)*no_depth_levels])

      inputs_range = inputs_max - inputs_min

      #-------------------------
      # Next calculate the std 
      #-------------------------
      count = 0.
      for input_batch, output_batch, masks, bdy_masks in mean_std_loader:
          
          # SHAPE: (batch_size, channels, y, x)
   
          inputs_dfm = np.zeros((input_batch.shape[1]))
          # Calc diff from mean for this batch
          for channel in range(input_batch.shape[1]):
             inputs_dfm[channel]  = inputs_dfm[channel] + np.sum( np.square( input_batch.numpy()[:,channel,:,:]-inputs_mean[channel] ) )
          count = count + input_batch.shape[0]*input_batch.shape[2]*input_batch.shape[3]
   
      ## sum across all depth levels in each field and calc std
      for i in range(3):  # Temp, U, V. Sal not included in model as const. Eta done below as already 2d channel.
         inputs_dfm[i*no_depth_levels:(i+1)*no_depth_levels]  = np.sum(inputs_dfm[i*no_depth_levels:(i+1)*no_depth_levels])

      inputs_var = inputs_dfm / (no_depth_levels*count)
       
      inputs_std = np.sqrt(inputs_var)
   
      #-------------------------

   elif dimension == '3d':
      mean_std_Dataset = rr.MITGCM_Dataset_3d( MITGCM_filename, 0.0, train_end_ratio, subsample_rate, 1, land, tic,
                                               transform=None)    # Use histlen=1 for this, as doesn't make much impact
      mean_std_loader = data.DataLoader(mean_std_Dataset, batch_size=batch_size, shuffle=False, num_workers=0)
      
      #------------------------------------
      # First calculate the mean and range
      #------------------------------------
      batch_no = 0
      count = 0.
      for input_batch, target_batch, masks, bdy_masks in mean_std_loader:
          
          # SHAPE: (batch_size, channels, z, y, x)
   
          # Calc mean of this batch
          batch_inputs_mean  = np.mean(input_batch.numpy(), axis = (0,2,3,4))
    
          #combine with previous mean and calc min and max
          if batch_no==0:
             inputs_mean = batch_inputs_mean  
             inputs_min  = np.amin( input_batch.numpy(), axis = (0,2,3,4) )
             inputs_max  = np.amax( input_batch.numpy(), axis = (0,2,3,4) )
          else:
             inputs_mean = 1./(count+input_batch.shape[0]) * ( count*inputs_mean + input_batch.shape[0]*batch_inputs_mean )
             inputs_min  = np.minimum( inputs_min, np.amin(input_batch.numpy()) )
             inputs_max  = np.maximum( inputs_max, np.amax(input_batch.numpy()) )
   
          batch_no = batch_no+1
          count = count + input_batch.shape[0]
   
      inputs_range = inputs_max - inputs_min
   
      #-------------------------
      # Next calculate the std 
      #-------------------------
      count = 0.
      for input_batch, output_batch, masks, bdy_masks in mean_std_loader:
          
          # SHAPE: (batch_size, channels, z, y, x)
   
          inputs_dfm = np.zeros((input_batch.shape[1]))
          # Calc diff from mean for this batch
          for channel in range(input_batch.shape[1]):
             inputs_dfm[channel]  = inputs_dfm[channel] + np.sum( np.square( input_batch.numpy()[:,channel,:,:,:]-inputs_mean[channel] ) )
          count = count + input_batch.shape[0]*input_batch.shape[2]*input_batch.shape[3]*input_batch.shape[4]
   
      inputs_var = inputs_dfm / count
       
      inputs_std = np.sqrt(inputs_var)
   
      #-------------------------

   ## Save to file, so can be used to un-normalise when using model to predict
   mean_std_file = '../../../Channel_nn_Outputs/'+MeanStd_prefix+'_MeanStd.npz'
   np.savez( mean_std_file, inputs_mean, inputs_std, inputs_range)
   
def ReadMeanStd(MeanStd_prefix):
   mean_std_file = '../../../Channel_nn_Outputs/'+MeanStd_prefix+'_MeanStd.npz'
   mean_std_data = np.load(mean_std_file)
   inputs_mean  = mean_std_data['arr_0']
   inputs_std   = mean_std_data['arr_1']
   inputs_range = mean_std_data['arr_2']

   return(inputs_mean, inputs_std, inputs_range)

def my_loss(output, target, bdy_mask):
    #loss = torch.mean( (output - target)**2 * bdy_mask )
    loss = torch.mean( (output - target)**2 * (bdy_mask-1) )
    return loss

def TrainModel(model_name, dimension, histlen, tic, TEST, no_tr_samples, no_val_samples, save_freq, train_loader, val_loader,
               h, optimizer, num_epochs, seed_value, losses, no_in_channels, no_out_channels, start_epoch=1, current_best_loss=0.1):

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

   histogram_data=[ np.zeros((0)),np.zeros((0)),np.zeros((0)),np.zeros((0)) ]

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

       for input_batch, target_batch, masks, bdy_masks in train_loader:

           input_batch = input_batch.to(device, non_blocking=True, dtype=torch.float)
           masks = masks.to(device, non_blocking=True, dtype=torch.float)
           bdy_masks = bdy_masks.to(device, non_blocking=True, dtype=torch.float)
           # mask targets to circumvent issues with normalisation!
           target_batch = target_batch.to(device, non_blocking=True, dtype=torch.float) * masks
           # input_batch SHAPE: (no_samples, no_channels, z, y, x) if 3d, (no_samples, no_channels, y, x) if 2d.
   
           # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
           optimizer.zero_grad()
       
           # get prediction from the model, given the inputs
           predicted_batch = h( torch.cat((input_batch, masks), dim=1) ) * masks
   
           logging.debug('For test run check if output is correct shape, the following two shapes should match!\n')
           logging.debug('target_batch.shape for training data: '+str(target_batch.shape)+'\n')
           logging.debug('predicted_batch.shape for training data: '+str(predicted_batch.shape)+'\n')
           logging.debug('masks.shape : '+str(masks.shape)+'\n')
              
           # Calculate and update loss values
           loss = my_loss(predicted_batch, target_batch, bdy_masks).double()
           losses['train'][-1] = losses['train'][-1] + loss.item() * input_batch.shape[0]
            
           if dimension == '2d': 
              losses['train_Temp'][-1] = losses['train_Temp'][-1] + \
                                         my_loss( predicted_batch[:,:no_depth_levels,:,:],
                                         target_batch[:,:no_depth_levels,:,:],
                                         bdy_masks[:,:no_depth_levels,:,:] ).item() * input_batch.shape[0]
              losses['train_U'][-1] = losses['train_U'][-1] + \
                                      my_loss( predicted_batch[:,no_depth_levels:2*no_depth_levels,:,:],
                                      target_batch[:,no_depth_levels:2*no_depth_levels,:,:],
                                      bdy_masks[:,no_depth_levels:2*no_depth_levels,:,:] ).item() * input_batch.shape[0]
              losses['train_V'][-1] = losses['train_V'][-1] + \
                                      my_loss( predicted_batch[:,2*no_depth_levels:3*no_depth_levels,:,:],
                                      target_batch[:,2*no_depth_levels:3*no_depth_levels,:,:],
                                      bdy_masks[:,2*no_depth_levels:3*no_depth_levels,:,:] ).item() * input_batch.shape[0]
              losses['train_Eta'][-1] = losses['train_Eta'][-1] + \
                                        my_loss( predicted_batch[:,3*no_depth_levels,:,:],
                                        target_batch[:,3*no_depth_levels,:,:],
                                        bdy_masks[:,3*no_depth_levels,:,:] ).item() * input_batch.shape[0]
           elif dimension == '3d': 
              losses['train_Temp'][-1] = losses['train_Temp'][-1] + \
                                         my_loss( predicted_batch[:,0,:,:,:], target_batch[:,0,:,:,:] ).item() * input_batch.shape[0]
              losses['train_U'][-1]    = losses['train_U'][-1] + \
                                         my_loss( predicted_batch[:,1,:,:,:], target_batch[:,1,:,:,:] ).item() * input_batch.shape[0]
              losses['train_V'][-1]    = losses['train_V'][-1] + \
                                         my_loss( predicted_batch[:,2,:,:,:], target_batch[:,2,:,:,:] ).item() * input_batch.shape[0]
              losses['train_Eta'][-1]  = losses['train_Eta'][-1] + \
                                         my_loss( predicted_batch[:,3,:,:,:], target_batch[:,3,0,:,:] ).item() * input_batch.shape[0]
  
           # get gradients w.r.t to parameters
           loss.backward()
        
           # update parameters
           optimizer.step()
   
           ## Doesn't work in current form with histlen changes to ordering of data - need to redo.
           #if epoch == start_epoch and batch_no == 0:
           #   # Save info to plot histogram of data
           #   if dimension == '2d':
           #      for var in range(3):
           #         for time in range(histlen):
           #            # no_out_channels = number of physical variable channels at each time step
           #            histogram_data[var] = np.concatenate(( histogram_data[var], 
           #                input_batch[:,time*no_out_channels+no_depth_levels*var:time*no_out_channels+no_depth_levels*(var+1),:,:]
           #                            .cpu().detach().numpy().reshape(-1) ))
           #      for time in range(histlen):
           #         histogram_data[3] = np.concatenate(( histogram_data[var],
           #                input_batch[:,time*no_out_channels+no_depth_levels*3,:,:].cpu().detach().numpy().reshape(-1) ))
           #   elif dimension == '3d':
           #      for var in range(3):
           #         for time in range(histlen):
           #            histogram_data[var] = np.concatenate(( histogram_data[var],
           #                input_batch[:,time*no_out_channels+var,:,:,:].cpu().detach().numpy().reshape(-1) ))
           #      for time in range(histlen):
           #         histogram_data[3] = np.concatenate(( histogram_data[var],
           #                input_batch[:,time*no_out_channels+3,0,:,:].cpu().detach().numpy().reshape(-1) ))

           del input_batch
           del target_batch
           del predicted_batch
           gc.collect()
           torch.cuda.empty_cache()

           batch_no = batch_no + 1

       losses['train'][-1] = losses['train'][-1] / no_tr_samples
       losses['train_Temp'][-1] = losses['train_Temp'][-1] / no_tr_samples
       losses['train_U'][-1] = losses['train_U'][-1] / no_tr_samples
       losses['train_V'][-1] = losses['train_V'][-1] / no_tr_samples
       losses['train_Eta'][-1] = losses['train_Eta'][-1] / no_tr_samples
  
       logging.info('epoch {}, training loss {}'.format(epoch, losses['train'][-1])+'\n')
       #TimeCheck(tic, 'finished epoch training')

       #### Validation ######
   
       batch_no = 0
       
       h.train(False)

       for input_batch, target_batch, masks, bdy_masks in val_loader:
   
           # Send inputs and labels to GPU if available
           input_batch = input_batch.to(device, non_blocking=True, dtype=torch.float)
           bdy_masks = bdy_masks.to(device, non_blocking=True, dtype=torch.float)
           masks = masks.to(device, non_blocking=True, dtype=torch.float)
           target_batch = target_batch.to(device, non_blocking=True, dtype=torch.float) * masks
           
           # get prediction from the model, given the inputs
           predicted_batch = h( torch.cat((input_batch, masks), dim=1) ) * masks
       
           # get loss for the predicted_batch output
           losses['val'][-1] = losses['val'][-1] +  \
                               my_loss( predicted_batch, target_batch, bdy_masks ).item() * input_batch.shape[0]
   
           del input_batch
           del target_batch
           del predicted_batch
           gc.collect()
           torch.cuda.empty_cache()

           batch_no = batch_no + 1
   
       losses['val'][-1] = losses['val'][-1] / no_val_samples
       logging.info('epoch {}, validation loss {}'.format(epoch, losses['val'][-1])+'\n')
  
       # Save model if this is the best version of the model 
       if losses['val'][-1] < current_best_loss :
           ### SAVE IT ###
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

       # Save model if this is the best version of the model 
       if ( epoch%save_freq == 0 or epoch == num_epochs+start_epoch-1 ) and epoch != start_epoch :
           ### Save Model ###
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
   return(losses, histogram_data)

def PlotScatter(model_name, dimension, data_loader, h, epoch, title, no_out_channels, MeanStd_prefix):

   logging.info('Making scatter plots')

   data_mean, data_std, data_range = ReadMeanStd(MeanStd_prefix)

   no_depth_levels = 38  # Hard coded...perhaps should change...?

   device = 'cuda' if torch.cuda.is_available() else 'cpu'

   h.train(False)

   with torch.no_grad():
      input_batch, target_batch, masks, bdy_masks = next(iter(data_loader))
      input_batch = input_batch.to(device, non_blocking=True, dtype=torch.float)
      predicted_batch = h( torch.cat( ( input_batch, masks.to(device, non_blocking=True, dtype=torch.float) ), dim=1) ).cpu().detach() * masks

   # Denormalise and mask
   target_batch = rr.RF_DeNormalise(target_batch, data_mean, data_std, data_range, no_out_channels, dimension) * masks
   predicted_batch = rr.RF_DeNormalise(predicted_batch, data_mean, data_std, data_range, no_out_channels, dimension) * masks

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
      targets = target_batch[:,:,:,:,:]
      predictions = predicted_batch[:,:,:,:,:]

   fig_main = plt.figure(figsize=(9,9))
   ax_temp  = fig_main.add_subplot(221)
   ax_U     = fig_main.add_subplot(222)
   ax_V     = fig_main.add_subplot(223)
   ax_Eta   = fig_main.add_subplot(224)

   ax_temp.scatter(targets[:,0,:,:,:],
                   predictions[:,0,:,:,:],
                   edgecolors=(0, 0, 0), alpha=0.15, color='blue', label='Temp')
   ax_U.scatter(targets[:,1,:,:,:],
                predictions[:,1,:,:,:],
                edgecolors=(0, 0, 0), alpha=0.15, color='red', label='U')
   ax_V.scatter(targets[:,2,:,:,:], 
                predictions[:,2,:,:,:],
                edgecolors=(0, 0, 0), alpha=0.15, color='orange', label='V')
   ax_Eta.scatter(targets[:,3,0,:,:], 
                  predictions[:,3,0,:,:],
                  edgecolors=(0, 0, 0), alpha=0.15, color='purple', label='Eta')
   bottom = min( np.amin(targets), np.amin(predictions) )
   top    = max( np.amax(targets), np.amax(predictions) )  
   
   ax_temp.set_xlabel('Truth')
   ax_temp.set_ylabel('Predictions')
   ax_temp.set_title('Temperature')
   ax_temp.set_xlim(bottom, top)
   ax_temp.set_ylim(bottom, top)
   ax_temp.plot([bottom, top], [bottom, top], 'k--', lw=1, color='black')

   ax_U.set_xlabel('Truth')
   ax_U.set_ylabel('Predictions')
   ax_U.set_title('U velocity')
   ax_U.set_xlim(bottom, top)
   ax_U.set_ylim(bottom, top)
   ax_U.plot([bottom, top], [bottom, top], 'k--', lw=1, color='black')

   ax_V.set_xlabel('Truth')
   ax_V.set_ylabel('Predictions')
   ax_V.set_title('V velocity')
   ax_V.set_xlim(bottom, top)
   ax_V.set_ylim(bottom, top)
   ax_V.plot([bottom, top], [bottom, top], 'k--', lw=1, color='black')

   ax_Eta.set_xlabel('Truth')
   ax_Eta.set_ylabel('Predictions')
   ax_Eta.set_title('Sea Surface Height')
   ax_Eta.set_xlim(bottom, top)
   ax_Eta.set_ylim(bottom, top)
   ax_Eta.plot([bottom, top], [bottom, top], 'k--', lw=1, color='black')

   plt.savefig('../../../Channel_nn_Outputs/'+model_name+'/PLOTS/'+model_name+'_scatter_epoch'+str(epoch).rjust(3,'0')+'_'+title+'.png',
               bbox_inches = 'tight', pad_inches = 0.1)
   plt.close()


def LoadModel(model_name, h, optimizer, saved_epoch, tr_inf, losses, best):
   if best:
      pkl_filename = '../../../Channel_nn_Outputs/'+model_name+'/MODELS/'+model_name+'_epoch'+str(saved_epoch)+'_SavedBESTModel.pt'
   else:
      pkl_filename = '../../../Channel_nn_Outputs/'+model_name+'/MODELS/'+model_name+'_epoch'+str(saved_epoch)+'_SavedModel.pt'
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

def plot_training_output(model_name, start_epoch, total_epochs, plot_freq, losses, histogram_data):
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
   plt.savefig(plot_dir+'/'+model_name+'_TrainingValLossPerEpoch_epoch'+str(total_epochs)+'.eps',
               bbox_inches = 'tight', pad_inches = 0.1)
   plt.close()
 
   # plotting histograms of data
   no_bins = 50
   labels = ['Temp', 'U_vel', 'V_vel', 'Eta']
   for var in range(4):
       fig_histogram = plt.figure(figsize=(10, 8))
       plt.hist(histogram_data[var], bins = no_bins)
       plt.savefig(plot_dir+'/'+model_name+'_histogram_'+str(labels[var])+'.png', 
                   bbox_inches = 'tight', pad_inches = 0.1)
       plt.close()

def OutputStats(model_name, MeanStd_prefix, mitgcm_filename, data_loader, h, no_epochs, y_dim_used, dimension, 
                histlen, no_in_channels, no_out_channels, land, file_append):
   #-------------------------------------------------------------
   # Output the rms and correlation coefficients over a dataset
   #-------------------------------------------------------------
   #  1. Make predictions over entire dataset (training, val or test), summing Mean Squared Error 
   #     as we go
   #  2. Store predictions and targets for a small number of smaples in an array
   #  3. Calculate the RMS of the dataset from the summed squared error
   #  4. Store the RMS and subset of samples in a netcdf file

   device = 'cuda' if torch.cuda.is_available() else 'cpu'

   data_mean, data_std, data_range = ReadMeanStd(MeanStd_prefix)

   # Read in grid data from MITgcm file
   mitgcm_ds = xr.open_dataset(mitgcm_filename)
   da_X = mitgcm_ds['X']
   da_Y = mitgcm_ds['Yp1']
   da_Z = mitgcm_ds['Zmd000038'] 

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
      for input_batch, target_batch, masks, bdy_masks in data_loader:
          
          input_batch = input_batch.to(device, non_blocking=True, dtype=torch.float)
          target_batch = target_batch.cpu().detach().numpy()

          # get prediction from the model, given the inputs
          if isinstance(h, list):
             predicted_batch = h[0](input_batch) 
             for i in range(1, len(h)):
                predicted_batch = predicted_batch + h[i]( torch.cat((input_batch, masks.to(device, non_blocking=True, dtype=torch.float)), dim=1) )
             predicted_batch = predicted_batch / len(h)
          else:
             predicted_batch = h( torch.cat((input_batch, masks.to(device, non_blocking=True, dtype=torch.float)), dim=1) ) 
          predicted_batch = predicted_batch.cpu().detach().numpy() 

          # Denormalise and mask
          target_batch = rr.RF_DeNormalise(target_batch, data_mean, data_std, data_range, no_out_channels, dimension) * masks.numpy()
          predicted_batch = rr.RF_DeNormalise(predicted_batch, data_mean, data_std, data_range, no_out_channels, dimension) * masks.numpy()

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

   # Set up netcdf files
   nc_filename = '../../../Channel_nn_Outputs/'+model_name+'/STATS/'+model_name+'_'+str(no_epochs)+'epochs_StatsOutput_'+file_append+'.nc'
   nc_file = nc4.Dataset(nc_filename,'w', format='NETCDF4') #'w' stands for write
   # Create Dimensions
   no_depth_levels = 38  # Hard coded...perhaps should change...?
   nc_file.createDimension('T', no_samples)
   nc_file.createDimension('Z', da_Z.values.shape[0])
   nc_file.createDimension('Y', y_dim_used)  #da_Y.values.shape[0])
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

   nc_TempMask   = nc_file.createVariable( 'Temp_Mask'  , 'f4', ('Z', 'Y', 'X') )
   nc_UMask      = nc_file.createVariable( 'U_Mask'     , 'f4', ('Z', 'Y', 'X') )
   nc_VMask      = nc_file.createVariable( 'V_Mask'     , 'f4', ('Z', 'Y', 'X') )
   nc_EtaMask    = nc_file.createVariable( 'Eta_Mask'   , 'f4', ('Y', 'X')      )

   nc_TempBdyMask   = nc_file.createVariable( 'Temp_Bdy_Mask'  , 'f4', ('Z', 'Y', 'X') )
   nc_UBdyMask      = nc_file.createVariable( 'U_Bdy_Mask'     , 'f4', ('Z', 'Y', 'X') )
   nc_VBdyMask      = nc_file.createVariable( 'V_Bdy_Mask'     , 'f4', ('Z', 'Y', 'X') )
   nc_EtaBdyMask    = nc_file.createVariable( 'Eta_Bdy_Mask'   , 'f4', ('Y', 'X')      )

   nc_PredTemp   = nc_file.createVariable( 'Pred_Temp'  , 'f4', ('T', 'Z', 'Y', 'X') )
   nc_PredU      = nc_file.createVariable( 'Pred_U'     , 'f4', ('T', 'Z', 'Y', 'X') )
   nc_PredV      = nc_file.createVariable( 'Pred_V'     , 'f4', ('T', 'Z', 'Y', 'X') ) 
   nc_PredEta    = nc_file.createVariable( 'Pred_Eta'   , 'f4', ('T', 'Y', 'X')      )

   nc_TempErrors = nc_file.createVariable( 'Temp_Errors', 'f4', ('T', 'Z', 'Y', 'X') )
   nc_UErrors    = nc_file.createVariable( 'U_Errors'   , 'f4', ('T', 'Z', 'Y', 'X') )
   nc_VErrors    = nc_file.createVariable( 'V_Errors'   , 'f4', ('T', 'Z', 'Y', 'X') )
   nc_EtaErrors  = nc_file.createVariable( 'Eta_Errors' , 'f4', ('T', 'Y', 'X')      )

   nc_TempRMS    = nc_file.createVariable( 'TempRMS'    , 'f4', ('Z', 'Y', 'X')      )
   nc_U_RMS      = nc_file.createVariable( 'U_RMS'      , 'f4', ('Z', 'Y', 'X')      )
   nc_V_RMS      = nc_file.createVariable( 'V_RMS'      , 'f4', ('Z', 'Y', 'X')      )
   nc_EtaRMS     = nc_file.createVariable( 'EtaRMS'     , 'f4', ('Y', 'X')           )

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

      nc_TempMask[:,:,:] = masks[0,0:no_depth_levels,:,:]
      nc_UMask[:,:,:]    = masks[0,1*no_depth_levels:2*no_depth_levels,:,:]
      nc_VMask[:,:,:]    = masks[0,2*no_depth_levels:3*no_depth_levels,:,:]
      nc_EtaMask[:,:]    = masks[0,3*no_depth_levels,:,:]

      nc_TempBdyMask[:,:,:] = bdy_masks[0,0:no_depth_levels,:,:]
      nc_UBdyMask[:,:,:]    = bdy_masks[0,1*no_depth_levels:2*no_depth_levels,:,:]
      nc_VBdyMask[:,:,:]    = bdy_masks[0,2*no_depth_levels:3*no_depth_levels,:,:]
      nc_EtaBdyMask[:,:]    = bdy_masks[0,3*no_depth_levels,:,:]

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

   elif dimension=='3d':
      nc_TrueTemp[:,:,:,:] = targets[:no_samples,0,:,:,:] 
      nc_TrueU[:,:,:,:]    = targets[:no_samples,1,:,:,:]
      nc_TrueV[:,:,:,:]    = targets[:no_samples,2,:,:,:]
      nc_TrueEta[:,:,:]    = targets[:no_samples,3,0,:,:]

      nc_TempMask[:,:,:] = masks[0,0,:,:,:]
      nc_UMask[:,:,:]    = masks[0,1,:,:,:]
      nc_VMask[:,:,:]    = masks[0,2,:,:,:]
      nc_EtaMask[:,:]    = masks[0,3,0,:,:]

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


def IterativelyPredict(model_name, MeanStd_prefix, mitgcm_filename, Iterate_Dataset, h, start, for_len, no_epochs,
                       y_dim_used, land, dimension, histlen, no_in_channels, no_out_channels):

   device = 'cuda' if torch.cuda.is_available() else 'cpu'

   data_mean, data_std, data_range = ReadMeanStd(MeanStd_prefix)

   no_depth_levels = 38  # Hard coded...perhaps should change...?

   # Read in grid data from MITgcm file
   mitgcm_ds = xr.open_dataset(mitgcm_filename)
   da_X = mitgcm_ds['X']
   da_Y = mitgcm_ds['Yp1']
   da_Z = mitgcm_ds['Zmd000038'] 

   if land == 'ExcLand':
      da_Y = da_Y[3:]
  
   if isinstance(h, list):
      for i in range(len(h)):
         h[i].train(False)
         if torch.cuda.is_available():
            h[i] = h[i].cuda()
   else:
      h.train(False)
      if torch.cuda.is_available():
         h = h.cuda()

   # Read in first sample from MITgcm dataset and save as first entry in both arrays
   input_sample, target_sample, masks, bdy_masks = Iterate_Dataset.__getitem__(start)
   input_sample = input_sample.unsqueeze(0).numpy()
   masks = np.expand_dims(masks, axis=0)
   if dimension == '2d':
      #input_sample = input_sample[:,:,:].unsqueeze(0).numpy()
      iterated_predictions = input_sample[:,:no_depth_levels*3+1,:,:]
      MITgcm_data = input_sample[:,:no_depth_levels*3+1,:,:]
      for time in range(0,histlen):
         iterated_predictions = np.concatenate( (iterated_predictions, 
                                input_sample[:,(no_depth_levels*3+1)*time:(no_depth_levels*3+1)*(time+1),:,:]), axis=0)
         MITgcm_data = np.concatenate( (MITgcm_data, 
                                input_sample[:,(no_depth_levels*3+1)*time:(no_depth_levels*3+1)*(time+1),:,:]), axis=0)
   elif dimension == '3d':
      #input_sample = input_sample[:,:,:,:].unsqueeze(0).numpy()
      iterated_predictions = input_sample[:,:4,:,:]
      MITgcm_data = input_sample[:,:4,:,:]
      for time in range(1,histlen):
         iterated_predictions = np.concatenate( (iterated_predictions, input_sample[:,4*time:4*(time+1),:,:]), axis=0)
         MITgcm_data = np.concatenate( (MITgcm_data, input_sample[:,4*time:4*(time+1),:,:]), axis=0)

   # Make iterative forecast, saving predictions as we go, and pull out MITgcm data
   with torch.no_grad():
      for time in range(for_len):

         # Make prediction
         # multi model ensemble, or single model predictions
         if isinstance(h, list):
            predicted = h[0]( torch.from_numpy( np.concatenate((input_sample, masks), axis=1) )
                              .to(device, non_blocking=True, dtype=torch.float) ).cpu().detach().numpy()
            for i in range(1, len(h)):
               predicted = predicted + \
                           h[i]( torch.from_numpy( np.concatenate((input_sample, masks), axis=1) )
                                 .to(device, non_blocking=True, dtype=torch.float) ).cpu().detach().numpy()
            predicted = predicted / len(h)
         else:
            predicted = h( torch.from_numpy( np.concatenate((input_sample, masks), axis=1) )
                           .to(device, non_blocking=True, dtype=torch.float) ).cpu().detach().numpy()

         # Remask land on outputs ready for next iteration
         # Denormalise and mask
         predicted = rr.RF_DeNormalise(predicted, data_mean, data_std, data_range, no_out_channels, dimension) * masks
         # Re-Normalise
         predicted = rr.RF_ReNormalise_predicted(predicted, data_mean, data_std, data_range, no_out_channels, dimension)

         # Cat prediction onto existing predicted fields
         iterated_predictions = np.concatenate( ( iterated_predictions, predicted ), axis=0 )

         # Get MITgcm data for relevant step, and prep new input sample
         if dimension == '2d':
            MITgcm_data = np.concatenate( ( MITgcm_data, Iterate_Dataset.__getitem__(start+time+1)[1]
                                            [(histlen-1)*(no_depth_levels*3+1):histlen*(no_depth_levels*3+1),:,:].unsqueeze(0).numpy() ),
                                            axis=0 )
            input_sample = np.zeros((1, histlen*3*no_depth_levels+histlen, y_dim_used, da_X.shape[0]))
            for time in range(histlen):
               input_sample[0,time*(3*no_depth_levels+1):(time+1)*(3*no_depth_levels+1),:,:]=iterated_predictions[-histlen+time,:,:,:]

         elif dimension == '3d':
            MITgcm_data = np.concatenate( ( MITgcm_data, 
                                            Iterate_Dataset.__getitem__(start+time+1)[1][(histlen-1)*4:histlen*4,:,:,:].unsqueeze(0).numpy() ),
                                            axis=0 )
            input_sample = np.zeros((1, histlen*4, da_Z.shape[0], y_dim_used, da_X.shape[0]))
            for time in range(histlen):
               input_sample[0,time*4:(time+1)*4,:,:,:]=iterated_predictions[-histlen+time,:,:,:,:]

         del predicted
         gc.collect()
         torch.cuda.empty_cache()

   logging.debug('iterated_predictions.shape ; '+str(iterated_predictions.shape))
   logging.debug('MITgcm_data.shape ; '+str(MITgcm_data.shape))

   ## Denormalise and mask
   iterated_predictions = rr.RF_DeNormalise(iterated_predictions, data_mean, data_std, data_range, no_out_channels, dimension) * masks
   MITgcm_data          = rr.RF_DeNormalise(MITgcm_data,          data_mean, data_std, data_range, no_out_channels, dimension) * masks

   # Set up netcdf files
   nc_filename = '../../../Channel_nn_Outputs/'+model_name+'/ITERATED_FORECAST/'+model_name+'_'+str(no_epochs)+'epochs_Forecast'+str(for_len)+'.nc'
   nc_file = nc4.Dataset(nc_filename,'w', format='NETCDF4') #'w' stands for write
   # Create Dimensions
   nc_file.createDimension('T', MITgcm_data.shape[0])
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

   # Fill variables
   nc_T[:] = np.arange(iterated_predictions.shape[0])
   nc_Z[:] = da_Z.values
   nc_Y[:] = da_Y.values[:y_dim_used]
   nc_X[:] = da_X.values
 
   if dimension == '2d':
      nc_TempMask[:,:,:] = masks[0,0:no_depth_levels,:,:]
      nc_UMask[:,:,:]    = masks[0,1*no_depth_levels:2*no_depth_levels,:,:]
      nc_VMask[:,:,:]    = masks[0,2*no_depth_levels:3*no_depth_levels,:,:]
      nc_EtaMask[:,:]    = masks[0,3*no_depth_levels,:,:]

      nc_TrueTemp[:,:,:,:] = MITgcm_data[:,0:no_depth_levels,:,:] 
      nc_TrueU[:,:,:,:]    = MITgcm_data[:,1*no_depth_levels:2*no_depth_levels,:,:]
      nc_TrueV[:,:,:,:]    = MITgcm_data[:,2*no_depth_levels:3*no_depth_levels,:,:]
      nc_TrueEta[:,:,:]    = MITgcm_data[:,3*no_depth_levels,:,:]

      nc_PredTemp[:,:,:,:] = iterated_predictions[:,0:no_depth_levels,:,:]
      nc_PredU[:,:,:,:]    = iterated_predictions[:,1*no_depth_levels:2*no_depth_levels,:,:]
      nc_PredV[:,:,:,:]    = iterated_predictions[:,2*no_depth_levels:3*no_depth_levels,:,:]
      nc_PredEta[:,:,:]    = iterated_predictions[:,3*no_depth_levels,:,:]
      
      nc_TempErrors[:,:,:,:] = iterated_predictions[:,0:no_depth_levels,:,:] - MITgcm_data[:,0:no_depth_levels,:,:]
      nc_UErrors[:,:,:,:]    = iterated_predictions[:,1*no_depth_levels:2*no_depth_levels,:,:] - MITgcm_data[:,1*no_depth_levels:2*no_depth_levels,:,:]
      nc_VErrors[:,:,:,:]    = iterated_predictions[:,2*no_depth_levels:3*no_depth_levels,:,:] - MITgcm_data[:,2*no_depth_levels:3*no_depth_levels,:,:]
      nc_EtaErrors[:,:,:]    = iterated_predictions[:,3*no_depth_levels,:,:] - MITgcm_data[:,3*no_depth_levels,:,:]

   elif dimension == '3d':
      nc_TempMask[:,:,:] = masks[0,0,:,:,:]
      nc_UMask[:,:,:]    = masks[0,1,:,:,:]
      nc_VMask[:,:,:]    = masks[0,2,:,:,:]
      nc_EtaMask[:,:]    = masks[0,3,0,:,:]

      nc_TrueTemp[:,:,:,:] = MITgcm_data[:,0,:,:,:] 
      nc_TrueU[:,:,:,:]    = MITgcm_data[:,1,:,:,:]
      nc_TrueV[:,:,:,:]    = MITgcm_data[:,2,:,:,:]
      nc_TrueEta[:,:,:]    = MITgcm_data[:,3,0,:,:]

      nc_PredTemp[:,:,:,:] = iterated_predictions[:,0,:,:,:]
      nc_PredU[:,:,:,:]    = iterated_predictions[:,1,:,:,:]
      nc_PredV[:,:,:,:]    = iterated_predictions[:,2,:,:,:]
      nc_PredEta[:,:,:]    = iterated_predictions[:,3,0,:,:]
      
      nc_TempErrors[:,:,:,:] = iterated_predictions[:,0,:,:,:] - MITgcm_data[:,0,:,:,:]
      nc_UErrors[:,:,:,:]    = iterated_predictions[:,1,:,:,:] - MITgcm_data[:,1,:,:,:]
      nc_VErrors[:,:,:,:]    = iterated_predictions[:,2,:,:,:] - MITgcm_data[:,2,:,:,:]
      nc_EtaErrors[:,:,:]    = iterated_predictions[:,3,0,:,:] - MITgcm_data[:,3,0,:,:]
