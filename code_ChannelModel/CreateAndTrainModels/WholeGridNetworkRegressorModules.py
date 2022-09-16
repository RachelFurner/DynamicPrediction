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

def CalcMeanStd(MeanStd_prefix, MITGCM_filename, train_end_ratio, subsample_rate,
                batch_size, land, dimension, bdy_weight, histlen, grid_filename, tic):

   no_depth_levels = 38 ## Note this is hard coded!!

   if dimension == '2d':
      mean_std_Dataset = rr.MITGCM_Dataset_2d( MITGCM_filename, 0.0, train_end_ratio, subsample_rate, 1, land, tic, bdy_weight,
                                               np.zeros(( histlen*(3*no_depth_levels+1) )), grid_filename,  transform=None) # Use histlen=1 for this, shouldn't impact it
      mean_std_loader = data.DataLoader(mean_std_Dataset, batch_size=batch_size, shuffle=False, num_workers=0)
      #------------------------------------
      # First calculate the mean and range
      #------------------------------------
      batch_no = 0
      count = 0.
      for input_batch, target_batch, masks in mean_std_loader:
          
          # Mask with Nans
          input_batch = np.where(masks == 1, input_batch, np.nan)
          target_batch = np.where(masks == 1, target_batch, np.nan)
 
          # SHAPE: (batch_size, channels, y, x)

          # Calc mean of this batch
          batch_inputs_mean  = np.nanmean(input_batch, axis = (0,2,3))
          batch_targets_mean  = np.nanmean(target_batch, axis = (0,2,3))
 
          #combine with previous mean and calc min and max
          if batch_no==0:
             inputs_mean = batch_inputs_mean  
             inputs_min  = np.nanmin( input_batch, axis = (0,2,3) )
             inputs_max  = np.nanmax( input_batch, axis = (0,2,3) )
             targets_mean = batch_targets_mean  
             targets_min  = np.nanmin( target_batch, axis = (0,2,3) )
             targets_max  = np.nanmax( target_batch, axis = (0,2,3) )
          else:
             inputs_mean = 1./(count+input_batch.shape[0]) * ( count*inputs_mean + input_batch.shape[0]*batch_inputs_mean )
             inputs_min  = np.minimum( inputs_min, np.nanmin(input_batch, axis=(0,2,3) ) )
             inputs_max  = np.maximum( inputs_max, np.nanmax(input_batch, axis=(0,2,3) ) )
             targets_mean = 1./(count+target_batch.shape[0]) * ( count*targets_mean + target_batch.shape[0]*batch_targets_mean )
             targets_min  = np.minimum( targets_min, np.nanmin(target_batch, axis=(0,2,3) ) )
             targets_max  = np.maximum( targets_max, np.nanmax(target_batch, axis=(0,2,3) ) )

          batch_no = batch_no+1
          count = count + input_batch.shape[0]

      ## combine across all depth levels in each field (samples sizes are the same across all depths, so don't need to account for this)
      for i in range(3):  # Temp, U, V. No need to do for Eta as already 2-d/1 channel. Sal not included in model as const.
         inputs_mean[i*no_depth_levels:(i+1)*no_depth_levels]  = 1./no_depth_levels * np.sum(inputs_mean[i*no_depth_levels:(i+1)*no_depth_levels])
         inputs_min [i*no_depth_levels:(i+1)*no_depth_levels]  = np.nanmin(inputs_min[i*no_depth_levels:(i+1)*no_depth_levels])
         inputs_max [i*no_depth_levels:(i+1)*no_depth_levels]  = np.nanmax(inputs_max[i*no_depth_levels:(i+1)*no_depth_levels])
         targets_mean[i*no_depth_levels:(i+1)*no_depth_levels] = 1./no_depth_levels * np.sum(targets_mean[i*no_depth_levels:(i+1)*no_depth_levels])
         targets_min [i*no_depth_levels:(i+1)*no_depth_levels] = np.nanmin(targets_min[i*no_depth_levels:(i+1)*no_depth_levels])
         targets_max [i*no_depth_levels:(i+1)*no_depth_levels] = np.nanmax(targets_max[i*no_depth_levels:(i+1)*no_depth_levels])

      inputs_range = inputs_max - inputs_min
      targets_range = targets_max - targets_min

      #------------------------
      # Next calculate the std 
      #------------------------
      count = 0.
      #for input_batch, target_batch, masks, bdy_masks in mean_std_loader:
      for input_batch, target_batch, masks in mean_std_loader:
          
          # Mask with Nans
          input_batch = np.where(masks == 1, input_batch, np.nan)
          target_batch = np.where(masks == 1, target_batch, np.nan)
 
          # SHAPE: (batch_size, channels, y, x)
   
          inputs_dfm = np.zeros((input_batch.shape[1]))
          targets_dfm = np.zeros((target_batch.shape[1]))
          # Calc diff from mean for this batch
          for channel in range(input_batch.shape[1]):
             inputs_dfm[channel]  = inputs_dfm[channel] + np.nansum( np.square( input_batch[:,channel,:,:]-inputs_mean[channel] ) )
          for channel in range(target_batch.shape[1]):
             targets_dfm[channel]  = targets_dfm[channel] + np.nansum( np.square( target_batch[:,channel,:,:]-targets_mean[channel] ) )
          #count = count + input_batch.shape[0]*input_batch.shape[2]*input_batch.shape[3]
          count = count + np.sum(~np.isnan(input_batch[:,0,:,:]))   # Calc number of non-Nan entries per channel
   
      ## sum across all depth levels in each field and divide by number of entries
      for i in range(3):  # Temp, U, V. Sal not included in model as const. Eta already 2d channel.
         inputs_dfm[i*no_depth_levels:(i+1)*no_depth_levels]  = np.nansum(inputs_dfm[i*no_depth_levels:(i+1)*no_depth_levels])/(no_depth_levels*count)
         targets_dfm[i*no_depth_levels:(i+1)*no_depth_levels]  = np.nansum(targets_dfm[i*no_depth_levels:(i+1)*no_depth_levels])/(no_depth_levels*count)
      inputs_dfm[3*no_depth_levels]  = np.nansum(inputs_dfm[3*no_depth_levels])/count
      targets_dfm[3*no_depth_levels] = np.nansum(targets_dfm[3*no_depth_levels])/count

      inputs_std = np.sqrt(inputs_dfm)
      targets_std = np.sqrt(targets_dfm)
   
      #-------------------------

   elif dimension == '3d':
      logging.info('Code not properly set up for 3d - Mean and Std still calculated over nextfield not DeltaT')
      sys.exit()
      mean_std_Dataset = rr.MITGCM_Dataset_3d( MITGCM_filename, 0.0, train_end_ratio, subsample_rate, 1, land, tic,
                                               transform=None)    # Use histlen=1 for this, as doesn't make much impact
      mean_std_loader = data.DataLoader(mean_std_Dataset, batch_size=batch_size, shuffle=False, num_workers=0)
      
      #------------------------------------
      # First calculate the mean and range
      #------------------------------------
      batch_no = 0
      count = 0.
      #for input_batch, target_batch, masks, bdy_masks in mean_std_loader:
      for input_batch, target_batch, masks in mean_std_loader:
          
          # SHAPE: (batch_size, channels, z, y, x)
   
          # Calc mean of this batch
          batch_inputs_mean  = np.mean(input_batch, axis = (0,2,3,4))
    
          #combine with previous mean and calc min and max
          if batch_no==0:
             inputs_mean = batch_inputs_mean  
             inputs_min  = np.amin( input_batch, axis = (0,2,3,4) )
             inputs_max  = np.amax( input_batch, axis = (0,2,3,4) )
          else:
             inputs_mean = 1./(count+input_batch.shape[0]) * ( count*inputs_mean + input_batch.shape[0]*batch_inputs_mean )
             inputs_min  = np.minimum( inputs_min, np.amin(input_batch) )
             inputs_max  = np.maximum( inputs_max, np.amax(input_batch) )
   
          batch_no = batch_no+1
          count = count + input_batch.shape[0]
   
      inputs_range = inputs_max - inputs_min
   
      #-------------------------
      # Next calculate the std 
      #-------------------------
      count = 0.
      #for input_batch, output_batch, masks, bdy_masks in mean_std_loader:
      for input_batch, output_batch, masks in mean_std_loader:
          
          # SHAPE: (batch_size, channels, z, y, x)
   
          inputs_dfm = np.zeros((input_batch.shape[1]))
          # Calc diff from mean for this batch
          for channel in range(input_batch.shape[1]):
             inputs_dfm[channel]  = inputs_dfm[channel] + np.sum( np.square( input_batch[:,channel,:,:,:]-inputs_mean[channel] ) )
          count = count + input_batch.shape[0]*input_batch.shape[2]*input_batch.shape[3]*input_batch.shape[4]
   
      inputs_var = inputs_dfm / count
       
      inputs_std = np.sqrt(inputs_var)
   
      #-------------------------

   ## Save to file, so can be used to un-normalise when using model to predict
   mean_std_file = '../../../Channel_nn_Outputs/'+MeanStd_prefix+'_MeanStd.npz'
   np.savez( mean_std_file, inputs_mean, inputs_std, inputs_range, targets_mean, targets_std, targets_range )
   
def ReadMeanStd(MeanStd_prefix):
   mean_std_file = '../../../Channel_nn_Outputs/'+MeanStd_prefix+'_MeanStd.npz'
   mean_std_data = np.load(mean_std_file)
   inputs_mean  = mean_std_data['arr_0']
   inputs_std   = mean_std_data['arr_1']
   inputs_range = mean_std_data['arr_2']
   targets_mean  = mean_std_data['arr_3']
   targets_std   = mean_std_data['arr_4']
   targets_range = mean_std_data['arr_5']

   return(inputs_mean, inputs_std, inputs_range, targets_mean, targets_std, targets_range)

def my_loss(output, target):
   loss = torch.mean( (output - target)**2 )
   return loss

# GAN attempts are a work in progress - nowhere near set up to test yet
def trainGAN_fn(train_loader, val_loader, disc, gen, loader, opt_disc, opt_gen, l1_loss, bce, g_scaler, d_scaler):

    disc = Discriminator(in_channels=no_in_channels).to(config.DEVICE)
    gen = Generator(in_channels=no_in_channels).to(config.DEVICE)
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999),)
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()

    #if config.LOAD_MODEL:
    #    load_checkpoint(
    #        config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE,
    #    )
    #    load_checkpoint(
    #        config.CHECKPOINT_DISC, disc, opt_disc, config.LEARNING_RATE,
    #    )

    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config.NUM_EPOCHS):

        for input_batch, target_batch, masks in train_loader:
            input_batch = input_batch.to(device, non_blocking=True, dtype=torch.float)
            target_batch = target_batch.to(device, non_blocking=True, dtype=torch.float) * masks
    
            # Train Discriminator
            with torch.cuda.amp.autocast():
                fake_target_batch = gen(input_batch)
                D_real = disc(input_batch, target_batch)
                D_real_loss = bce(D_real, torch.ones_like(D_real))
                D_fake = disc(input_batch, fake_target_batch.detach())
                D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
                D_loss = (D_real_loss + D_fake_loss) / 2
    
            disc.zero_grad()
            d_scaler.scale(D_loss).backward()
            d_scaler.step(opt_disc)
            d_scaler.update()
    
            # Train generator
            with torch.cuda.amp.autocast():
                D_fake = disc(input_batch, fake_target_batch)
                G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
                L1 = l1_loss(fake_target_batch, target_batch) * config.L1_LAMBDA
                G_loss = G_fake_loss + L1
    
            opt_gen.zero_grad()
            g_scaler.scale(G_loss).backward()
            g_scaler.step(opt_gen)
            g_scaler.update()

        #if config.SAVE_MODEL and epoch % 5 == 0:
        #    save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
        #    save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)

        #save_some_examples(gen, val_loader, epoch, folder="evaluation")

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

       #for input_batch, target_batch, masks, bdy_masks in train_loader:
       for input_batch, target_batch, masks in train_loader:

           input_batch = input_batch.to(device, non_blocking=True, dtype=torch.float)
           masks = masks.to(device, non_blocking=True, dtype=torch.float)
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
           loss = my_loss(predicted_batch, target_batch).double()
           losses['train'][-1] = losses['train'][-1] + loss.item() * input_batch.shape[0]
            
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
              logging.info('Code not properly set up for 3d - Mean and Std still calculated over nextfield not DeltaT')
              sys.exit()
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
           if epoch == start_epoch and batch_no == 0:
              # Save info to plot histogram of data
              if dimension == '2d':
                 for var in range(3):
                    histogram_data[var] = np.concatenate(( histogram_data[var], 
                        input_batch[:,no_depth_levels*var:no_depth_levels*(var+1),:,:]
                                    .cpu().detach().numpy().reshape(-1) ))
                 histogram_data[3] = np.concatenate(( histogram_data[3],
                        input_batch[:,no_depth_levels*3,:,:].cpu().detach().numpy().reshape(-1) ))
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

       #### Validation ######
   
       batch_no = 0
       
       h.train(False)

       #for input_batch, target_batch, masks, bdy_masks in val_loader:
       for input_batch, target_batch, masks in val_loader:
   
           # Send inputs and labels to GPU if available
           input_batch = input_batch.to(device, non_blocking=True, dtype=torch.float)
           masks = masks.to(device, non_blocking=True, dtype=torch.float)
           target_batch = target_batch.to(device, non_blocking=True, dtype=torch.float) * masks
           
           # get prediction from the model, given the inputs
           #predicted_batch = h( torch.cat((input_batch, masks, bdy_masks), dim=1) ) * masks
           predicted_batch = h( torch.cat((input_batch, masks), dim=1) ) * masks
       
           # get loss for the predicted_batch output
           losses['val'][-1] = losses['val'][-1] +  \
                               my_loss( predicted_batch, target_batch ).item() * input_batch.shape[0]
   
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
   return(losses, histogram_data)

def PlotScatter(model_name, dimension, data_loader, h, epoch, title, no_out_channels, MeanStd_prefix):

   logging.info('Making scatter plots')

   inputs_mean, inputs_std, inputs_range, targets_mean, targets_std, targets_range = ReadMeanStd(MeanStd_prefix)

   no_depth_levels = 38  # Hard coded...perhaps should change...?

   device = 'cuda' if torch.cuda.is_available() else 'cpu'

   h.train(False)

   with torch.no_grad():
      #input_batch, target_batch, masks, bdy_masks = next(iter(data_loader))
      input_batch, target_batch, masks = next(iter(data_loader))
      input_batch = input_batch.to(device, non_blocking=True, dtype=torch.float)
      predicted_batch = h( torch.cat(
                           ( input_batch,
                           masks.to(device, non_blocking=True, dtype=torch.float)
                           ), dim=1)
                         ).cpu().detach() * masks

   # Denormalise and mask
   target_batch = rr.RF_DeNormalise(target_batch, targets_mean, targets_std, targets_range, no_out_channels, dimension) * masks
   predicted_batch = rr.RF_DeNormalise(predicted_batch, targets_mean, targets_std, targets_range, no_out_channels, dimension) * masks

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
   plt.savefig(plot_dir+'/'+model_name+'_TrainingValLossPerEpoch_epoch'+str(total_epochs)+'.png',
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
   logging.info('Outputting stats')

   device = 'cuda' if torch.cuda.is_available() else 'cpu'

   inputs_mean, inputs_std, inputs_range, targets_mean, targets_std, targets_range = ReadMeanStd(MeanStd_prefix)

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
      for input_batch, target_batch, masks in data_loader:
          
          input_batch = input_batch.to(device, non_blocking=True, dtype=torch.float)
          target_batch = target_batch.cpu().detach().numpy()

          # get prediction from the model, given the inputs
          if isinstance(h, list):
             predicted_batch = h[0](input_batch) 
             for i in range(1, len(h)):
                predicted_batch = predicted_batch + h[i]( torch.cat( (input_batch, 
                                                                      masks.to(device, non_blocking=True, dtype=torch.float)
                                                                     ), dim=1) )
             predicted_batch = predicted_batch / len(h)
          else:
             predicted_batch = h( torch.cat((input_batch,
                                             masks.to(device, non_blocking=True, dtype=torch.float)
                                            ), dim=1) ) 
          predicted_batch = predicted_batch.cpu().detach().numpy() 

          # Denormalise
          target_batch = rr.RF_DeNormalise(target_batch, targets_mean, targets_std, targets_range, no_out_channels, dimension)
          predicted_batch = rr.RF_DeNormalise(predicted_batch, targets_mean, targets_std, targets_range, no_out_channels, dimension)

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
   print('rms_error.shape')
   print(rms_error.shape)
   print('masks.shape')
   print(masks.shape)
   rms_error = np.where( masks[0,:,:,:]==1, rms_error, np.nan)
   print('rms_error.shape')
   print(rms_error.shape)

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
                       y_dim_used, land, dimension, histlen, no_in_channels, no_out_channels, landvalues, iterate_method):
   logging.info('Iterating')

   device = 'cuda' if torch.cuda.is_available() else 'cpu'

   inputs_mean, inputs_std, inputs_range, targets_mean, targets_std, targets_range = ReadMeanStd(MeanStd_prefix)

   no_depth_levels = 38  # Hard coded...perhaps should change...?

   landvalues = torch.tensor(landvalues)

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
   #input_sample, target_sample, masks, bdy_masks = Iterate_Dataset.__getitem__(start)
   input_sample, target_sample, masks = Iterate_Dataset.__getitem__(start)
   # Give extra dimension at front (as usually number of samples)
   input_sample = input_sample.unsqueeze(0)
   masks = masks.unsqueeze(0)
   #### Above isn't the case - need to change to this shape!!
   if dimension == '2d':
      iterated_fields = rr.RF_DeNormalise(Iterate_Dataset.__getitem__(start)[0][:(no_depth_levels*3+1),:,:].unsqueeze(0),
                                          inputs_mean, inputs_std, inputs_range, no_out_channels, dimension)
      MITgcm_data = Iterate_Dataset.__getitem__(start)[0][:no_depth_levels*3+1,:,:].unsqueeze(0)
      for time in range(1,histlen):
          iterated_fields = torch.cat( (iterated_fields,
                                       rr.RF_DeNormalise(Iterate_Dataset.__getitem__(start)[0]
                                                         [time*(no_depth_levels*3+1):(time+1)*(no_depth_levels*3+1),:,:].unsqueeze(0),
                                                         inputs_mean, inputs_std, inputs_range, no_out_channels, dimension) ),
                                     axis = 0)
          MITgcm_data = torch.cat( (MITgcm_data, 
                                   Iterate_Dataset.__getitem__(start)[0][time*(no_depth_levels*3+1):(time+1)*(no_depth_levels*3+1),:,:].unsqueeze(0) ),
                                   axis=0)
      grid_predictions = torch.zeros(MITgcm_data.shape)
      grid_norm_predictions = torch.zeros(MITgcm_data.shape)
   elif dimension == '3d':
      iterated_fields = rr.RF_DeNormalise(input_sample[:,:4,:,:], inputs_mean, inputs_std, inputs_range, no_out_channels, dimension)
      MITgcm_data = input_sample[:,:4,:,:]
      for time in range(1,histlen+1):
         iterated_fields = torch.cat( (iterated_fields, 
                                       rr.RF_DeNormalise(input_sample[:,4*time:4*(time+1),:,:], inputs_mean, inputs_std, inputs_range, no_out_channels, dimension)
                                      ), axis=0)
         MITgcm_data = torch.cat( (MITgcm_data, input_sample[:,4*time:4*(time+1),:,:]), axis=0)

   # Make iterative forecast, saving predictions as we go, and pull out MITgcm data
   with torch.no_grad():
      for time in range(for_len):
         # Make prediction
         # multi model ensemble, or single model predictions
         print('')
         if isinstance(h, list):
            predicted = h[0]( torch.cat((input_sample, masks), axis=1)
                              .to(device, non_blocking=True, dtype=torch.float) ).cpu().detach()
            for i in range(1, len(h)):
               predicted = predicted + \
                           h[i]( torch.cat((input_sample, masks), axis=1)
                                 .to(device, non_blocking=True, dtype=torch.float) ).cpu().detach()
            predicted = predicted / len(h)
         else:
            predicted = h( torch.cat( (input_sample, masks),axis=1 ).to(device, non_blocking=True, dtype=torch.float)
                         ).cpu().detach()
         print('predicted in norm space:'+str(predicted[:,2,10,10]))
         grid_norm_predictions = torch.cat( (grid_norm_predictions, predicted), axis=0)

         # Denormalise 
         predicted = rr.RF_DeNormalise(predicted, targets_mean, targets_std, targets_range, no_out_channels, dimension)
         print('predicted denormed:'+str(predicted[:,2,10,10]))
         grid_predictions = torch.cat( (grid_predictions, predicted), axis=0)

         # Calculate next field, by combining increment with field
         if iterate_method == 'simple':
            next_field = iterated_fields[-1,:,:,:] + predicted
         elif iterate_method == 'AB2':
            if time == 0:
               next_field = iterated_fields[-1,:,:,:] + predicted
            else: 
               next_field = iterated_fields[-1,:,:,:] + 3./2. * predicted - 1./2. * old_predicted
            old_predicted = predicted
         elif iterate_method == 'RK4':
            k1 = predicted
            # (k1)/2 added to state, masked, then normalised, and finally masks catted on. And then passed through neural net)
            k2 = h( torch.cat( (
                       rr.RF_ReNormalise_predicted( 
                          torch.where( masks==1, ( iterated_fields[-1,:,:,:] + (k1)/2. ), landvalues.unsqueeze(0).unsqueeze(2).unsqueeze(3) ),
                          inputs_mean, inputs_std, inputs_range, no_out_channels, dimension), 
                       masks),axis=1 ).to(device, non_blocking=True, dtype=torch.float)
                  ).cpu().detach()
            k3 = h( torch.cat( (
                       rr.RF_ReNormalise_predicted( 
                          torch.where( masks==1, ( iterated_fields[-1,:,:,:] + (k2)/2. ), landvalues.unsqueeze(0).unsqueeze(2).unsqueeze(3) ),
                          inputs_mean, inputs_std, inputs_range, no_out_channels, dimension), 
                       masks),axis=1 ).to(device, non_blocking=True, dtype=torch.float)
                  ).cpu().detach()
            k4 = h( torch.cat( (
                       rr.RF_ReNormalise_predicted( 
                          torch.where( masks==1, ( iterated_fields[-1,:,:,:] + (k3)    ), landvalues.unsqueeze(0).unsqueeze(2).unsqueeze(3) ),
                          inputs_mean, inputs_std, inputs_range, no_out_channels, dimension), 
                       masks),axis=1 ).to(device, non_blocking=True, dtype=torch.float)
                  ).cpu().detach()
            next_field = iterated_fields[-1,:,:,:] + (1./6.) * ( k1 + 2*k2 +2*k3 + k4 )         
         print('iterated fields:'+str(iterated_fields[:,2,10,10]))
         print('next field shape:'+str(next_field.shape))
         print('next field:'+str(next_field[:,2,10,10]))

         # Mask field
         next_field = torch.where( masks==1, next_field, landvalues.unsqueeze(0).unsqueeze(2).unsqueeze(3) )
         # Cat prediction onto existing fields
         iterated_fields = torch.cat( ( iterated_fields, next_field ), axis=0 )
         # Re-Normalise ready to use as inputs
         print('iterated fields:'+str(iterated_fields[:,2,10,10]))
         next_field = rr.RF_ReNormalise_predicted(next_field, inputs_mean, inputs_std, inputs_range, no_out_channels, dimension)

         # Get MITgcm data for relevant step, and prep new input sample
         if dimension == '2d':
            # Plus one to time dimension as we want the next step, to match time stamp of prediction
            MITgcm_data = torch.cat( ( MITgcm_data, Iterate_Dataset.__getitem__(start+time+1)[0]
                                       [(histlen-1)*(no_depth_levels*3+1):histlen*(no_depth_levels*3+1),:,:].unsqueeze(0) ),
                                       axis=0 )
            input_sample = torch.zeros((1, histlen*(3*no_depth_levels+1), y_dim_used, da_X.shape[0]))
            for time in range(histlen-1):
               input_sample[0,time*(3*no_depth_levels+1):(time+1)*(3*no_depth_levels+1),:,:] = \
                                 input_sample[0,(time+1)*(3*no_depth_levels+1):(time+2)*(3*no_depth_levels+1),:,:]
            input_sample[0,(histlen-1)*(3*no_depth_levels+1):(histlen)*(3*no_depth_levels+1),:,:] = next_field
           

         elif dimension == '3d':
            MITgcm_data = torch.cat( ( MITgcm_data, 
                                       Iterate_Dataset.__getitem__(start+time+1)[0][(histlen-1)*4:histlen*4,:,:,:].unsqueeze(0) ),
                                       axis=0 )
            input_sample = torch.zeros((1, histlen*4, da_Z.shape[0], y_dim_used, da_X.shape[0]))
            for time in range(histlen):
               input_sample[0,time*4:(time+1)*4,:,:,:]=next_field[:,:,:,:]

         del next_field
         del predicted 
         gc.collect()
         torch.cuda.empty_cache()

   logging.debug('iterated_fields.shape ; '+str(iterated_fields.shape))
   logging.debug('MITgcm_data.shape ; '+str(MITgcm_data.shape))

   ## Denormalise MITgcm Data 
   MITgcm_data          = rr.RF_DeNormalise(MITgcm_data, inputs_mean, inputs_std, inputs_range, no_out_channels, dimension) 

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

   nc_TrueTempInc = nc_file.createVariable( 'Tr_TInc'  , 'f4', ('T', 'Z', 'Y', 'X') )
   nc_TrueUInc    = nc_file.createVariable( 'Tr_UInc'     , 'f4', ('T', 'Z', 'Y', 'X') )
   nc_TrueVInc    = nc_file.createVariable( 'Tr_VInc'     , 'f4', ('T', 'Z', 'Y', 'X') )
   nc_TrueEtaInc  = nc_file.createVariable( 'Tr_EInc'   , 'f4', ('T', 'Y', 'X')      )

   nc_TempMask   = nc_file.createVariable( 'Temp_Mask'  , 'f4', ('Z', 'Y', 'X') )
   nc_UMask      = nc_file.createVariable( 'U_Mask'     , 'f4', ('Z', 'Y', 'X') )
   nc_VMask      = nc_file.createVariable( 'V_Mask'     , 'f4', ('Z', 'Y', 'X') )
   nc_EtaMask    = nc_file.createVariable( 'Eta_Mask'   , 'f4', ('Y', 'X')      )

   nc_NormPredTempInc= nc_file.createVariable( 'Pr_T_NormInc'  , 'f4', ('T', 'Z', 'Y', 'X') )
   nc_NormPredUInc   = nc_file.createVariable( 'Pr_U_NormInc'     , 'f4', ('T', 'Z', 'Y', 'X') )
   nc_NormPredVInc   = nc_file.createVariable( 'Pr_V_NormInc'     , 'f4', ('T', 'Z', 'Y', 'X') ) 
   nc_NormPredEtaInc = nc_file.createVariable( 'Pr_E_NormInc'   , 'f4', ('T', 'Y', 'X')      )

   nc_PredTempInc= nc_file.createVariable( 'Pr_T_Inc'  , 'f4', ('T', 'Z', 'Y', 'X') )
   nc_PredUInc   = nc_file.createVariable( 'Pr_U_Inc'     , 'f4', ('T', 'Z', 'Y', 'X') )
   nc_PredVInc   = nc_file.createVariable( 'Pr_V_Inc'     , 'f4', ('T', 'Z', 'Y', 'X') ) 
   nc_PredEtaInc = nc_file.createVariable( 'Pr_E_Inc'   , 'f4', ('T', 'Y', 'X')      )

   nc_PredTempFld= nc_file.createVariable( 'Pred_Temp'  , 'f4', ('T', 'Z', 'Y', 'X') )
   nc_PredUFld   = nc_file.createVariable( 'Pred_U'     , 'f4', ('T', 'Z', 'Y', 'X') )
   nc_PredVFld   = nc_file.createVariable( 'Pred_V'     , 'f4', ('T', 'Z', 'Y', 'X') ) 
   nc_PredEtaFld = nc_file.createVariable( 'Pred_Eta'   , 'f4', ('T', 'Y', 'X')      )

   nc_CalcTempInc= nc_file.createVariable( 'Calc_Temp_Inc'  , 'f4', ('T', 'Z', 'Y', 'X') )
   nc_CalcUInc   = nc_file.createVariable( 'Calc_U_Inc'     , 'f4', ('T', 'Z', 'Y', 'X') )
   nc_CalcVInc   = nc_file.createVariable( 'Calc_V_Inc'     , 'f4', ('T', 'Z', 'Y', 'X') ) 
   nc_CalcEtaInc = nc_file.createVariable( 'Calc_Eta_Inc'   , 'f4', ('T', 'Y', 'X')      )

   nc_TempErrors = nc_file.createVariable( 'Temp_Errors', 'f4', ('T', 'Z', 'Y', 'X') )
   nc_UErrors    = nc_file.createVariable( 'U_Errors'   , 'f4', ('T', 'Z', 'Y', 'X') )
   nc_VErrors    = nc_file.createVariable( 'V_Errors'   , 'f4', ('T', 'Z', 'Y', 'X') )
   nc_EtaErrors  = nc_file.createVariable( 'Eta_Errors' , 'f4', ('T', 'Y', 'X')      )

   # Fill variables
   nc_T[:] = np.arange(iterated_fields.shape[0])
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

      nc_NormPredTempInc[:,:,:,:] = grid_norm_predictions[:,0:no_depth_levels,:,:]
      nc_NormPredUInc[:,:,:,:]    = grid_norm_predictions[:,1*no_depth_levels:2*no_depth_levels,:,:]
      nc_NormPredVInc[:,:,:,:]    = grid_norm_predictions[:,2*no_depth_levels:3*no_depth_levels,:,:]
      nc_NormPredEtaInc[:,:,:]    = grid_norm_predictions[:,3*no_depth_levels,:,:]
      
      nc_PredTempInc[:,:,:,:] = grid_predictions[:,0:no_depth_levels,:,:]
      nc_PredUInc[:,:,:,:]    = grid_predictions[:,1*no_depth_levels:2*no_depth_levels,:,:]
      nc_PredVInc[:,:,:,:]    = grid_predictions[:,2*no_depth_levels:3*no_depth_levels,:,:]
      nc_PredEtaInc[:,:,:]    = grid_predictions[:,3*no_depth_levels,:,:]
      
      nc_PredTempFld[:,:,:,:] = iterated_fields[:,0:no_depth_levels,:,:]
      nc_PredUFld[:,:,:,:]    = iterated_fields[:,1*no_depth_levels:2*no_depth_levels,:,:]
      nc_PredVFld[:,:,:,:]    = iterated_fields[:,2*no_depth_levels:3*no_depth_levels,:,:]
      nc_PredEtaFld[:,:,:]    = iterated_fields[:,3*no_depth_levels,:,:]
      
      nc_CalcTempInc[1:,:,:,:] = iterated_fields[1:,0:no_depth_levels,:,:] - iterated_fields[:-1,0:no_depth_levels,:,:]
      nc_CalcUInc[1:,:,:,:]    = iterated_fields[1:,1*no_depth_levels:2*no_depth_levels,:,:] - iterated_fields[:-1,1*no_depth_levels:2*no_depth_levels,:,:]
      nc_CalcVInc[1:,:,:,:]    = iterated_fields[1:,2*no_depth_levels:3*no_depth_levels,:,:] - iterated_fields[:-1,2*no_depth_levels:3*no_depth_levels,:,:]
      nc_CalcEtaInc[1:,:,:]    = iterated_fields[1:,3*no_depth_levels,:,:] - iterated_fields[:-1,3*no_depth_levels,:,:]

      nc_TrueTempInc[1:,:,:,:] = MITgcm_data[1:,0:no_depth_levels,:,:] - MITgcm_data[:-1,0:no_depth_levels,:,:] 
      nc_TrueUInc[1:,:,:,:]    = MITgcm_data[1:,1*no_depth_levels:2*no_depth_levels,:,:] - MITgcm_data[:-1,1*no_depth_levels:2*no_depth_levels,:,:]
      nc_TrueVInc[1:,:,:,:]    = MITgcm_data[1:,2*no_depth_levels:3*no_depth_levels,:,:] - MITgcm_data[:-1,2*no_depth_levels:3*no_depth_levels,:,:]
      nc_TrueEtaInc[1:,:,:]    = MITgcm_data[1:,3*no_depth_levels,:,:] - MITgcm_data[:-1,3*no_depth_levels,:,:]

      nc_TempErrors[:,:,:,:] = iterated_fields[:,0:no_depth_levels,:,:] - MITgcm_data[:,0:no_depth_levels,:,:]
      nc_UErrors[:,:,:,:]    = iterated_fields[:,1*no_depth_levels:2*no_depth_levels,:,:] - MITgcm_data[:,1*no_depth_levels:2*no_depth_levels,:,:]
      nc_VErrors[:,:,:,:]    = iterated_fields[:,2*no_depth_levels:3*no_depth_levels,:,:] - MITgcm_data[:,2*no_depth_levels:3*no_depth_levels,:,:]
      nc_EtaErrors[:,:,:]    = iterated_fields[:,3*no_depth_levels,:,:] - MITgcm_data[:,3*no_depth_levels,:,:]

   elif dimension == '3d':
      nc_TempMask[:,:,:] = masks[0,0,:,:,:]
      nc_UMask[:,:,:]    = masks[0,1,:,:,:]
      nc_VMask[:,:,:]    = masks[0,2,:,:,:]
      nc_EtaMask[:,:]    = masks[0,3,0,:,:]

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
