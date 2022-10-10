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

def ReadMeanStd(dim, no_levels):
   if dim =='2d':
      mean_std_file = '../../../Channel_nn_Outputs/Spits_2d_MeanStd.npz'
   elif dim =='3d':
      mean_std_file = '../../../Channel_nn_Outputs/Spits_3d_MeanStd.npz'
   mean_std_data = np.load(mean_std_file)
   inputs_mean  = mean_std_data['arr_0']
   inputs_std   = mean_std_data['arr_1']
   inputs_range = mean_std_data['arr_2']
   targets_mean  = mean_std_data['arr_3']
   targets_std   = mean_std_data['arr_4']
   targets_range = mean_std_data['arr_5']

   ## Expand arrays to multiple levels for 2d approach
   #if dim =='2d':
   #   tmp_inputs_mean  = np.zeros((3*no_levels+1))
   #   tmp_inputs_std   = np.zeros((3*no_levels+1))
   #   tmp_inputs_range = np.zeros((3*no_levels+1))
   #   tmp_targets_mean  = np.zeros((3*no_levels+1))
   #   tmp_targets_std   = np.zeros((3*no_levels+1))
   #   tmp_targets_range = np.zeros((3*no_levels+1))

   #   for var in range(3):
   #      tmp_inputs_mean[var*no_levels:(var+1)*no_levels] = tmp_inputs_mean[var]
   #      tmp_inputs_std[var*no_levels:(var+1)*no_levels] = tmp_inputs_std[var]
   #      tmp_inputs_range[var*no_levels:(var+1)*no_levels] = tmp_inputs_range[var]
   #      tmp_targets_mean[var*no_levels:(var+1)*no_levels] = tmp_targets_mean[var]
   #      tmp_targets_std[var*no_levels:(var+1)*no_levels] = tmp_targets_std[var]
   #      tmp_targets_range[var*no_levels:(var+1)*no_levels] = tmp_targets_range[var]
   #   tmp_inputs_mean[3*no_levels] = tmp_inputs_mean[3]
   #   tmp_inputs_std[3*no_levels] = tmp_inputs_std[3]
   #   tmp_inputs_range[3*no_levels] = tmp_inputs_range[3]
   #   tmp_targets_mean[3*no_levels] = tmp_targets_mean[3]
   #   tmp_targets_std[3*no_levels] = tmp_targets_std[3]
   #   tmp_targets_range[3*no_levels] = tmp_targets_range[3]

   #   inputs_mean  = tmp_inputs_mean
   #   inputs_std   = tmp_inputs_std 
   #   inputs_range = tmp_inputs_range
   #   targets_mean  = tmp_targets_mean
   #   targets_std   = tmp_targets_std 
   #   targets_range = tmp_targets_range

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
           # multiply by masks, to give 0 at land values, to circumvent issues with normalisation!
           target_batch = target_batch.to(device, non_blocking=True, dtype=torch.float) * masks
           # input_batch SHAPE: (no_samples, no_channels, z, y, x) if 3d, (no_samples, no_channels, y, x) if 2d.
   
           # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
           optimizer.zero_grad()
       
           # get prediction from the model, given the inputs
           # multiply by masks, to give 0 at land values, to circumvent issues with normalisation!
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
   return(losses)

def PlotScatter(model_name, dimension, data_loader, h, epoch, title, no_out_channels, MeanStd_prefix):

   logging.info('Making scatter plots')
   plt.rcParams.update({'font.size': 14})

   no_depth_levels = 38  # Hard coded...perhaps should change...?

   inputs_mean, inputs_std, inputs_range, targets_mean, targets_std, targets_range = ReadMeanStd(dimension, no_depth_levels)

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
                         ).cpu().detach().double()

   # Denormalise and mask
   target_batch = torch.where(masks==1, rr.RF_DeNormalise(target_batch, targets_mean, targets_std, targets_range, dimension), np.nan)
   predicted_batch = torch.where(masks==1, rr.RF_DeNormalise(predicted_batch, targets_mean, targets_std, targets_range, dimension), np.nan)

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
   bottom = min( np.nanmin(targets), np.nanmin(predictions) )
   top    = max( np.nanmax(targets), np.nanmax(predictions) )  
   
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

   # Read in grid data from MITgcm file
   mitgcm_ds = xr.open_dataset(mitgcm_filename)
   da_X = mitgcm_ds['X']
   da_Y = mitgcm_ds['Yp1']
   da_Z = mitgcm_ds['Zmd000038'] 

   inputs_mean, inputs_std, inputs_range, targets_mean, targets_std, targets_range = ReadMeanStd(dimension, da_Z)

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
             predicted_batch = h[0]( torch.cat((input_batch,
                                             masks.to(device, non_blocking=True, dtype=torch.float)
                                            ), dim=1) ) 
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
          target_batch = rr.RF_DeNormalise(target_batch, targets_mean, targets_std, targets_range, dimension)
          predicted_batch = rr.RF_DeNormalise(predicted_batch, targets_mean, targets_std, targets_range, dimension)
          input_batch = rr.RF_DeNormalise(input_batch.cpu().detach().numpy(), inputs_mean, inputs_std, inputs_range, dimension)

          # Convert from increments to fields for outputting
          target_batch = input_batch + target_batch
          predicted_batch = input_batch + predicted_batch

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
   rms_error = np.where( masks[0,:,:,:]==1, rms_error, np.nan)

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

   # Read in grid data from MITgcm file
   mitgcm_ds = xr.open_dataset(mitgcm_filename)
   da_X = mitgcm_ds['X']
   da_Y = mitgcm_ds['Yp1']
   da_Z = mitgcm_ds['Zmd000038'] 
   no_depth_levels = 38   #Unesseccary duplication, could use with da_Z shape , should change...
   inputs_mean, inputs_std, inputs_range, targets_mean, targets_std, targets_range = ReadMeanStd(dimension, da_Z)

   landvalues = torch.tensor(landvalues)

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
   input_sample, target_sample, masks = Iterate_Dataset.__getitem__(start)
   # Give extra dimension at front (as usually number of samples)
   input_sample = input_sample.unsqueeze(0)
   masks = masks.unsqueeze(0)
   if dimension == '2d':
      iterated_fields = rr.RF_DeNormalise(Iterate_Dataset.__getitem__(start)[0][:(no_depth_levels*3+1),:,:].unsqueeze(0),
                                          inputs_mean, inputs_std, inputs_range, dimension)
      MITgcm_data = Iterate_Dataset.__getitem__(start)[0][:no_depth_levels*3+1,:,:].unsqueeze(0)
      for time in range(1,histlen):
          iterated_fields = torch.cat( (iterated_fields,
                                       rr.RF_DeNormalise(Iterate_Dataset.__getitem__(start)[0]
                                                         [time*(no_depth_levels*3+1):(time+1)*(no_depth_levels*3+1),:,:].unsqueeze(0),
                                                         inputs_mean, inputs_std, inputs_range, dimension) ),
                                     axis = 0)
          MITgcm_data = torch.cat( (MITgcm_data, 
                                   Iterate_Dataset.__getitem__(start)[0][time*(no_depth_levels*3+1):(time+1)*(no_depth_levels*3+1),:,:].unsqueeze(0) ),
                                   axis=0)
   elif dimension == '3d':
      iterated_fields = rr.RF_DeNormalise(Iterate_Dataset.__getitem__(start)[0][:4,:,:,:].unsqueeze(0),
                                          inputs_mean, inputs_std, inputs_range, dimension)
      MITgcm_data = Iterate_Dataset.__getitem__(start)[0][:4,:,:,:].unsqueeze(0)
      for time in range(1,histlen):
          iterated_fields = torch.cat( (iterated_fields,
                                       rr.RF_DeNormalise(Iterate_Dataset.__getitem__(start)[0]
                                                         [time*4:(time+1)*4,:,:,:].unsqueeze(0),
                                                         inputs_mean, inputs_std, inputs_range, dimension) ),
                                     axis = 0)
          MITgcm_data = torch.cat( (MITgcm_data, 
                                   Iterate_Dataset.__getitem__(start)[0][time*4:(time+1)*4,:,:].unsqueeze(0) ),
                                   axis=0)

   grid_predictions = torch.zeros(MITgcm_data.shape)
   grid_norm_predictions = torch.zeros(MITgcm_data.shape)

   # Make iterative forecast, saving predictions as we go, and pull out MITgcm data
   with torch.no_grad():
      for time in range(for_len):
         # Make prediction
         # multi model ensemble, or single model predictions
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
         grid_norm_predictions = torch.cat( (grid_norm_predictions, predicted), axis=0)

         # Denormalise 
         predicted = rr.RF_DeNormalise(predicted, targets_mean, targets_std, targets_range, dimension)
         grid_predictions = torch.cat( (grid_predictions, predicted), axis=0)

         # Calculate next field, by combining increment with field
         if iterate_method == 'simple':
            next_field = iterated_fields[-1] + predicted
         elif iterate_method == 'smooth':
            next_field = iterated_fields[-1] + predicted
            next_field = torch.from_numpy( gaussian_filter(next_field.numpy(), sigma=.44 ) )
         elif iterate_method == 'AB2':
            if time == 0:
               next_field = rr.RF_DeNormalise(Iterate_Dataset.__getitem__(start+time+1)[0]
                                       [(histlen-1)*(no_depth_levels*3+1):histlen*(no_depth_levels*3+1),:,:].unsqueeze(0),
                                          inputs_mean, inputs_std, inputs_range, dimension)
            else: 
               next_field = iterated_fields[-1] + 3./2. * predicted - 1./2. * old_predicted
            old_predicted = predicted
         elif iterate_method == 'RK4':
            if dimension == '2d':
               k1 = predicted
               # (k1)/2 added to state, masked, then normalised, and finally masks catted on. And then passed through neural net)
               k2 = h( torch.cat( (
                          rr.RF_Normalise( 
                             torch.where( masks==1, ( iterated_fields[-1,:,:,:] + (k1)/2. ), landvalues.unsqueeze(1).unsqueeze(2) ),
                             inputs_mean, inputs_std, inputs_range, dimension), 
                          masks),axis=1 ).to(device, non_blocking=True, dtype=torch.float)
                     ).cpu().detach()
               k3 = h( torch.cat( (
                          rr.RF_Normalise( 
                             torch.where( masks==1, ( iterated_fields[-1,:,:,:] + (k2)/2. ), landvalues.unsqueeze(1).unsqueeze(2) ),
                             inputs_mean, inputs_std, inputs_range, dimension), 
                          masks),axis=1 ).to(device, non_blocking=True, dtype=torch.float)
                     ).cpu().detach()
               k4 = h( torch.cat( (
                          rr.RF_Normalise( 
                             torch.where( masks==1, ( iterated_fields[-1,:,:,:] + (k3)    ), landvalues.unsqueeze(1).unsqueeze(2) ),
                             inputs_mean, inputs_std, inputs_range, dimension), 
                          masks),axis=1 ).to(device, non_blocking=True, dtype=torch.float)
                     ).cpu().detach()
               next_field = iterated_fields[-1,:,:,:] + (1./6.) * ( k1 + 2*k2 +2*k3 + k4 )         
            elif dimension == '3d':
               k1 = predicted
               # (k1)/2 added to state, masked, then normalised, and finally masks catted on. And then passed through neural net)
               k2 = h( torch.cat( (
                          rr.RF_Normalise( 
                             torch.where( masks==1, ( iterated_fields[-1,:,:,:,:] + (k1)/2. ), landvalues.unsqueeze(1).unsqueeze(2).unsqueeze(3) ),
                             inputs_mean, inputs_std, inputs_range, dimension), 
                          masks),axis=1 ).to(device, non_blocking=True, dtype=torch.float)
                     ).cpu().detach()
               k3 = h( torch.cat( (
                          rr.RF_Normalise( 
                             torch.where( masks==1, ( iterated_fields[-1,:,:,:,:] + (k2)/2. ), landvalues.unsqueeze(1).unsqueeze(2).unsqueeze(3) ),
                             inputs_mean, inputs_std, inputs_range, dimension), 
                          masks),axis=1 ).to(device, non_blocking=True, dtype=torch.float)
                     ).cpu().detach()
               k4 = h( torch.cat( (
                          rr.RF_Normalise( 
                             torch.where( masks==1, ( iterated_fields[-1,:,:,:,:] + (k3)    ), landvalues.unsqueeze(1).unsqueeze(2).unsqueeze(3) ),
                             inputs_mean, inputs_std, inputs_range, dimension), 
                          masks),axis=1 ).to(device, non_blocking=True, dtype=torch.float)
                     ).cpu().detach()
               next_field = iterated_fields[-1,:,:,:,:] + (1./6.) * ( k1 + 2*k2 +2*k3 + k4 )         
         else:
            sys.exit('no suitable iteration method chosen')

         # Mask field
         if dimension == '2d':
            next_field = torch.where( masks==1, next_field, landvalues.unsqueeze(1).unsqueeze(2) )
         if dimension == '3d':
            next_field = torch.where( masks==1, next_field, landvalues.unsqueeze(1).unsqueeze(2).unsqueeze(3) )
         # Cat prediction onto existing fields
         iterated_fields = torch.cat( ( iterated_fields, next_field ), axis=0 )
         # Re-Normalise ready to use as inputs
         next_field = rr.RF_Normalise(next_field, inputs_mean, inputs_std, inputs_range, dimension)

         # Get MITgcm data for relevant step, and prep new input sample
         if dimension == '2d':
            # Plus one to time dimension as we want the next step, to match time stamp of prediction
            MITgcm_data = torch.cat( ( MITgcm_data, Iterate_Dataset.__getitem__(start+time+1)[0]
                                       [(histlen-1)*(no_depth_levels*3+1):histlen*(no_depth_levels*3+1),:,:].unsqueeze(0) ),
                                       axis=0 )
            for time in range(histlen-1):
               input_sample[0,time*(3*no_depth_levels+1):(time+1)*(3*no_depth_levels+1),:,:] = \
                                 input_sample[0,(time+1)*(3*no_depth_levels+1):(time+2)*(3*no_depth_levels+1),:,:]
            input_sample[0,(histlen-1)*(3*no_depth_levels+1):(histlen)*(3*no_depth_levels+1),:,:] = next_field[0,:,:,:]

         elif dimension == '3d':
            # Plus one to time dimension as we want the next step, to match time stamp of prediction
            MITgcm_data = torch.cat( ( MITgcm_data, Iterate_Dataset.__getitem__(start+time+1)[0]
                                       [(histlen-1)*4:histlen*4,:,:,:].unsqueeze(0) ),
                                       axis=0 )
            input_sample = torch.zeros((1, 4, da_Z.shape[0], y_dim_used, da_X.shape[0]))
            for time in range(histlen-1):
               input_sample[0,time*4:(time+1)*4,:,:] = \
                                 input_sample[0,(time+1)*4:(time+2)*4,:,:]
            input_sample[0,(histlen-1)*4:(histlen)*4,:,:] = next_field
           
         del next_field
         del predicted 
         gc.collect()
         torch.cuda.empty_cache()

   logging.debug('iterated_fields.shape ; '+str(iterated_fields.shape))
   logging.debug('MITgcm_data.shape ; '+str(MITgcm_data.shape))

   ## Denormalise MITgcm Data 
   MITgcm_data          = rr.RF_DeNormalise(MITgcm_data, inputs_mean, inputs_std, inputs_range, dimension) 

   # Set up netcdf files
   nc_filename = '../../../Channel_nn_Outputs/'+model_name+'/ITERATED_FORECAST/'+model_name+'_'+str(no_epochs)+'epochs_'+iterate_method+'_Forecast'+str(for_len)+'.nc'
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


def PlotTrainingEvolution(model_name, MeanStd_prefix, mitgcm_filename, Evolution_Dataset, h, optimizer, epochs_list,
                          dimension, histlen, no_in_channels, no_out_channels, landvalues):

   level = 2
   time = 5
   
   rootdir = '../../../Channel_nn_Outputs/'+model_name
   
   device = 'cuda' if torch.cuda.is_available() else 'cpu'
   mitgcm_ds = xr.open_dataset(mitgcm_filename)
   da_X = mitgcm_ds['X']
   da_Y = mitgcm_ds['Y']
   da_Z = mitgcm_ds['Zmd000038']
   no_levels = da_Z.shape[0]
   
   inputs_mean, inputs_std, inputs_range, targets_mean, targets_std, targets_range = ReadMeanStd(dimension, no_levels)

   input_sample, target_sample, masks = Evolution_Dataset.__getitem__(time)
   input_sample = input_sample.unsqueeze(0)
   target_sample = target_sample.unsqueeze(0)

   prev_field = rr.RF_DeNormalise(input_sample, inputs_mean, inputs_std, inputs_range, dimension)[0]
   target_tend = rr.RF_DeNormalise(target_sample, targets_mean, targets_std, targets_range, dimension)[0]
   target_field = prev_field + target_tend
   target_tend = np.where( masks.numpy()==1., target_tend.numpy(), np.nan )
   target_field = np.where( masks.numpy()==1., target_field.numpy(), np.nan )
   
   for epochs in epochs_list:
   
      # Load model
      losses = {'train' : [], 'train_Temp': [], 'train_U' : [], 'train_V' : [], 'train_Eta' : [], 'val' : [] }
      losses, h, optimizer, current_best_loss = LoadModel(model_name, h, optimizer, epochs, 'inf', losses, False)  
      h.train(False)
      if torch.cuda.is_available():
         h = h.cuda()
   
      # Make prediction
      with torch.no_grad():
         predicted_tend = h( torch.cat( (input_sample, masks.unsqueeze(0)), dim=1 ).to(device, non_blocking=True, dtype=torch.float) ).cpu().detach()
         predicted_tend = rr.RF_DeNormalise(predicted_tend, targets_mean, targets_std, targets_range, dimension)[0]
         predicted_field = prev_field + predicted_tend
   
      # Make plot
      predicted_field = np.where( masks.numpy()==1., predicted_field.numpy(), np.nan )
      predicted_tend = np.where( masks.numpy()==1., predicted_tend.numpy(), np.nan )
      # Need to add code for 3d options here and make sure it works for histlen>1
      if dimension == '2d':
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
