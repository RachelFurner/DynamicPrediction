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
import Dev_ReadRoutines as rr

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
import torch.autograd.profiler as profiler

import netCDF4 as nc4

import time as time
import gc

import logging

import multiprocessing as mp
#import torch.multiprocessing
#torch.multiprocessing.set_sharing_strategy('file_system')

import GPUtil 

#add this routine for memory profiling
def sizeof_fmt(num, suffix='B'):
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)


def TimeCheck(tic, task):
   toc = time.time()
   logging.info('Finished '+task+' at {:0.4f} seconds'.format(toc - tic)+'\n')

def CalcMeanStd(MeanStd_prefix, MITGCM_filename, train_end_ratio, subsample_rate, dataset_end_index,
                batch_size, hist_len, land, dimension, tic):

   no_depth_levels = 38 ## Note this is hard coded!!

   if dimension == '2d':
      mean_std_Dataset = rr.MITGCM_Dataset_2d( MITGCM_filename, 0.0, train_end_ratio, subsample_rate, dataset_end_index, hist_len, land, tic,
                                               transform=None)
      mean_std_loader = data.DataLoader(mean_std_Dataset, batch_size=batch_size, shuffle=False, num_workers=0)
      #------------------------------------
      # First calculate the mean and range
      #------------------------------------
      FirstPass = True 
      batch_no = 0
      count = 0.
      for batch_sample in mean_std_loader:
          
          # shape (batch_size, channels, y, x)
          input_batch  = batch_sample['input']

          # Calc mean of this batch
          batch_inputs_mean  = np.mean(input_batch.numpy(), axis = (0,2,3))
 
          #combine with previous mean and calc min and max
          if FirstPass:
             inputs_mean = batch_inputs_mean  
             inputs_min  = np.amin( input_batch.numpy(), axis = (0,2,3) )
             inputs_max  = np.amax( input_batch.numpy(), axis = (0,2,3) )
          else:
             inputs_mean = 1./(count+input_batch.shape[0]) * ( count*inputs_mean + input_batch.shape[0]*batch_inputs_mean )
             inputs_min  = np.minimum( inputs_min, np.amin(input_batch.numpy()) )
             inputs_max  = np.maximum( inputs_max, np.amax(input_batch.numpy()) )

          batch_no = batch_no+1
          count = count + input_batch.shape[0]
          FirstPass = False

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
      for batch_sample in mean_std_loader:
          
          # shape (batch_size, channels, y, x)
          input_batch  = batch_sample['input']
          output_batch  = batch_sample['output']
   
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
      mean_std_Dataset = rr.MITGCM_Dataset_3d( MITGCM_filename, 0.0, train_end_ratio, subsample_rate, dataset_end_index, hist_len, land, tic,
                                               transform=None)
      mean_std_loader = data.DataLoader(mean_std_Dataset, batch_size=batch_size, shuffle=False, num_workers=0)
      
      #------------------------------------
      # First calculate the mean and range
      #------------------------------------
      FirstPass = True 
      batch_no = 0
      count = 0.
      for batch_sample in mean_std_loader:
          
          # shape (batch_size, channels, z, y, x)
          input_batch  = batch_sample['input']
   
          # Calc mean of this batch
          batch_inputs_mean  = np.mean(input_batch.numpy(), axis = (0,2,3,4))
    
          #combine with previous mean and calc min and max
          if FirstPass:
             inputs_mean = batch_inputs_mean  
             inputs_min  = np.amin( input_batch.numpy(), axis = (0,2,3,4) )
             inputs_max  = np.amax( input_batch.numpy(), axis = (0,2,3,4) )
          else:
             inputs_mean = 1./(count+input_batch.shape[0]) * ( count*inputs_mean + input_batch.shape[0]*batch_inputs_mean )
             inputs_min  = np.minimum( inputs_min, np.amin(input_batch.numpy()) )
             inputs_max  = np.maximum( inputs_max, np.amax(input_batch.numpy()) )
   
          batch_no = batch_no+1
          count = count + input_batch.shape[0]
          FirstPass = False
   
      inputs_range = inputs_max - inputs_min
   
      #-------------------------
      # Next calculate the std 
      #-------------------------
      count = 0.
      for batch_sample in mean_std_loader:
          
          # shape (batch_size, channels, z, y, x)
          input_batch  = batch_sample['input']
          output_batch  = batch_sample['output']
   
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

def TrainModel(model_name, dimension, tic, TEST, no_tr_samples, no_val_samples, plot_freq, save_freq, train_loader,
               val_loader, h, optimizer, hyper_params, seed_value, losses, start_epoch=0):

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

   # define dictionary to store data to plot sccatter graphs 
   plotting_data = {'targets_train'     : [],
                    'predictions_train' : [],
                    'targets_val'       : [],
                    'predictions_val'   : []}

   current_best_loss = 0.0001

   logging.info('###### TRAINING MODEL ######')
   for epoch in range(start_epoch,hyper_params["num_epochs"]+start_epoch):
   
       #logging.info('epoch ; '+str(epoch))
       print('start of epoch ; '+str(epoch))
       GPUtil.showUtilization() 

       # Training
   
       losses['train'].append(0.)
       losses['train_Temp'].append(0.)
       losses['train_U'].append(0.)
       losses['train_V'].append(0.)
       losses['train_Eta'].append(0.)
       losses['val'].append(0.)

       batch_no = 0
       for batch_sample in train_loader:

           #logging.info('batch_no ;'+str(batch_no))
           #print('batch_no ;'+str(batch_no))
           #GPUtil.showUtilization() 

           with profiler.profile(record_shapes=True, profile_memory=True) as prof:
              with profiler.record_function("Read inputs"):
                 input_batch  = batch_sample['input']
                 target_batch = batch_sample['output']
           input_batch = input_batch.to(device, non_blocking=True, dtype=torch.float)
           target_batch = target_batch.to(device, non_blocking=True, dtype=torch.float)
           #logging.info('Read in samples and sent to GPU')
           #print('Read in samples and sent to GPU')
           #GPUtil.showUtilization() 
           
           h.train(True)
   
           # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
           optimizer.zero_grad()
           #logging.info('Set train to true, and cleared ptimizer grad')
           #print('Set train to true, and cleared ptimizer grad')
           #GPUtil.showUtilization() 
       
           # get prediction from the model, given the inputs
           with profiler.profile(record_shapes=True, profile_memory=True) as prof:
              with profiler.record_function("Make predictions"):
                 predicted = h(input_batch)
           #logging.info('Made predicitons')
           #print('Made predicitons')
           #GPUtil.showUtilization() 
   
           if epoch==start_epoch and batch_no == 0:
              logging.info('For test run check if output is correct shape, the following two shapes should match!\n')
              logging.info('target_batch.shape for training data: '+str(target_batch.shape)+'\n')
              logging.info('predicted.shape for training data: '+str(predicted.shape)+'\n')
              
           # get loss for the predicted output
           with profiler.profile(record_shapes=True, profile_memory=True) as prof:
              with profiler.record_function("Calc Loss"):
                 loss = hyper_params["criterion"](predicted, target_batch)
           #logging.info('Calculated Loss')
           #print('Calculated Loss')
           #GPUtil.showUtilization() 
           
           # get gradients w.r.t to parameters
           with profiler.profile(record_shapes=True, profile_memory=True) as prof:
              with profiler.record_function("Back Prop"):
                 loss.backward()
           #logging.info('Calculated gradients')
           #print('Calculated gradients')
           #GPUtil.showUtilization() 
        
           # update parameters
           with profiler.profile(record_shapes=True, profile_memory=True) as prof:
              with profiler.record_function("optimizer step"):
                 optimizer.step()
           #logging.info('stepped optimiser')
           #print('stepped optimiser')
           #GPUtil.showUtilization() 
   
           if TEST and epoch == start_epoch and batch_no == 0:  # Recalculate loss post optimizer step to ensure its decreased
              h.train(False)
              post_opt_predicted = h(input_batch)
              post_opt_loss = hyper_params["criterion"](post_opt_predicted, target_batch)
              logging.info('For test run check if loss decreases over a single training step on a batch of data')
              logging.info('loss pre optimization: ' + str(loss.item()) )
              logging.info('loss post optimization: ' + str(post_opt_loss.item()) )
          
           # Update loss values
           losses['train'][-1] = losses['train'][-1] + loss.item()*input_batch.shape[0]
           if dimension == '2d': 
              losses['train_Temp'][-1] = losses['train_Temp'][-1] + \
                                         hyper_params["criterion"]( predicted[:,:no_depth_levels,:,:],
                                         target_batch[:,:no_depth_levels,:,:] ).item() * input_batch.shape[0]
              losses['train_U'][-1] = losses['train_U'][-1] + \
                                         hyper_params["criterion"]( predicted[:,no_depth_levels:2*no_depth_levels,:,:],
                                         target_batch[:,no_depth_levels:2*no_depth_levels,:,:] ).item() * input_batch.shape[0]
              losses['train_V'][-1] = losses['train_V'][-1] + \
                                         hyper_params["criterion"]( predicted[:,2*no_depth_levels:3*no_depth_levels,:,:],
                                         target_batch[:,2*no_depth_levels:3*no_depth_levels,:,:] ).item() * input_batch.shape[0]
              losses['train_Eta'][-1] = losses['train_Eta'][-1] + \
                                         hyper_params["criterion"]( predicted[:,3*no_depth_levels,:,:],
                                         target_batch[:,3*no_depth_levels,:,:] ).item() * input_batch.shape[0]
           elif dimension == '3d': 
              losses['train_Temp'][-1] = losses['train_Temp'][-1] + hyper_params["criterion"]    \
                                             ( predicted[:,0,:,:,:], target_batch[:,0,:,:,:] ).item() * input_batch.shape[0]
              losses['train_U'][-1] = losses['train_U'][-1] + hyper_params["criterion"]          \
                                             ( predicted[:,1,:,:,:], target_batch[:,1,:,:,:] ).item() * input_batch.shape[0]
              losses['train_V'][-1] = losses['train_V'][-1] + hyper_params["criterion"]          \
                                             ( predicted[:,2,:,:,:], target_batch[:,2,:,:,:] ).item() * input_batch.shape[0]
              losses['train_Eta'][-1] = losses['train_Eta'][-1] + hyper_params["criterion"]      \
                                             ( predicted[:,3,0,:,:], target_batch[:,3,0,:,:] ).item() * input_batch.shape[0]
           #logging.info('updated stored loss values for plotting later')
           #print('updated stored loss values for plotting later')
           #GPUtil.showUtilization() 
   
           if ( epoch%plot_freq == 0  or epoch == hyper_params["num_epochs"]+start_epoch-1 ) and epoch != start_epoch and batch_no == 0:
              # Save details to plot later
              if dimension == '2d':

                 # reshape to give each variable its own dimension
                 tmp_target     = np.zeros(( 4, no_depth_levels, target_batch.shape[2], target_batch.shape[3] ))
                 tmp_prediction = np.zeros(( 4, no_depth_levels, target_batch.shape[2], target_batch.shape[3] ))
                 for i in range(3):
                    tmp_target[i,:,:,:]  = target_batch[0,i*no_depth_levels:(i+1)*no_depth_levels,:,:].cpu().detach().numpy()
                    tmp_prediction[i,:,:,:] = predicted[0,i*no_depth_levels:(i+1)*no_depth_levels,:,:].cpu().detach().numpy()
                 tmp_target[3,:,:,:] = np.broadcast_to( target_batch[0,i*no_depth_levels:(i+1)*no_depth_levels,:,:].cpu().detach().numpy(),
                                                        (1, no_depth_levels, target_batch.shape[2], target_batch.shape[3] ) )
                 tmp_prediction[3,:,:,:] = np.broadcast_to( predicted[0,i*no_depth_levels:(i+1)*no_depth_levels,:,:].cpu().detach().numpy(),
                                                        (1, no_depth_levels, target_batch.shape[2], target_batch.shape[3] ) )

                 plotting_data['targets_train'].append( tmp_target )
                 plotting_data['predictions_train'].append( tmp_prediction )

              elif dimension == '3d':
                 plotting_data['targets_train'].append( target_batch[0,:,:,:,:].cpu().detach().numpy() )
                 plotting_data['predictions_train'].append( predicted[0,:,:,:,:].cpu().detach().numpy() )
           #logging.info('Saved data for scatter plots later')
           #print('Saved data for scatter plots later')
           #GPUtil.showUtilization() 
   
           if epoch == start_epoch and batch_no == 0:
              # Save info to plot histogram of data
              hist_data=[]
              if dimension == '2d':
                 for no_var in range(3):
                    hist_data.append( input_batch[:,no_var*no_depth_levels:(no_var+1)*no_depth_levels,:,:].cpu().reshape(-1) )
                 hist_data.append( input_batch[:,3*no_depth_levels,:,:].cpu().reshape(-1) )
              elif dimension == '3d':
                 for no_var in range(4):
                    hist_data.append( input_batch[:,no_var,:,:,:].cpu().reshape(-1) )
                 hist_data.append( input_batch[:,3,0,:,:].cpu().reshape(-1) )
           #logging.info('Saved data for histograms')
           #print('Saved data for histograms')
           #GPUtil.showUtilization() 

           first_pass = False
           #logging.info('Batch number '+str(batch_no)+' ')
           #TimeCheck(tic,'finished batch training')
           #GPUtil.showUtilization() 

           del batch_sample
           del input_batch
           del target_batch
           del predicted
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
       print('end of training epoch ; '+str(epoch))
       GPUtil.showUtilization() 

       # Validation
   
       batch_no = 0
       
       for batch_sample in val_loader:
   
           input_batch  = batch_sample['input']
           target_batch = batch_sample['output']
       
           # Send inputs and labels to GPU if available
           input_batch = input_batch.to(device, non_blocking=True, dtype=torch.float)
           target_batch = target_batch.to(device, non_blocking=True, dtype=torch.float)
           
           h.train(False)
   
           # get prediction from the model, given the inputs
           predicted = h(input_batch)
       
           # get loss for the predicted output
           val_loss = hyper_params["criterion"](predicted, target_batch)
   
           losses['val'][-1] = losses['val'][-1] + val_loss.item()*input_batch.shape[0]
   
           if ( epoch%plot_freq == 0 or epoch == hyper_params["num_epochs"]+start_epoch-1 ) and epoch != start_epoch and batch_no == 0:
              # Save details to plot later
              if dimension == '2d':

                 # reshape to give each variable its own dimension
                 tmp_target     = np.zeros(( 4, no_depth_levels, target_batch.shape[2], target_batch.shape[3] ))
                 tmp_prediction = np.zeros(( 4, no_depth_levels, target_batch.shape[2], target_batch.shape[3] ))
                 for i in range(3):
                    tmp_target[i,:,:,:]  = target_batch[0,i*no_depth_levels:(i+1)*no_depth_levels,:,:].cpu().detach().numpy()
                    tmp_prediction[i,:,:,:] = predicted[0,i*no_depth_levels:(i+1)*no_depth_levels,:,:].cpu().detach().numpy()
                 tmp_target[3,:,:,:] = np.broadcast_to( target_batch[0,i*no_depth_levels:(i+1)*no_depth_levels,:,:].cpu().detach().numpy(),
                                                        (1, no_depth_levels, target_batch.shape[2], target_batch.shape[3] ) )
                 tmp_prediction[3,:,:,:] = np.broadcast_to( predicted[0,i*no_depth_levels:(i+1)*no_depth_levels,:,:].cpu().detach().numpy(),
                                                        (1, no_depth_levels, target_batch.shape[2], target_batch.shape[3] ) )

                 plotting_data['targets_val'].append( tmp_target )
                 plotting_data['predictions_val'].append( tmp_prediction )

              elif dimension == '3d':
                 
                 plotting_data['targets_val'].append( target_batch[0,:,:,:].cpu().detach().numpy() )
                 plotting_data['predictions_val'].append( predicted[0,:,:,:].cpu().detach().numpy() )
   
           del batch_sample
           del input_batch
           del target_batch
           del predicted
           del val_loss
           gc.collect()
           torch.cuda.empty_cache()

           batch_no = batch_no + 1
           #logging.info('batch number '+str(batch_no)+' ')
           #TimeCheck(tic, 'batch complete')
           #GPUtil.showUtilization() 
   
       logging.info('epoch {}, validation loss {}'.format(epoch, losses['val'][-1])+'\n')
       losses['val'][-1] = losses['val'][-1] / no_val_samples
   
       if losses['val'][-1] < current_best_loss :
           ### SAVE IT ###
           logging.info('save model \n')
           for model_file in glob.glob('../../../Channel_nn_Outputs/'+model_name+'/MODELS/'+model_name+'_epoch*_SavedBESTModel.pt'):
              os.remove(model_file)
           pkl_filename = '../../../Channel_nn_Outputs/'+model_name+'/MODELS/'+model_name+'_epoch'+str(epoch)+'_SavedBESTModel.pt'
           torch.save({
                       'epoch': epoch,
                       'model_state_dict': h.state_dict(),
                       'optimizer_state_dict': optimizer.state_dict(),
                       'loss': loss,
                       'losses': losses,
                       }, pkl_filename)
           current_best_loss = losses['val'][-1]
  
       if ( epoch%save_freq == 0 or epoch == hyper_params["num_epochs"]+start_epoch-1 ) and epoch != start_epoch :
           ### SAVE IT ###
           logging.info('save model \n')
           pkl_filename = '../../../Channel_nn_Outputs/'+model_name+'/MODELS/'+model_name+'_epoch'+str(epoch)+'_SavedModel.pt'
           torch.save({
                       'epoch': epoch,
                       'model_state_dict': h.state_dict(),
                       'optimizer_state_dict': optimizer.state_dict(),
                       'loss': loss,
                       'losses': losses,
                       }, pkl_filename)
  
       TimeCheck(tic, 'Epoch finished')
       print('end of epoch ; '+str(epoch))
       GPUtil.showUtilization() 
   
   logging.info('hyper_params:    '+str(hyper_params)+'\n')
 
   logging.info(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20)) 
 
   logging.info('End of train model')
   return(losses, plotting_data, hist_data)

def LoadModel(model_name, h, optimizer, saved_epoch, tr_inf, losses):
   pkl_filename = '../../../Channel_nn_Outputs/'+model_name+'/MODELS/'+model_name+'_epoch'+str(saved_epoch)+'_SavedModel.pt'
   checkpoint = torch.load(pkl_filename)
   h.load_state_dict(checkpoint['model_state_dict'])
   optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
   epoch = checkpoint['epoch']
   loss = checkpoint['loss']
   losses = checkpoint['losses']

   if tr_inf == 'tr':
      h.train()
   elif tr_inf == 'inf':
      h.eval()

 
   return losses, h, optimizer

def plot_training_output(model_name, start_epoch, total_epochs, plot_freq, losses, plotting_data, hist_data):
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
 
   ## Plot scatter plots of errors after various epochs
 
   for plot_no in range(len(plotting_data['targets_train'])):

      epoch = min(start_epoch + (plot_no+1)*plot_freq, total_epochs)

      fig_train = plt.figure(figsize=(9,9))
      ax_temp  = fig_train.add_subplot(221)
      ax_U     = fig_train.add_subplot(222)
      ax_V     = fig_train.add_subplot(223)
      ax_Eta   = fig_train.add_subplot(224)

      ax_temp.scatter(plotting_data['targets_train'][plot_no][0,:,:,:],
                      plotting_data['predictions_train'][plot_no][0,:,:,:],
                      edgecolors=(0, 0, 0), alpha=0.15, color='blue', label='Temp')
      ax_U.scatter(plotting_data['targets_train'][plot_no][1,:,:,:],
                   plotting_data['predictions_train'][plot_no][1,:,:,:],
                   edgecolors=(0, 0, 0), alpha=0.15, color='red', label='U')
      ax_V.scatter(plotting_data['targets_train'][plot_no][2,:,:,:], 
                   plotting_data['predictions_train'][plot_no][2,:,:,:],
                   edgecolors=(0, 0, 0), alpha=0.15, color='orange', label='V')
      ax_Eta.scatter(plotting_data['targets_train'][plot_no][3,0,:,:], 
                     plotting_data['predictions_train'][plot_no][3,0,:,:],
                     edgecolors=(0, 0, 0), alpha=0.15, color='purple', label='Eta')
      bottom = min( np.amin(plotting_data['targets_train'][plot_no]), np.amin(plotting_data['predictions_train'][plot_no]) )
      top    = max( np.amax(plotting_data['targets_train'][plot_no]), np.amax(plotting_data['predictions_train'][plot_no]) )  
   
      ax_temp.set_xlabel('Truth')
      ax_temp.set_ylabel('Predictions')
      ax_temp.set_title('Training errors, Temperature, epoch '+str(epoch))
      ax_temp.set_xlim(bottom, top)
      ax_temp.set_ylim(bottom, top)
      ax_temp.plot([bottom, top], [bottom, top], 'k--', lw=1, color='black')

      ax_U.set_xlabel('Truth')
      ax_U.set_ylabel('Predictions')
      ax_U.set_title('Training errors, U velocity, epoch '+str(epoch))
      ax_U.set_xlim(bottom, top)
      ax_U.set_ylim(bottom, top)
      ax_U.plot([bottom, top], [bottom, top], 'k--', lw=1, color='black')

      ax_V.set_xlabel('Truth')
      ax_V.set_ylabel('Predictions')
      ax_V.set_title('Training errors, V velocity, epoch '+str(epoch))
      ax_V.set_xlim(bottom, top)
      ax_V.set_ylim(bottom, top)
      ax_V.plot([bottom, top], [bottom, top], 'k--', lw=1, color='black')

      ax_Eta.set_xlabel('Truth')
      ax_Eta.set_ylabel('Predictions')
      ax_Eta.set_title('Training errors, Sea Surface Height, epoch '+str(epoch))
      ax_Eta.set_xlim(bottom, top)
      ax_Eta.set_ylim(bottom, top)
      ax_Eta.plot([bottom, top], [bottom, top], 'k--', lw=1, color='black')

      plt.savefig(plot_dir+'/'+model_name+'_scatter_epoch'+str(epoch).rjust(3,'0')+'training.png',
                  bbox_inches = 'tight', pad_inches = 0.1)
      plt.close()

      # plotting histograms of data	§
      no_bins = 50
      labels = ['Temp', 'U_vel', 'V_vel', 'Eta']
      for no_var in range(4):
          fig_hist = plt.figure(figsize=(10, 8))
          plt.hist(hist_data[no_var], bins = no_bins)
          plt.savefig(plot_dir+'/'+model_name+'_histogram_'+str(labels[no_var])+'.png', 
                      bbox_inches = 'tight', pad_inches = 0.1)
          plt.close()

def OutputStats(model_name, MeanStd_prefix, mitgcm_filename, data_loader, h, no_epochs, y_dim_used, dimension):
   #-------------------------------------------------------------
   # Output the rms and correlation coefficients over a dataset
   #-------------------------------------------------------------
   #  1. Make predictions over entire dataset (training, val or test)
   #  2. Store predictions and targets in a large array (size of samples in dataset) 
   #     (if this is not possible, cause of memory, need to add RMS errors as 
   #     we go, and the variables needed to calculate the correlation coeff)
   #  3. Calculate the RMS and correlation coefficient of the prediction and target arrays
   #  4. Store the RMS and correlation coeffs in netcdf file, along with ~ 50 samples
   #     (saving all samples takes up loads of space and has little benefit)

   device = 'cuda' if torch.cuda.is_available() else 'cpu'

   data_mean, data_std, data_range, no_in_channels, no_out_channels = ReadMeanStd(MeanStd_prefix)

   # Read in grid data from MITgcm file
   mitgcm_ds = xr.open_dataset(mitgcm_filename)
   da_X = mitgcm_ds['X']
   da_Y = mitgcm_ds['Yp1']
   da_Z = mitgcm_ds['diag_levels'] 

   FirstPass = True

   batch_no = 0

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
      for batch_sample in data_loader:
  
          print('batch_no; '+str(batch_no))
          input_batch  = batch_sample['input']
          target_batch = batch_sample['output']
          input_batch = input_batch.to(device, non_blocking=True, dtype=torch.float)
          target_batch = target_batch.to(device, non_blocking=True, dtype=torch.float)
      
          # get prediction from the model, given the inputs
          if isinstance(h, list):
             predicted = h[0](input_batch)
             for i in range(1, len(h)):
                predicted = predicted + h[i](input_batch)
             predicted = predicted / len(h)
          else:
             predicted = h(input_batch)
     
          if FirstPass:
             if dimension == '2d': 
                Mask_channels = input_batch[0,no_out_channels:,:,:].cpu().detach().numpy()
             elif dimension == '3d': 
                 Mask_channels = input_batch[0,no_out_channels:,:,:,:].cpu().detach().numpy()
             targets     = target_batch.cpu().detach().numpy()
             predictions = predicted.cpu().detach().numpy() 
          else:
             targets     = np.concatenate( ( targets, target_batch.cpu().detach().numpy() ), axis=0)
             predictions = np.concatenate( ( predictions, predicted.cpu().detach().numpy() ), axis=0)

          del input_batch
          del target_batch
          del predicted
          gc.collect()
          torch.cuda.empty_cache()
 
          FirstPass = False
          batch_no = batch_no + 1

   print('batches done')

   logging.info(targets.shape)
   logging.info(predictions.shape)

   # Denormalise
   targets = rr.RF_DeNormalise(targets, data_mean, data_std, data_range, no_out_channels)
   predictions = rr.RF_DeNormalise(predictions, data_mean, data_std, data_range, no_out_channels)
   print('de-normalised')
   
   # Set up netcdf files
   nc_filename = '../../../Channel_nn_Outputs/'+model_name+'/STATS/'+model_name+'_'+str(no_epochs)+'epochs_StatsOutput.nc'
   nc_file = nc4.Dataset(nc_filename,'w', format='NETCDF4') #'w' stands for write
   # Create Dimensions
   no_depth_levels = 38  # Hard coded...perhaps should change...?
   no_samples      = min(50, targets.shape[0]) 
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

   nc_TempCC     = nc_file.createVariable( 'TempCC'     , 'f4', ('Z', 'Y', 'X')      )
   nc_U_CC       = nc_file.createVariable( 'U_CC'       , 'f4', ('Z', 'Y', 'X')      )
   nc_V_CC       = nc_file.createVariable( 'V_CC'       , 'f4', ('Z', 'Y', 'X')      )
   nc_EtaCC      = nc_file.createVariable( 'EtaCC'      , 'f4', ('Y', 'X')           )

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

      nc_TempMask[:,:,:] = Mask_channels[0:no_depth_levels,:,:]
      nc_UMask[:,:,:]    = Mask_channels[1*no_depth_levels:2*no_depth_levels,:,:]
      nc_VMask[:,:,:]    = Mask_channels[2*no_depth_levels:3*no_depth_levels,:,:]
      nc_EtaMask[:,:]    = Mask_channels[3*no_depth_levels,:,:]

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

      nc_TempRMS[:,:,:] = np.sqrt( np.mean( np.square(
                                                predictions[:,0:no_depth_levels,:,:]
                                                - targets[:,0:no_depth_levels,:,:]), axis=0) )
      nc_U_RMS[:,:,:]   = np.sqrt( np.mean( np.square(
                                                predictions[:,1*no_depth_levels:2*no_depth_levels,:,:] 
                                                - targets[:,1*no_depth_levels:2*no_depth_levels,:,:]), axis=0) )
      nc_V_RMS[:,:,:]   = np.sqrt( np.mean( np.square(
                                                predictions[:,2*no_depth_levels:3*no_depth_levels,:,:]
                                                - targets[:,2*no_depth_levels:3*no_depth_levels,:,:]), axis=0) )
      nc_EtaRMS[:,:]    = np.sqrt( np.mean( np.square(
                                                predictions[:,3*no_depth_levels,:,:]
                                                - targets[:,3*no_depth_levels,:,:]), axis=0) )

      if np.isnan(predictions).any():
         logging.info('predictions contains a NaN')
         logging.info('nans at '+str(np.argwhere(np.isnan(predictions))) )
      if np.isnan(targets).any():
         logging.info('targets contains a NaN')
         logging.info('nans at '+str(np.argwhere(np.isnan(targets))) )
      for x in range(targets.shape[3]):
        for y in range(targets.shape[2]):
           for z in range(no_depth_levels):
              nc_TempCC[z,y,x] = np.corrcoef( predictions[:,                  z,y,x], 
                                                  targets[:,                  z,y,x], rowvar=False )[0,1]
              nc_U_CC[z,y,x]   = np.corrcoef( predictions[:,1*no_depth_levels+z,y,x],
                                                  targets[:,1*no_depth_levels+z,y,x], rowvar=False )[0,1]
              nc_V_CC[z,y,x]   = np.corrcoef( predictions[:,2*no_depth_levels+z,y,x], 
                                                  targets[:,2*no_depth_levels+z,y,x], rowvar=False )[0,1]
           nc_EtaCC[y,x]  = np.corrcoef( predictions[:,3*no_depth_levels,y,x],
                                             targets[:,3*no_depth_levels,y,x], rowvar=False )[0,1]
   
   elif dimension=='3d':
      nc_TrueTemp[:,:,:,:] = targets[:no_samples,0,:,:,:] 
      nc_TrueU[:,:,:,:]    = targets[:no_samples,1,:,:,:]
      nc_TrueV[:,:,:,:]    = targets[:no_samples,2,:,:,:]
      nc_TrueEta[:,:,:]    = targets[:no_samples,3,0,:,:]

      nc_TempMask[:,:,:] = Mask_channels[0,:,:,:]
      nc_UMask[:,:,:]    = Mask_channels[1,:,:,:]
      nc_VMask[:,:,:]    = Mask_channels[2,:,:,:]
      nc_EtaMask[:,:]    = Mask_channels[3,0,:,:]

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

      nc_TempRMS[:,:,:] = np.sqrt( np.mean( np.square(
                                                predictions[:,0,:,:,:]
                                                - targets[:,0,:,:,:]), axis=0) )
      nc_U_RMS[:,:,:]   = np.sqrt( np.mean( np.square(
                                                predictions[:,1,:,:,:] 
                                                - targets[:,1,:,:,:]), axis=0) )
      nc_V_RMS[:,:,:]   = np.sqrt( np.mean( np.square(
                                                predictions[:,2,:,:,:]
                                                - targets[:,2,:,:,:]), axis=0) )
      nc_EtaRMS[:,:]    = np.sqrt( np.mean( np.square(
                                                predictions[:,3,0,:,:]
                                                - targets[:,3,0,:,:]), axis=0) )

      if np.isnan(predictions).any():
         logging.info('predictions contains a NaN')
         logging.info('nans at '+str(np.argwhere(np.isnan(predictions))) )
      if np.isnan(targets).any():
         logging.info('targets contains a NaN')
         logging.info('nans at '+str(np.argwhere(np.isnan(targets))) )
      for x in range(targets.shape[3]):
        for y in range(targets.shape[2]):
           for z in range(no_depth_levels):
              nc_TempCC[z,y,x] = np.corrcoef( predictions[:,0,z,y,x], targets[:,0,z,y,x], rowvar=False )[0,1]
              nc_U_CC[z,y,x]   = np.corrcoef( predictions[:,1,z,y,x], targets[:,1,z,y,x], rowvar=False )[0,1]
              nc_V_CC[z,y,x]   = np.corrcoef( predictions[:,2,z,y,x], targets[:,2,z,y,x], rowvar=False )[0,1]
           nc_EtaCC[y,x]  = np.corrcoef( predictions[:,3,0,y,x], targets[:,3,0,y,x], rowvar=False )[0,1]
   
def IterativelyPredict(model_name, MeanStd_prefix, mitgcm_filename, Iterate_Dataset, h, start, for_len, no_epochs, y_dim_used, land, dimension):
   #-------------------------------------------------------------

   device = 'cuda' if torch.cuda.is_available() else 'cpu'

   data_mean, data_std, data_range, no_in_channels, no_out_channels = ReadMeanStd(MeanStd_prefix)

   no_depth_levels = 38  # Hard coded...perhaps should change...?

   # Read in grid data from MITgcm file
   mitgcm_ds = xr.open_dataset(mitgcm_filename)
   da_X = mitgcm_ds['X']
   da_Y = mitgcm_ds['Yp1']
   da_Z = mitgcm_ds['diag_levels'] 

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
   if dimension == '2d':
      input_sample = Iterate_Dataset.__getitem__(start)['input'][:,:,:].unsqueeze(0).numpy()
      iterated_predictions = input_sample[:,:no_out_channels,:,:]
      MITgcm_data = input_sample[:,:no_out_channels,:,:]
      Mask_channels = input_sample[:,no_out_channels:,:,:]
   elif dimension == '3d':
      input_sample = Iterate_Dataset.__getitem__(start)['input'][:,:,:,:].unsqueeze(0).numpy()
      iterated_predictions = input_sample[:,:no_out_channels,:,:,:]
      MITgcm_data = input_sample[:,:no_out_channels,:,:,:]
      Mask_channels = input_sample[:,no_out_channels:,:,:,:]

   # Make iterative forecast, saving predictions as we go, and pull out MITgcm data
   with torch.no_grad():
      for time in range(for_len):
         print(time)
         if isinstance(h, list):
            predicted = h[0]( torch.from_numpy(input_sample).to(device, non_blocking=True, dtype=torch.float) ).cpu().detach().numpy()
            for i in range(1, len(h)):
               predicted = predicted + \
                           h[i]( torch.from_numpy(input_sample).to(device, non_blocking=True, dtype=torch.float) ).cpu().detach().numpy()
            predicted = predicted / len(h)
         else:
            predicted = h( torch.from_numpy(input_sample).to(device, non_blocking=True, dtype=torch.float) ).cpu().detach().numpy()

         iterated_predictions = np.concatenate( ( iterated_predictions, predicted ), axis=0 )

         if dimension == '2d':
            MITgcm_data = np.concatenate( ( MITgcm_data, 
                           Iterate_Dataset.__getitem__(start+time+1)['input'][:no_out_channels,:,:].unsqueeze(0).numpy() ), axis=0 )
         elif dimension == '3d':
            MITgcm_data = np.concatenate( ( MITgcm_data, 
                           Iterate_Dataset.__getitem__(start+time+1)['input'][:no_out_channels,:,:,:].unsqueeze(0).numpy() ), axis=0 )

         # Cat mask channels (which are const) to the output ready for inputs for next predictions
         input_sample = np.concatenate( (predicted, Mask_channels), axis = 1 )

         # Remask land on outputs ready for next iteration - NEED TO TEST!
         if dimension == '2d':
            for i in range(3): 
               input_sample[:,i*no_depth_levels:(i+1)*no_depth_levels,:,:] = \
                   np.where(Mask_channels[:,i*no_depth_levels:(i+1)*no_depth_levels,:,:]==1, 
                            0, input_sample[:,i*no_depth_levels:(i+1)*no_depth_levels,:,:])
            input_sample[:,3*no_depth_levels,:,:] = np.where(Mask_channels[:,3*no_depth_levels,:,:]==1, 
                                                             0, input_sample[:,3*no_depth_levels,:,:])
         elif dimension == '3d':
            input_sample[:,0,:,:,:] = np.where(Mask_channels[:,0,:,:,:]==1, 0, input_sample[:,0,:,:,:])
            input_sample[:,1,:,:,:] = np.where(Mask_channels[:,1,:,:,:]==1, 0, input_sample[:,1,:,:,:])
            input_sample[:,2,:,:,:] = np.where(Mask_channels[:,2,:,:,:]==1, 0, input_sample[:,2,:,:,:])
            input_sample[:,3,:,:,:] = np.where(Mask_channels[:,3,:,:,:]==1, 0, input_sample[:,3,:,:,:])

         del predicted
         gc.collect()
         torch.cuda.empty_cache()

   logging.info('iterated_predictions.shape') 
   logging.info(iterated_predictions.shape) 
   logging.info('MITgcm_data.shape') 
   logging.info(MITgcm_data.shape) 
   # Denormalise 
   iterated_predictions = rr.RF_DeNormalise(iterated_predictions, data_mean, data_std, data_range, no_out_channels)
   MITgcm_data          = rr.RF_DeNormalise(MITgcm_data,          data_mean, data_std, data_range, no_out_channels)

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
      nc_TempMask[:,:,:] = Mask_channels[0,0:no_depth_levels,:,:]
      nc_UMask[:,:,:]    = Mask_channels[0,1*no_depth_levels:2*no_depth_levels,:,:]
      nc_VMask[:,:,:]    = Mask_channels[0,2*no_depth_levels:3*no_depth_levels,:,:]
      nc_EtaMask[:,:]    = Mask_channels[0,3*no_depth_levels,:,:]

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
      nc_TempMask[:,:,:] = Mask_channels[0,0,:,:,:]
      nc_UMask[:,:,:]    = Mask_channels[0,1,:,:,:]
      nc_VMask[:,:,:]    = Mask_channels[0,2,:,:,:]
      nc_EtaMask[:,:]    = Mask_channels[0,3,0,:,:]

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
