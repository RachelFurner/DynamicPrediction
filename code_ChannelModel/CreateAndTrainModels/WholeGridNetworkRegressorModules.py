#!/usr/bin/env python
# coding: utf-8

# Code to train a network to learn the dynamic evolution of an MITgcm channel configuration
# Designed to take the entire field for all variables, apply a NN (with conv layers), and output
# the entire field for all variables one day later (i.e. the next iteration of the MITgcm netcdf 
# file here, although note this is 2 steps of the underlying model which runs a 12 hourly timestep.
# The data is subsampled in time to give quasi-independence

import sys
sys.path.append('../Tools')
import ReadRoutines as rr
import numpy as np
import os
import glob
import xarray as xr
import torch
from torch.profiler import profile, record_function, ProfilerActivity
import netCDF4 as nc4
import gc as gc
import logging
import wandb
import gcm_filters as gcm_filters
os.environ[ 'MPLCONFIGDIR' ] = '/data/hpcdata/users/racfur/tmp/'
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pickle
import random
from scipy import stats

def ReadMeanStd(mean_std_file, dim, no_phys_in_channels, no_out_channels, z_dim, seed_value):

   os.environ['PYTHONHASHSEED'] = str(seed_value)
   np.random.seed(seed_value)
   torch.manual_seed(seed_value)
   torch.cuda.manual_seed(seed_value)
   torch.cuda.manual_seed_all(seed_value)
   torch.backends.cudnn.enabled = False
   torch.backends.cudnn.deterministic = True
   torch.backends.cudnn.benchmark = False
   torch.use_deterministic_algorithms(True, warn_only=True)
   torch.backends.cuda.matmul.allow_tf32 = True

   mean_std_data = np.load(mean_std_file)
   inputs_mean  = mean_std_data['arr_0']
   inputs_std   = mean_std_data['arr_1']
   inputs_range = mean_std_data['arr_2']
   targets_mean  = mean_std_data['arr_3']
   targets_std   = mean_std_data['arr_4']
   targets_range = mean_std_data['arr_5']

   # Expand arrays to multiple levels for 2d approach
   if dim =='2d':
      tmp_inputs_mean  = np.zeros((no_phys_in_channels))
      tmp_inputs_std   = np.zeros((no_phys_in_channels))
      tmp_inputs_range = np.zeros((no_phys_in_channels))
      tmp_targets_mean  = np.zeros((no_out_channels))
      tmp_targets_std   = np.zeros((no_out_channels))
      tmp_targets_range = np.zeros((no_out_channels))

      for var in range(3):
         tmp_inputs_mean[var*z_dim:(var+1)*z_dim] = inputs_mean[var]
         tmp_inputs_std[var*z_dim:(var+1)*z_dim] = inputs_std[var]
         tmp_inputs_range[var*z_dim:(var+1)*z_dim] = inputs_range[var]
         tmp_targets_mean[var*z_dim:(var+1)*z_dim] = targets_mean[var]
         tmp_targets_std[var*z_dim:(var+1)*z_dim] = targets_std[var]
         tmp_targets_range[var*z_dim:(var+1)*z_dim] = targets_range[var]
      tmp_targets_mean[3*z_dim] = targets_mean[3]
      tmp_targets_std[3*z_dim] = targets_std[3]
      tmp_targets_range[3*z_dim] = targets_range[3]
      for var in range(3,6):
         tmp_inputs_mean[3*z_dim+var-3] = inputs_mean[var]
         tmp_inputs_std[3*z_dim+var-3] = inputs_std[var]
         tmp_inputs_range[3*z_dim+var-3] = inputs_range[var]

      inputs_mean  = tmp_inputs_mean
      inputs_std   = tmp_inputs_std 
      inputs_range = tmp_inputs_range
      targets_mean  = tmp_targets_mean
      targets_std   = tmp_targets_std 
      targets_range = tmp_targets_range

   return(inputs_mean, inputs_std, inputs_range, targets_mean, targets_std, targets_range)

def mse_loss(output, target, z_dim):
   loss = torch.mean( (output - target)**2 )
   return loss

def cons_loss(output, target, z_dim):
   loss = ( torch.mean( (output - target)**2 )     # MSE over all space and samples
          + 0.01 * torch.mean( ( torch.nansum(output[:,3*z_dim,:,:], dim=(1,2)) - 
                               torch.nansum(target[:,3*z_dim,:,:], dim=(1,2)) )**2 )  # MSE in total mass content over samples
          + 0.01 * torch.mean( ( torch.nansum( torch.add(torch.square(output[:, z_dim:2*z_dim,:,:]),
                                                       torch.square(output[:, z_dim:2*z_dim,:,:])), dim=(1,2,3) )
                             - torch.nansum( torch.add(torch.square(output[:,2*z_dim:3*z_dim,:,:]),
                                                       torch.square(output[:,2*z_dim:3*z_dim,:,:])), dim=(1,2,3) ) )**2 ) ) #TKE MSE over samples
   return loss

def TrainModel(model_name, model_style, dimension, land, histlen, rolllen, TEST, no_tr_samples, no_val_samples, save_freq, train_loader,
               val_loader, h, optimizer, num_epochs, seed_value, losses, channel_dim, no_phys_in_channels, no_out_channels,
               wandb, start_epoch=1, current_best_loss=0.1):

   device = 'cuda' if torch.cuda.is_available() else 'cpu'

   z_dim = 38  # Hard coded...perhaps should change...?
   # Set variables to remove randomness and ensure reproducible results
   os.environ['PYTHONHASHSEED'] = str(seed_value)
   np.random.seed(seed_value)
   torch.manual_seed(seed_value)
   torch.cuda.manual_seed(seed_value)
   torch.cuda.manual_seed_all(seed_value)
   torch.backends.cudnn.enabled = False
   torch.backends.cudnn.deterministic = True
   torch.backends.cudnn.benchmark = False
   torch.use_deterministic_algorithms(True, warn_only=True)

   logging.info('###### TRAINING MODEL ######')
   for epoch in range( start_epoch, num_epochs+start_epoch ):

       # Training
       losses['train'].append(0.)
       losses['train_Temp'].append(0.)
       losses['train_U'].append(0.)
       losses['train_V'].append(0.)
       losses['train_Eta'].append(0.)
       losses['val'].append(0.)

       h.train(True)

       for input_batch, target_batch, extrafluxes_batch, Tmask, Umask, Vmask, out_masks, _ in train_loader:

           # Clear gradient buffers - dont want to cummulate gradients
           optimizer.zero_grad()
       
           #with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
           #              record_shapes=True, with_stack=True, profile_memory=True) as prof:
           #   with record_function("training"):
           if True:
                 target_batch = target_batch.to(device, non_blocking=True)
                 # get prediction from the model, given the inputs
                 # multiply by masks, to give 0 at land values, to circumvent issues with normalisation!
                 if rolllen == 1:
                    if land == 'ExcLand':
                       predicted_batch = h( input_batch ) * out_masks.to(device, non_blocking=True)
                    else:
                       predicted_batch = h( torch.cat((input_batch, Tmask, Umask, Vmask), dim=channel_dim) ) * out_masks.to(device, non_blocking=True)
                 else:
                    if dimension != '2d' and model_style != 'UNet2dTransp':
                       sys.exit('Cannot currently use rolllen>1 with options other than 2d tansp network')
                    predicted_batch = torch.zeros(target_batch.shape, device=device)
                    # If rolllen>1 calc iteratively for 'roll out' loss function
                    for roll_time in range(rolllen):
                       torch.cuda.empty_cache()
                       if land == 'ExcLand':
                          predicted_batch[:,roll_time*no_out_channels:(roll_time+1)*no_out_channels,:,:] =  \
                                                       h( input_batch ) * out_masks.to(device, non_blocking=True)
                       else:
                          predicted_batch[:,roll_time*no_out_channels:(roll_time+1)*no_out_channels,:,:] =  \
                                                  h( torch.cat((input_batch, Tmask, Umask, Vmask), dim=channel_dim) ) * out_masks.to(device, non_blocking=True)
                       for hist_time in range(histlen-1):
                          input_batch[:, hist_time*no_phys_in_channels:(hist_time+1)*no_phys_in_channels, :, :] =   \
                                    input_batch[:, (hist_time+1)*no_phys_in_channels:(hist_time+2)*no_phys_in_channels, :, :]
                       input_batch[:, (histlen-1)*no_phys_in_channels:, :, :] =  \
                                    torch.cat( ( (input_batch[:,(histlen-1)*no_phys_in_channels:histlen*no_phys_in_channels-2,:,:].to(
                                                                                                            device, non_blocking=True)
                                                  + predicted_batch[:,roll_time*no_out_channels:(roll_time+1)*no_out_channels,:,:]),
                                                extrafluxes_batch[:,roll_time*2:roll_time*2+2,:,:].to(device, non_blocking=True) ),
                                              dim=1 )
      	         
                 # Calculate and update loss values
                 
                 loss = mse_loss(predicted_batch, target_batch, z_dim).double()
                 losses['train'][-1] = losses['train'][-1] + loss.item() * input_batch.shape[0]
                 torch.cuda.empty_cache()
      
                 if model_style == 'ConvLSTM' or model_style == 'UNetConvLSTM':
                    if dimension == '2d': 
                       losses['train_Temp'][-1] = losses['train_Temp'][-1] + \
                                                  mse_loss( predicted_batch[:,:z_dim,:,:],
      	                			            target_batch[:,:z_dim,:,:],
                                                            z_dim).item() * input_batch.shape[0]
                       losses['train_U'][-1] = losses['train_U'][-1] + \
                                               mse_loss( predicted_batch[:,z_dim:2*z_dim,:,:],
                                                         target_batch[:,z_dim:2*z_dim,:,:],
                                                         z_dim).item() * input_batch.shape[0]
                       losses['train_V'][-1] = losses['train_V'][-1] + \
                                               mse_loss( predicted_batch[:,2*z_dim:3*z_dim,:,:],
                                                         target_batch[:,2*z_dim:3*z_dim,:,:],
                                                         z_dim).item() * input_batch.shape[0]
                       losses['train_Eta'][-1] = losses['train_Eta'][-1] + \
                                                 mse_loss( predicted_batch[:,3*z_dim,:,:],
                                                           target_batch[:,3*z_dim,:,:],
                                                           z_dim).item() * input_batch.shape[0]
                 else:  
                    if dimension == '2d': 
                       losses['train_Temp'][-1] = losses['train_Temp'][-1] + \
                                                  mse_loss( predicted_batch[:,:z_dim,:,:],
                                                            target_batch[:,:z_dim,:,:],
                                                            z_dim).item() * input_batch.shape[0]
                       losses['train_U'][-1] = losses['train_U'][-1] + \
                                               mse_loss( predicted_batch[:,z_dim:2*z_dim,:,:],
                                                         target_batch[:,z_dim:2*z_dim,:,:],
                                                         z_dim).item() * input_batch.shape[0]
                       losses['train_V'][-1] = losses['train_V'][-1] + \
                                               mse_loss( predicted_batch[:,2*z_dim:3*z_dim,:,:],
                                                         target_batch[:,2*z_dim:3*z_dim,:,:],
                                                         z_dim).item() * input_batch.shape[0]
                       losses['train_Eta'][-1] = losses['train_Eta'][-1] + \
                                                 mse_loss( predicted_batch[:,3*z_dim,:,:],
                                                           target_batch[:,3*z_dim,:,:],
                                                           z_dim).item() * input_batch.shape[0]
                    elif dimension == '3d': 
                       losses['train_Temp'][-1] = losses['train_Temp'][-1] + \
                                                  mse_loss( predicted_batch[:,0,:,:,:], target_batch[:,0,:,:,:], z_dim ).item() * input_batch.shape[0]
                       losses['train_U'][-1]    = losses['train_U'][-1] + \
                                                  mse_loss( predicted_batch[:,1,:,:,:], target_batch[:,1,:,:,:], z_dim ).item() * input_batch.shape[0]
                       losses['train_V'][-1]    = losses['train_V'][-1] + \
                                                  mse_loss( predicted_batch[:,2,:,:,:], target_batch[:,2,:,:,:], z_dim ).item() * input_batch.shape[0]
                       losses['train_Eta'][-1]  = losses['train_Eta'][-1] + \
                                                  mse_loss( predicted_batch[:,3,:,:,:], target_batch[:,3,:,:,:], z_dim ).item() * input_batch.shape[0]
        
                 # get gradients w.r.t to parameters
                 loss.backward()
                 
                 # update parameters
                 optimizer.step()
         
           del input_batch
           del target_batch
           del predicted_batch
           del extrafluxes_batch
           del Tmask
           del Umask
           del Vmask
           del out_masks
           gc.collect()
           torch.cuda.empty_cache()
      
           #print('test line 3')
           #print('')
           #print('')
           #print('sorting by CPU time')
           #print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
           #print('')
           #print('')
           #print('sorting by Cuda time')
           #print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
           #print('')
           #print('')
           #print('sorting by CPU memory')
           #print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
           #print('')
           #print('')
           #print('sorting by cuda memory')
           #print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))
           #prof.export_chrome_trace("trace.json")
           #print('end of batch')
 
       losses['train'][-1] = np.sqrt(losses['train'][-1] / no_tr_samples)
       losses['train_Temp'][-1] = np.sqrt(losses['train_Temp'][-1] / no_tr_samples)
       losses['train_U'][-1] = np.sqrt(losses['train_U'][-1] / no_tr_samples)
       losses['train_V'][-1] = np.sqrt(losses['train_V'][-1] / no_tr_samples)
       losses['train_Eta'][-1] = np.sqrt(losses['train_Eta'][-1] / no_tr_samples)

       if wandb:
          wandb.log({"train loss" : losses['train'][-1]})
       logging.info('epoch {}, training loss {}'.format(epoch, losses['train'][-1])+'\n')

       #### Validation ######
   
       h.train(False)

       with torch.no_grad():
          for input_batch, target_batch, extrafluxes_batch, Tmask, Umask, Vmask, out_masks, _ in val_loader:
   
              target_batch = target_batch.to(device, non_blocking=True)
              # get prediction from the model, given the inputs
              if rolllen == 1:
                 if land == 'ExcLand':
                    predicted_batch = h( input_batch ) * out_masks.to(device, non_blocking=True)
                 else:
                    predicted_batch = h( torch.cat((input_batch, Tmask, Umask, Vmask), dim=channel_dim) ) * out_masks.to(device, non_blocking=True)
              else:
                 if dimension != '2d' and model_style != 'UNet2dTransp':
                    sys.exit('Cannot currently use rolllen>1 with options other than 2d tansp network')
                 predicted_batch = torch.zeros(target_batch.shape, device=device)
                 for roll_time in range(rolllen):
                    if land == 'ExcLand':
                       predicted_batch[:,roll_time*no_out_channels:(roll_time+1)*no_out_channels,:,:] = \
                                 h( input_batch ) * out_masks.to(device, non_blocking=True)
                    else:
                       predicted_batch[:,roll_time*no_out_channels:(roll_time+1)*no_out_channels,:,:] = \
                                 h( torch.cat((input_batch, Tmask, Umask, Vmask), dim=channel_dim) ) * out_masks.to(device, non_blocking=True)
                    for hist_time in range(histlen-1):
                       input_batch[:, hist_time*no_phys_in_channels:(hist_time+1)*no_phys_in_channels, :, :] =   \
                                 input_batch[:, (hist_time+1)*no_phys_in_channels:(hist_time+2)*no_phys_in_channels, :, :]
                    input_batch[:, (histlen-1)*no_phys_in_channels:, :, :] = \
                                 torch.cat( ( (input_batch[:,(histlen-1)*no_phys_in_channels:histlen*no_phys_in_channels-2,:,:].to(
                                                                                                         device, non_blocking=True)
                                               + predicted_batch[:,roll_time*no_out_channels:(roll_time+1)*no_out_channels,:,:]),
                                             extrafluxes_batch[:,roll_time*2:roll_time*2+2,:,:].to(device, non_blocking=True) ),
                                           dim=1 )
          
              # get loss for the predicted_batch output
              losses['val'][-1] = losses['val'][-1] +  \
                                  mse_loss( predicted_batch, target_batch, z_dim ).item() * input_batch.shape[0]
   
              del input_batch
              del target_batch
              del predicted_batch
              gc.collect()
              torch.cuda.empty_cache()
 
          losses['val'][-1] = np.sqrt(losses['val'][-1] / no_val_samples)

          if wandb:
             wandb.log({"val loss" : losses['val'][-1]})
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
          if ( epoch%save_freq == 0 or epoch == num_epochs+start_epoch-1 ) and epoch != 0 :
             pkl_filename = '../../../Channel_nn_Outputs/'+model_name+'/MODELS/'+model_name+'_epoch'+str(epoch)+'_SavedModel.pt'
             torch.save({
                         'epoch': epoch,
                         'model_state_dict': h.state_dict(),
                         'optimizer_state_dict': optimizer.state_dict(),
                         'loss': loss,
                         'losses': losses,
                         'best_loss': current_best_loss,
                         }, pkl_filename)
             with open('../../../Channel_nn_Outputs/'+model_name+'/MODELS/'+model_name+'_epoch'+str(epoch)+'_losses.pkl', 'wb') as fp:
                pickle.dump(losses, fp) 

   logging.info('End of train model, final losses: \n')
   logging.info('Training loss {}'.format(losses['train'][-1])+'\n')
   logging.info('Validation loss {}'.format(losses['val'][-1])+'\n')
   logging.info('Training Temp loss {}'.format(losses['train_Temp'][-1])+'\n')
   logging.info('Training Eta loss {}'.format(losses['train_Eta'][-1])+'\n')
   logging.info('Training U loss {}'.format(losses['train_U'][-1])+'\n')
   logging.info('Training V loss {}'.format(losses['train_V'][-1])+'\n')
   return(losses)

def plot_training_output(model_name, start_epoch, total_epochs, plot_freq, losses):
   plot_dir = '../../../Channel_nn_Outputs/'+model_name+'/TRAINING_PLOTS/'
   if not os.path.isdir(plot_dir):
      os.system("mkdir %s" % (plot_dir))
 
   ## Plot training loss over time
   fig = plt.figure()
   ax1 = fig.add_subplot(111)
   ax1.plot(range(0, len(losses['train'])),      losses['train'],      color='black',  label='Training Loss')
   ax1.plot(range(0, len(losses['train_Temp'])), losses['train_Temp'], color='blue',   label='Temperature Training Loss')
   ax1.plot(range(0, len(losses['train_U'])),    losses['train_U'],    color='red',    label='U Training Loss')
   ax1.plot(range(0, len(losses['train_V'])),    losses['train_V'],    color='orange', label='V Training Loss')
   ax1.plot(range(0, len(losses['train_Eta'])),  losses['train_Eta'],  color='purple', label='Eta Training Loss')
   ax1.plot(range(0, len(losses['val'])),        losses['val'],        color='grey',   label='Validation Loss')
   ax1.set_xlabel('Epochs')
   ax1.set_ylabel('RMS Error')
   ax1.set_yscale('log')
   ax1.legend()
   plt.savefig(plot_dir+'/'+model_name+'_TrainingValLossPerEpoch_epoch'+str(total_epochs)+'.png',
               bbox_inches = 'tight', pad_inches = 0.1)
   plt.close()

def LoadModel(model_name, h, optimizer, saved_epoch, tr_inf, losses, best, seed_value):
   os.environ['PYTHONHASHSEED'] = str(seed_value)
   np.random.seed(seed_value)
   torch.manual_seed(seed_value)
   torch.cuda.manual_seed(seed_value)
   torch.cuda.manual_seed_all(seed_value)
   torch.backends.cudnn.enabled = False
   torch.backends.cudnn.deterministic = True
   torch.backends.cudnn.benchmark = False
   torch.use_deterministic_algorithms(True, warn_only=True)
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


def PlotDensScatter(model_name, dimension, data_loader, h, epoch, title, norm_method, channel_dim, mean_std_file, no_out_channels,
                    no_phys_in_channels, z_dim, land, seed, no_batches, multipredmethod):

   logging.info('Making density scatter plots')
   logging.info('No_batches; '+str(no_batches))
   plt.rcParams.update({'font.size': 14})

   inputs_mean, inputs_std, inputs_range, targets_mean, targets_std, targets_range =  \
                                              ReadMeanStd(mean_std_file, dimension, no_phys_in_channels, no_out_channels, z_dim, seed)

   if isinstance(h, list):
      for i in range(len(h)):
         h[i].train(False)
         if torch.cuda.is_available():
            h[i] = h[i].cuda()
   else:
      h.train(False)
      if torch.cuda.is_available():
         h = h.cuda()

   RF_batch_no = 0
   with torch.no_grad():
      for input_batch, target_batch, extrafluxes_batch, Tmask, Umask, Vmask, out_masks, bdy_masks in data_loader:
         if RF_batch_no < 35: #memory limits, cap here to catch all of test and val, while also reducing run time
            print(RF_batch_no)

            target_batch = target_batch.cpu().detach().numpy()
            bdy_masks = bdy_masks[:,:,:,:].bool()
   
            # get prediction from the model, given the inputs
            if multipredmethod == 'random':
               pred_batch = h[random.randrange(len(h))]( torch.cat( (input_batch, Tmask, Umask, Vmask), dim=channel_dim) )
            elif multipredmethod == 'average':
               pred_batch = h[0]( torch.cat( (input_batch, Tmask, Umask, Vmask), dim=channel_dim) ) 
               for i in range(1, len(h)):
                  pred_batch = pred_batch + h[i]( torch.cat( (input_batch, Tmask, Umask, Vmask ), dim=channel_dim) )
               pred_batch = pred_batch / len(h)
            elif multipredmethod == 'singlepred':
               if land == 'ExcLand':
                  pred_batch = h( input_batch ) 
               else:
                  pred_batch = h( torch.cat( (input_batch, Tmask, Umask, Vmask), dim=channel_dim) ) 
            pred_batch = pred_batch.cpu().detach().numpy() 
   
            # Denormalise, if rolllen>1 (loss function calc over multiple steps) only take first step
            target_batch = rr.RF_DeNormalise(target_batch[:, :no_out_channels, :, :], targets_mean, targets_std, targets_range,
                                                  dimension, norm_method, seed)
            pred_batch = rr.RF_DeNormalise(pred_batch[:, :no_out_channels, :, :], targets_mean, targets_std, targets_range,
                                                     dimension, norm_method, seed)
            
            # Mask
            target_batch = np.where( out_masks[0]==1, target_batch, np.nan)
            pred_batch = np.where( out_masks[0]==1, pred_batch, np.nan)
   
            # Add summed error of this batch
            target_batch = np.where( out_masks[0]==1, target_batch, np.nan)
            pred_batch = np.where( out_masks[0]==1, pred_batch, np.nan)
            if RF_batch_no==0:
               targets      = target_batch
               predictions  = pred_batch 
            else:
               targets     = np.concatenate( ( targets, target_batch ), axis=0)
               predictions = np.concatenate( ( predictions, pred_batch ), axis=0)
   
         RF_batch_no = RF_batch_no+1

   # Reshape 
   if dimension == '2d':
      tmp_targets = np.zeros((targets.shape[0], 4, z_dim,targets.shape[2], targets.shape[3]))
      tmp_predictions = np.zeros((predictions.shape[0], 4, z_dim, predictions.shape[2], predictions.shape[3]))
      
      for i in range(3):
         tmp_targets[:,i,:,:,:] = targets[:,i*z_dim:(i+1)*z_dim,:,:]
         tmp_predictions[:,i,:,:,:] = predictions[:,i*z_dim:(i+1)*z_dim,:,:]
      tmp_targets[:,3,:,:,:] = targets[:,3*z_dim:(3*z_dim)+1,:,:]
      tmp_predictions[:,3,:,:,:] = predictions[:,3*z_dim:(3*z_dim)+1,:,:]

      targets = tmp_targets
      predictions = tmp_predictions

   # Make plots of the full dataset
   fig_main = plt.figure(figsize=(9,10))
   ax_temp  = fig_main.add_subplot(221)
   ax_U     = fig_main.add_subplot(223)
   ax_V     = fig_main.add_subplot(224)
   ax_Eta   = fig_main.add_subplot(222)

   counts, xedges, yedges, im_temp = \
              ax_temp.hist2d(targets[:,0,:,:,:].reshape(-1),
                             predictions[:,0,:,:,:].reshape(-1),
                             bins=(50, 50),
                             range=[[min( np.nanmin(targets[:,0,:,:,:]), np.nanmin(predictions[:,0,:,:,:]) ),
                                     max( np.nanmax(targets[:,0,:,:,:]), np.nanmax(predictions[:,0,:,:,:]) )],
                                    [min( np.nanmin(targets[:,0,:,:,:]), np.nanmin(predictions[:,0,:,:,:]) ),
                                     max( np.nanmax(targets[:,0,:,:,:]), np.nanmax(predictions[:,0,:,:,:]) )]],
                             cmap='Blues', norm=colors.LogNorm() )
   cb=plt.colorbar(im_temp, ax=(ax_temp), shrink=0.9, location='bottom')

   counts, xedges, yedges, im_U = \
              ax_U.hist2d(targets[:,1,:,:,:].reshape(-1),
                          predictions[:,1,:,:,:].reshape(-1),
                          bins=(50, 50),
                          range=[[min( np.nanmin(targets[:,1,:,:,:]), np.nanmin(predictions[:,1,:,:,:]) ),
                                  max( np.nanmax(targets[:,1,:,:,:]), np.nanmax(predictions[:,1,:,:,:]) )],
                                 [min( np.nanmin(targets[:,1,:,:,:]), np.nanmin(predictions[:,1,:,:,:]) ),
                                  max( np.nanmax(targets[:,1,:,:,:]), np.nanmax(predictions[:,1,:,:,:]) )]],
                          cmap='Reds', norm=colors.LogNorm() )
   cb=plt.colorbar(im_U, ax=(ax_U), shrink=0.9, location='bottom')

   counts, xedges, yedges, im_V = \
              ax_V.hist2d(targets[:,2,:,:,:].reshape(-1), 
                          predictions[:,2,:,:,:].reshape(-1),
                          bins=(50, 50),
                          range=[[min( np.nanmin(targets[:,2,:,:,:]), np.nanmin(predictions[:,2,:,:,:]) ),
                                  max( np.nanmax(targets[:,2,:,:,:]), np.nanmax(predictions[:,2,:,:,:]) )],
                                 [min( np.nanmin(targets[:,2,:,:,:]), np.nanmin(predictions[:,2,:,:,:]) ),
                                  max( np.nanmax(targets[:,2,:,:,:]), np.nanmax(predictions[:,2,:,:,:]) )]],
                          cmap='Oranges', norm=colors.LogNorm() )
   cb=plt.colorbar(im_V, ax=(ax_V), shrink=0.9, location='bottom')

   counts, xedges, yedges, im_Eta = \
              ax_Eta.hist2d(targets[:,3,0,:,:].reshape(-1), 
                            predictions[:,3,0,:,:].reshape(-1),
                            bins=(50, 50),
                            range=[[min( np.nanmin(targets[:,3,:,:,:]), np.nanmin(predictions[:,3,:,:,:]) ),
                                    max( np.nanmax(targets[:,3,:,:,:]), np.nanmax(predictions[:,3,:,:,:]) )],
                                   [min( np.nanmin(targets[:,3,:,:,:]), np.nanmin(predictions[:,3,:,:,:]) ),
                                    max( np.nanmax(targets[:,3,:,:,:]), np.nanmax(predictions[:,3,:,:,:]) )]],
                            cmap='Purples', norm=colors.LogNorm() )
   cb=plt.colorbar(im_Eta, ax=(ax_Eta), shrink=0.9, location='bottom')
   
   ax_temp.set_xlabel('Truth')
   ax_temp.set_ylabel('Predictions')
   ax_temp.set_title('Change in Temperature ('+u'\xb0'+'C)')

   ax_U.set_xlabel('Truth')
   ax_U.set_ylabel('Predictions')
   ax_U.set_title('Change in Eastward Velocity (m/s)')

   ax_V.set_xlabel('Truth')
   ax_V.set_ylabel('Predictions')
   ax_V.set_title('Change in Northward Velocity (m/s)')

   ax_Eta.set_xlabel('Truth')
   ax_Eta.set_ylabel('Predictions')
   ax_Eta.set_title('Change in Sea Surface Height (m)')

   plt.tight_layout()

   plt.savefig('../../../Channel_nn_Outputs/'+model_name+'/TRAINING_PLOTS/'+
               model_name+'_densescatter_epoch'+str(epoch).rjust(3,'0')+'_'+title+'.png',
               bbox_inches = 'tight', pad_inches = 0.1)
   plt.close()

   # Plot just Temperature
   fig_main = plt.figure(figsize=(7 ,7 ))
   ax_temp  = fig_main.add_subplot(111)

   counts, xedges, yedges, im_temp = \
              ax_temp.hist2d(targets[:,0,:,:,:].reshape(-1),
                             predictions[:,0,:,:,:].reshape(-1),
                             bins=(50, 50),
                             range=[[min( np.nanmin(targets[:,0,:,:,:]), np.nanmin(predictions[:,0,:,:,:]) ),
                                     max( np.nanmax(targets[:,0,:,:,:]), np.nanmax(predictions[:,0,:,:,:]) )],
                                    [min( np.nanmin(targets[:,0,:,:,:]), np.nanmin(predictions[:,0,:,:,:]) ),
                                     max( np.nanmax(targets[:,0,:,:,:]), np.nanmax(predictions[:,0,:,:,:]) )]],
                             cmap='Blues', norm=colors.LogNorm() )
   ax_temp.annotate( 'Pearsons Correlation Coefficient: {mystats:.2e}'.format(mystats=stats.pearsonr(
                                targets[:,0,:,:,:][~np.isnan(targets[:,0,:,:,:])], 
                                predictions[:,0,:,:,:][~np.isnan(targets[:,0,:,:,:])] )[0]),
                     xy=(.02, .95), xycoords='axes fraction' ) 
   ax_temp.set_xlabel('Truth')
   ax_temp.set_ylabel('Predictions')
   ax_temp.set_title('Change in Temperature ('+u'\xb0'+'C)')
   cb=plt.colorbar(im_temp, ax=(ax_temp), shrink=0.9, location='bottom', pad=0.1)

   plt.tight_layout()
   plt.savefig('../../../Channel_nn_Outputs/'+model_name+'/TRAINING_PLOTS/'+
               model_name+'_densescatterTemp_epoch'+str(epoch).rjust(3,'0')+'_'+title+'.png',
               bbox_inches = 'tight', pad_inches = 0.1)
   plt.close()


   # PLOT RESULTS FROM SPECIFIC AREAS
   bdy_masks = bdy_masks[0:1,:,:,:]
   bdy_masks = bdy_masks.bool().expand(targets.shape[0], bdy_masks.shape[1], bdy_masks.shape[2], bdy_masks.shape[3])

   # Make plots of the just near land points
   fig_main = plt.figure(figsize=(9,10))
   ax_temp  = fig_main.add_subplot(221)
   ax_U     = fig_main.add_subplot(223)
   ax_V     = fig_main.add_subplot(224)
   ax_Eta   = fig_main.add_subplot(222)

   counts, xedges, yedges, im_temp = \
              ax_temp.hist2d(targets[:,0,:,:,:][bdy_masks[:,:z_dim,:,:]],
                             predictions[:,0,:,:,:][bdy_masks[:,:z_dim,:,:]],
                             bins=(50, 50),
                             range=[[min( np.nanmin(targets[:,0,:,:,:][bdy_masks[:,:z_dim,:,:]]), np.nanmin(predictions[:,0,:,:,:][bdy_masks[:,:z_dim,:,:]]) ),
                                     max( np.nanmax(targets[:,0,:,:,:][bdy_masks[:,:z_dim,:,:]]), np.nanmax(predictions[:,0,:,:,:][bdy_masks[:,:z_dim,:,:]]) )],
                                    [min( np.nanmin(targets[:,0,:,:,:][bdy_masks[:,:z_dim,:,:]]), np.nanmin(predictions[:,0,:,:,:][bdy_masks[:,:z_dim,:,:]]) ),
                                     max( np.nanmax(targets[:,0,:,:,:][bdy_masks[:,:z_dim,:,:]]), np.nanmax(predictions[:,0,:,:,:][bdy_masks[:,:z_dim,:,:]]) )]],
                             cmap='Blues', norm=colors.LogNorm() )
   cb=plt.colorbar(im_temp, ax=(ax_temp), shrink=0.9, location='bottom')

   counts, xedges, yedges, im_U = \
              ax_U.hist2d(targets[:,1,:,:,:][bdy_masks[:,z_dim:2*z_dim,:,:]],
                          predictions[:,1,:,:,:][bdy_masks[:,z_dim:2*z_dim,:,:]],
                          bins=(50, 50),
                          range=[[min( np.nanmin(targets[:,1,:,:,:][bdy_masks[:,z_dim:2*z_dim,:,:]]), np.nanmin(predictions[:,1,:,:,:][bdy_masks[:,z_dim:2*z_dim,:,:]]) ),
                                  max( np.nanmax(targets[:,1,:,:,:][bdy_masks[:,z_dim:2*z_dim,:,:]]), np.nanmax(predictions[:,1,:,:,:][bdy_masks[:,z_dim:2*z_dim,:,:]]) )],
                                 [min( np.nanmin(targets[:,1,:,:,:][bdy_masks[:,z_dim:2*z_dim,:,:]]), np.nanmin(predictions[:,1,:,:,:][bdy_masks[:,z_dim:2*z_dim,:,:]]) ),
                                  max( np.nanmax(targets[:,1,:,:,:][bdy_masks[:,z_dim:2*z_dim,:,:]]), np.nanmax(predictions[:,1,:,:,:][bdy_masks[:,z_dim:2*z_dim,:,:]]) )]],
                          cmap='Reds', norm=colors.LogNorm() )
   cb=plt.colorbar(im_U, ax=(ax_U), shrink=0.9, location='bottom')

   counts, xedges, yedges, im_V = \
              ax_V.hist2d(targets[:,2,:,:,:][bdy_masks[:,2*z_dim:3*z_dim,:,:]], 
                          predictions[:,2,:,:,:][bdy_masks[:,2*z_dim:3*z_dim,:,:]],
                          bins=(50, 50),
                          range=[[min( np.nanmin(targets[:,2,:,:,:][bdy_masks[:,2*z_dim:3*z_dim,:,:]]), np.nanmin(predictions[:,2,:,:,:][bdy_masks[:,2*z_dim:3*z_dim,:,:]]) ),
                                  max( np.nanmax(targets[:,2,:,:,:][bdy_masks[:,2*z_dim:3*z_dim,:,:]]), np.nanmax(predictions[:,2,:,:,:][bdy_masks[:,2*z_dim:3*z_dim,:,:]]) )],
                                 [min( np.nanmin(targets[:,2,:,:,:][bdy_masks[:,2*z_dim:3*z_dim,:,:]]), np.nanmin(predictions[:,2,:,:,:][bdy_masks[:,2*z_dim:3*z_dim,:,:]]) ),
                                  max( np.nanmax(targets[:,2,:,:,:][bdy_masks[:,2*z_dim:3*z_dim,:,:]]), np.nanmax(predictions[:,2,:,:,:][bdy_masks[:,2*z_dim:3*z_dim,:,:]]) )]],
                          cmap='Oranges', norm=colors.LogNorm() )
   cb=plt.colorbar(im_V, ax=(ax_V), shrink=0.9, location='bottom')

   counts, xedges, yedges, im_Eta = \
           ax_Eta.hist2d(targets[:,3,0,:,:][bdy_masks[:,3*z_dim,:,:]], 
                         predictions[:,3,0,:,:][bdy_masks[:,3*z_dim,:,:]],
                         bins=(50, 50),
                         range=[[min( np.nanmin(targets[:,3,0,:,:][bdy_masks[:,3*z_dim,:,:]]), np.nanmin(predictions[:,3,0,:,:][bdy_masks[:,3*z_dim,:,:]]) ),
                                 max( np.nanmax(targets[:,3,0,:,:][bdy_masks[:,3*z_dim,:,:]]), np.nanmax(predictions[:,3,0,:,:][bdy_masks[:,3*z_dim,:,:]]) )],
                                [min( np.nanmin(targets[:,3,0,:,:][bdy_masks[:,3*z_dim,:,:]]), np.nanmin(predictions[:,3,0,:,:][bdy_masks[:,3*z_dim,:,:]]) ),
                                 max( np.nanmax(targets[:,3,0,:,:][bdy_masks[:,3*z_dim,:,:]]), np.nanmax(predictions[:,3,0,:,:][bdy_masks[:,3*z_dim,:,:]]) )]],
                            cmap='Purples', norm=colors.LogNorm() )
   cb=plt.colorbar(im_Eta, ax=(ax_Eta), shrink=0.9, location='bottom')
   
   ax_temp.set_xlabel('Truth')
   ax_temp.set_ylabel('Predictions')
   ax_temp.set_title('Change in Temperature ('+u'\xb0'+'C)')

   ax_U.set_xlabel('Truth')
   ax_U.set_ylabel('Predictions')
   ax_U.set_title('Change in Eastward Velocity (m/s)')

   ax_V.set_xlabel('Truth')
   ax_V.set_ylabel('Predictions')
   ax_V.set_title('Change in Northward Velocity (m/s)')

   ax_Eta.set_xlabel('Truth')
   ax_Eta.set_ylabel('Predictions')
   ax_Eta.set_title('Change in Sea Surface Height (m)')

   plt.tight_layout()

   plt.savefig('../../../Channel_nn_Outputs/'+model_name+'/TRAINING_PLOTS/'+
               model_name+'_densescatter_bdy_epoch'+str(epoch).rjust(3,'0')+'_'+title+'.png',
               bbox_inches = 'tight', pad_inches = 0.1)
   plt.close()

   # Make plots of the non-bdy_points dataset
   fig_main = plt.figure(figsize=(9,10))
   ax_temp  = fig_main.add_subplot(221)
   ax_U     = fig_main.add_subplot(223)
   ax_V     = fig_main.add_subplot(224)
   ax_Eta   = fig_main.add_subplot(222)

   counts, xedges, yedges, im_temp = \
              ax_temp.hist2d(targets[:,0,:,:,:][~bdy_masks[:,:z_dim,:,:]],
                             predictions[:,0,:,:,:][~bdy_masks[:,:z_dim,:,:]],
                             bins=(50, 50),
                             range=[[min( np.nanmin(targets[:,0,:,:,:][~bdy_masks[:,:z_dim,:,:]]), np.nanmin(predictions[:,0,:,:,:][~bdy_masks[:,:z_dim,:,:]]) ),
                                     max( np.nanmax(targets[:,0,:,:,:][~bdy_masks[:,:z_dim,:,:]]), np.nanmax(predictions[:,0,:,:,:][~bdy_masks[:,:z_dim,:,:]]) )],
                                    [min( np.nanmin(targets[:,0,:,:,:][~bdy_masks[:,:z_dim,:,:]]), np.nanmin(predictions[:,0,:,:,:][~bdy_masks[:,:z_dim,:,:]]) ),
                                     max( np.nanmax(targets[:,0,:,:,:][~bdy_masks[:,:z_dim,:,:]]), np.nanmax(predictions[:,0,:,:,:][~bdy_masks[:,:z_dim,:,:]]) )]],
                             cmap='Blues', norm=colors.LogNorm() )
   cb=plt.colorbar(im_temp, ax=(ax_temp), shrink=0.9, location='bottom')

   counts, xedges, yedges, im_U = \
              ax_U.hist2d(targets[:,1,:,:,:][~bdy_masks[:,z_dim:z_dim*2,:,:]],
                          predictions[:,1,:,:,:][~bdy_masks[:,z_dim:z_dim*2,:,:]],
                          bins=(50, 50),
                          range=[[min( np.nanmin(targets[:,1,:,:,:][~bdy_masks[:,z_dim:z_dim*2,:,:]]), np.nanmin(predictions[:,1,:,:,:][~bdy_masks[:,z_dim:z_dim*2,:,:]]) ),
                                  max( np.nanmax(targets[:,1,:,:,:][~bdy_masks[:,z_dim:z_dim*2,:,:]]), np.nanmax(predictions[:,1,:,:,:][~bdy_masks[:,z_dim:z_dim*2,:,:]]) )],
                                 [min( np.nanmin(targets[:,1,:,:,:][~bdy_masks[:,z_dim:z_dim*2,:,:]]), np.nanmin(predictions[:,1,:,:,:][~bdy_masks[:,z_dim:z_dim*2,:,:]]) ),
                                  max( np.nanmax(targets[:,1,:,:,:][~bdy_masks[:,z_dim:z_dim*2,:,:]]), np.nanmax(predictions[:,1,:,:,:][~bdy_masks[:,z_dim:z_dim*2,:,:]]) )]],
                          cmap='Reds', norm=colors.LogNorm() )
   cb=plt.colorbar(im_U, ax=(ax_U), shrink=0.9, location='bottom')

   counts, xedges, yedges, im_V = \
              ax_V.hist2d(targets[:,2,:,:,:][~bdy_masks[:,2*z_dim:3*z_dim,:,:]], 
                          predictions[:,2,:,:,:][~bdy_masks[:,2*z_dim:3*z_dim,:,:]],
                          bins=(50, 50),
                          range=[[min( np.nanmin(targets[:,2,:,:,:][~bdy_masks[:,2*z_dim:3*z_dim,:,:]]), np.nanmin(predictions[:,2,:,:,:][~bdy_masks[:,2*z_dim:3*z_dim,:,:]]) ),
                                  max( np.nanmax(targets[:,2,:,:,:][~bdy_masks[:,2*z_dim:3*z_dim,:,:]]), np.nanmax(predictions[:,2,:,:,:][~bdy_masks[:,2*z_dim:3*z_dim,:,:]]) )],
                                 [min( np.nanmin(targets[:,2,:,:,:][~bdy_masks[:,2*z_dim:3*z_dim,:,:]]), np.nanmin(predictions[:,2,:,:,:][~bdy_masks[:,2*z_dim:3*z_dim,:,:]]) ),
                                  max( np.nanmax(targets[:,2,:,:,:][~bdy_masks[:,2*z_dim:3*z_dim,:,:]]), np.nanmax(predictions[:,2,:,:,:][~bdy_masks[:,2*z_dim:3*z_dim,:,:]]) )]],
                          cmap='Oranges', norm=colors.LogNorm() )
   cb=plt.colorbar(im_V, ax=(ax_V), shrink=0.9, location='bottom')

   counts, xedges, yedges, im_Eta = \
         ax_Eta.hist2d(targets[:,3,0,:,:][~bdy_masks[:,3*z_dim,:,:]], 
                       predictions[:,3,0,:,:][~bdy_masks[:,3*z_dim,:,:]],
                       bins=(50, 50),
                       range=[[min( np.nanmin(targets[:,3,0,:,:][~bdy_masks[:,3*z_dim,:,:]]), np.nanmin(predictions[:,3,0,:,:][~bdy_masks[:,3*z_dim,:,:]]) ),
                               max( np.nanmax(targets[:,3,0,:,:][~bdy_masks[:,3*z_dim,:,:]]), np.nanmax(predictions[:,3,0,:,:][~bdy_masks[:,3*z_dim,:,:]]) )],
                              [min( np.nanmin(targets[:,3,0,:,:][~bdy_masks[:,3*z_dim,:,:]]), np.nanmin(predictions[:,3,0,:,:][~bdy_masks[:,3*z_dim,:,:]]) ),
                               max( np.nanmax(targets[:,3,0,:,:][~bdy_masks[:,3*z_dim,:,:]]), np.nanmax(predictions[:,3,0,:,:][~bdy_masks[:,3*z_dim,:,:]]) )]],
                       cmap='Purples', norm=colors.LogNorm() )
   cb=plt.colorbar(im_Eta, ax=(ax_Eta), shrink=0.9, location='bottom')
   
   ax_temp.set_xlabel('Truth')
   ax_temp.set_ylabel('Predictions')
   ax_temp.set_title('Change in Temperature ('+u'\xb0'+'C)')

   ax_U.set_xlabel('Truth')
   ax_U.set_ylabel('Predictions')
   ax_U.set_title('Change in Eastward Velocity (m/s)')

   ax_V.set_xlabel('Truth')
   ax_V.set_ylabel('Predictions')
   ax_V.set_title('Change in Northward Velocity (m/s)')

   ax_Eta.set_xlabel('Truth')
   ax_Eta.set_ylabel('Predictions')
   ax_Eta.set_title('Change in Sea Surface Height (m)')

   plt.tight_layout()

   plt.savefig('../../../Channel_nn_Outputs/'+model_name+'/TRAINING_PLOTS/'+
               model_name+'_densescatter_nonbdy_epoch'+str(epoch).rjust(3,'0')+'_'+title+'.png',
               bbox_inches = 'tight', pad_inches = 0.1)
   plt.close()

def OutputStats(model_name, model_style, MITgcm_filename, data_loader, h, no_epochs, y_dim_used, dimension, 
                histlen, land, file_append, norm_method, channel_dim, mean_std_file, no_phys_in_channels,
                no_out_channels, MITgcm_stats_filename, seed, multipredmethod):
   #-------------------------------------------------------------
   # Output the rms and correlation coefficients over a dataset
   #-------------------------------------------------------------
   #  1. Make predictions over entire dataset (training, val or test), summing Mean Squared Error 
   #     as we go
   #  2. Store predictions and targets for a small number of smaples in an array
   #  3. Calculate the RMS of the dataset from the summed squared error
   #  4. Store the RMS and subset of samples in a netcdf file

   logging.info('Outputting stats')

   # Read in grid data from MITgcm file
   MITgcm_ds = xr.open_dataset(MITgcm_filename)
   da_X = MITgcm_ds['X']
   da_Y = MITgcm_ds['Y']
   da_Z = MITgcm_ds['Zmd000038'] 

   MITgcm_stats_ds = xr.open_dataset(MITgcm_stats_filename)

   inputs_mean, inputs_std, inputs_range, targets_mean, targets_std, targets_range = \
                               ReadMeanStd(mean_std_file, dimension, no_phys_in_channels, no_out_channels, da_Z.shape[0], seed)

   if land == 'ExcLand':
      da_Y = da_Y[3:101]
  
   no_samples = 1100   # We calculate stats over the entire dataset, but only output this many samples - covers all of test and val sets.
   z_dim = da_Z.shape[0]

   count_samples = count_3d = count_2d = 0
   bdy_count_3d = bdy_count_2d = nonbdy_count_3d = nonbdy_count_2d = 0
   extbdy_count_3d = extbdy_count_2d = nonextbdy_count_3d = nonextbdy_count_2d = 0
   spatial_sumsq_er = pers_spatial_sumsq_er = 0
   sumsq_er = pers_sumsq_er = bdy_rms_er = perse_bdy_rms_er = nonbdy_rms_er = perse_nonbdy_rms_er = 0
   temp_sumsq_er = u_sumsq_er = v_sumsq_er = eta_sumsq_er = 0
   pers_temp_sumsq_er = pers_u_sumsq_er = pers_v_sumsq_er = pers_eta_sumsq_er = 0
   bdy_temp_sumsq_er = bdy_u_sumsq_er = bdy_v_sumsq_er = bdy_eta_sumsq_er = 0
   pers_bdy_temp_sumsq_er = pers_bdy_u_sumsq_er = pers_bdy_v_sumsq_er = pers_bdy_eta_sumsq_er = 0
   nonbdy_temp_sumsq_er = nonbdy_u_sumsq_er = nonbdy_v_sumsq_er = nonbdy_eta_sumsq_er = 0
   pers_nonbdy_temp_sumsq_er = pers_nonbdy_u_sumsq_er = pers_nonbdy_v_sumsq_er = pers_nonbdy_eta_sumsq_er = 0
   extbdy_temp_sumsq_er = extbdy_u_sumsq_er = extbdy_v_sumsq_er = extbdy_eta_sumsq_er = 0
   pers_extbdy_temp_sumsq_er = pers_extbdy_u_sumsq_er = pers_extbdy_v_sumsq_er = pers_extbdy_eta_sumsq_er = 0
   nonextbdy_temp_sumsq_er = nonextbdy_u_sumsq_er = nonextbdy_v_sumsq_er = nonextbdy_eta_sumsq_er = 0
   pers_nonextbdy_temp_sumsq_er = pers_nonextbdy_u_sumsq_er = pers_nonextbdy_v_sumsq_er = pers_nonextbdy_eta_sumsq_er = 0
   mean_targets_tend = 0
   mean_temp_targets_tend = mean_u_targets_tend = mean_v_targets_tend = mean_eta_targets_tend = 0
   mean_bdy_temp_targets_tend = mean_bdy_u_targets_tend = mean_bdy_v_targets_tend = mean_bdy_eta_targets_tend = 0
   mean_nonbdy_temp_targets_tend = mean_nonbdy_u_targets_tend = mean_nonbdy_v_targets_tend = mean_nonbdy_eta_targets_tend = 0
   mean_extbdy_temp_targets_tend = mean_extbdy_u_targets_tend = mean_extbdy_v_targets_tend = mean_extbdy_eta_targets_tend = 0
   mean_nonextbdy_temp_targets_tend = mean_nonextbdy_u_targets_tend = mean_nonextbdy_v_targets_tend = mean_nonextbdy_eta_targets_tend = 0

   if isinstance(h, list):
      for i in range(len(h)):
         h[i].train(False)
         if torch.cuda.is_available():
            h[i] = h[i].cuda()
   else:
      h.train(False)
      if torch.cuda.is_available():
         h = h.cuda()

   RF_batch_no = 0
   with torch.no_grad():
      for input_batch, target_tend_batch, extrafluxes_batch, Tmask, Umask, Vmask, out_masks, bdy_masks in data_loader:
          target_tend_batch = target_tend_batch.cpu().detach().numpy()

          bdy_masks = bdy_masks[:,:,:,:].bool().numpy()
          # get prediction from the model, given the inputs
          if multipredmethod == 'random':
             pred_tend_batch = h[random.randrange(len(h))]( torch.cat( (input_batch, Tmask, Umask, Vmask), dim=channel_dim) )
          elif multipredmethod == 'average':
             pred_tend_batch = h[0]( torch.cat( (input_batch, Tmask, Umask, Vmask), dim=channel_dim) ) 
             for i in range(1, len(h)):
                pred_tend_batch = pred_tend_batch + h[i]( torch.cat( (input_batch, Tmask, Umask, Vmask ), dim=channel_dim) )
             pred_tend_batch = pred_tend_batch / len(h)
          elif multipredmethod == 'singlepred':
             if land == 'ExcLand':
                pred_tend_batch = h( input_batch ) 
             else:
                pred_tend_batch = h( torch.cat( (input_batch, Tmask, Umask, Vmask), dim=channel_dim) ) 
          pred_tend_batch = pred_tend_batch.cpu().detach().numpy() 

          # Denormalise, if rolllen>1 (loss function calc over multiple steps) only take first step
          target_tend_batch = rr.RF_DeNormalise(target_tend_batch[:, :no_out_channels, :, :], targets_mean, targets_std, targets_range,
                                                dimension, norm_method, seed)
          pred_tend_batch = rr.RF_DeNormalise(pred_tend_batch[:, :no_out_channels, :, :], targets_mean, targets_std, targets_range,
                                                   dimension, norm_method, seed)

          # Mask
          target_tend_batch = np.where( out_masks[0]==1, target_tend_batch, np.nan)
          pred_tend_batch = np.where( out_masks[0]==1, pred_tend_batch, np.nan)

          # Convert from increments to fields for outputting
          input_batch = input_batch#.cpu().detach().numpy()

          if model_style == 'ConvLSTM' or model_style == 'UNetConvLSTM':
             input_batch[:,-1,:3*z_dim+1,:,:] = rr.RF_DeNormalise( input_batch[:,-1,:3*z_dim+1,:,:],
                                                                   inputs_mean[:3*z_dim+1], inputs_std[:3*z_dim+1], inputs_range[:3*z_dim+1],
                                                                   dimension, norm_method, seed )
             target_fld_batch = input_batch[:,-1,:3*z_dim+1,:,:] + target_tend_batch
             predicted_fld_batch = input_batch[:,-1,:3*z_dim+1,:,:] + pred_tend_batch
          elif dimension == '2d':
             input_batch[:,(histlen-1)*no_phys_in_channels:(histlen-1)*no_phys_in_channels+3*z_dim+1,:,:] = \
                                    rr.RF_DeNormalise( input_batch[:,(histlen-1)*no_phys_in_channels:(histlen-1)*no_phys_in_channels+3*z_dim+1,:,:],
                                                       inputs_mean[:3*z_dim+1], inputs_std[:3*z_dim+1], inputs_range[:3*z_dim+1],
                                                       dimension, norm_method, seed )
             target_fld_batch = ( input_batch[:,(histlen-1)*no_phys_in_channels:(histlen-1)*no_phys_in_channels+3*z_dim+1,:,:] 
                                  + target_tend_batch )
             predicted_fld_batch = ( input_batch[:,(histlen-1)*no_phys_in_channels:(histlen-1)*no_phys_in_channels+3*z_dim+1,:,:]
                                     + pred_tend_batch )
          elif dimension == '3d':
             input_batch[:,(histlen-1)*no_phys_in_channels:(histlen-1)*no_phys_in_channels+3*z_dim+1,:,:,:] = \
                                      rr.RF_DeNormalise( input_batch[:,(histlen-1)*no_phys_in_channels:(histlen-1)*no_phys_in_channels+3*z_dim+1,:,:,:],
                                                         inputs_mean[:3*z_dim+1], inputs_std[:3*z_dim+1], inputs_range[:3*z_dim+1],
                                                         dimension, norm_method, seed )
             target_fld_batch = ( input_batch[:,(histlen-1)*no_phys_in_channels:(histlen-1)*no_phys_in_channels+3*z_dim+1,:,:,:]
                                  + target_tend_batch )
             predicted_fld_batch = ( input_batch[:,(histlen-1)*no_phys_in_channels:(histlen-1)*no_phys_in_channels+3*z_dim+1,:,:,:]
                                     + pred_tend_batch )

          # Add summed error of this batch
          target_tend_batch = np.where( out_masks[0]==1, target_tend_batch, np.nan)
          pred_tend_batch = np.where( out_masks[0]==1, pred_tend_batch, np.nan)
          if RF_batch_no==0:
             nc_targets_tend      = target_tend_batch
             nc_predictions_tend  = pred_tend_batch 
             nc_targets_fld       = target_fld_batch
             nc_predictions_fld   = predicted_fld_batch 
          else:
             if count_samples < no_samples:
                nc_targets_tend     = np.concatenate( ( nc_targets_tend, target_tend_batch ), axis=0)
                nc_predictions_tend = np.concatenate( ( nc_predictions_tend, pred_tend_batch ), axis=0)
                nc_targets_fld      = np.concatenate( ( nc_targets_fld, target_fld_batch ), axis=0)
                nc_predictions_fld  = np.concatenate( ( nc_predictions_fld, predicted_fld_batch ), axis=0)

          spatial_sumsq_er = spatial_sumsq_er + np.nansum( np.square(pred_tend_batch-target_tend_batch), axis=0 )
          pers_spatial_sumsq_er = pers_spatial_sumsq_er + np.nansum( np.square(target_tend_batch), axis=0 )

          sumsq_er = sumsq_er + np.nansum( np.square(pred_tend_batch[:,:3*z_dim+1,:,:]-target_tend_batch[:,:3*z_dim+1,:,:]) )
          pers_sumsq_er = pers_sumsq_er + np.nansum( np.square(target_tend_batch[:,:3*z_dim+1,:,:]) )

          temp_sumsq_er = temp_sumsq_er + np.nansum( np.square(pred_tend_batch[:,:z_dim,:,:] - 
                                                                             target_tend_batch[:,:z_dim,:,:]) ) 
          u_sumsq_er    = u_sumsq_er + np.nansum( np.square(pred_tend_batch[:,z_dim:2*z_dim,:,:] - 
                                                                          target_tend_batch[:,z_dim:2*z_dim,:,:]) ) 
          v_sumsq_er    = v_sumsq_er + np.nansum( np.square(pred_tend_batch[:,2*z_dim:3*z_dim,:,:] - 
                                                                          target_tend_batch[:,2*z_dim:3*z_dim,:,:]) ) 
          eta_sumsq_er  = eta_sumsq_er + np.nansum( np.square(pred_tend_batch[:,3*z_dim,:,:] - 
                                                                            target_tend_batch[:,3*z_dim,:,:]) ) 
          pers_temp_sumsq_er = pers_temp_sumsq_er + np.nansum( np.square(target_tend_batch[:,:z_dim,:,:]) )
          pers_u_sumsq_er    = pers_u_sumsq_er + np.nansum( np.square(target_tend_batch[:,z_dim:2*z_dim,:,:]) ) 
          pers_v_sumsq_er    = pers_v_sumsq_er + np.nansum( np.square(target_tend_batch[:,2*z_dim:3*z_dim,:,:]) ) 
          pers_eta_sumsq_er  = pers_eta_sumsq_er + np.nansum( np.square(target_tend_batch[:,3*z_dim,:,:]) ) 

          bdy_temp_sumsq_er = bdy_temp_sumsq_er + np.nansum( np.square(pred_tend_batch[:,:z_dim,:,:] - 
                                                                       target_tend_batch[:,:z_dim,:,:])[bdy_masks[:,:z_dim,:,:]] ) 
          bdy_u_sumsq_er    = bdy_u_sumsq_er + np.nansum( np.square(pred_tend_batch[:,z_dim:2*z_dim,:,:] - 
                                                                       target_tend_batch[:,z_dim:2*z_dim,:,:])[bdy_masks[:,z_dim:2*z_dim,:,:]] ) 
          bdy_v_sumsq_er    = bdy_v_sumsq_er + np.nansum( np.square(pred_tend_batch[:,2*z_dim:3*z_dim,:,:] - 
                                                                       target_tend_batch[:,2*z_dim:3*z_dim,:,:])[bdy_masks[:,2*z_dim:3*z_dim,:,:]] ) 
          bdy_eta_sumsq_er  = bdy_eta_sumsq_er + np.nansum( np.square(pred_tend_batch[:,3*z_dim,:,:] - 
                                                                       target_tend_batch[:,3*z_dim,:,:])[bdy_masks[:,3*z_dim,:,:]] ) 
          bdy_sumsq_er      = bdy_temp_sumsq_er + bdy_u_sumsq_er + bdy_v_sumsq_er + bdy_eta_sumsq_er
          pers_bdy_temp_sumsq_er = pers_bdy_temp_sumsq_er + np.nansum( np.square(
                                                                       target_tend_batch[:,:z_dim,:,:])[bdy_masks[:,:z_dim,:,:]] ) 
          pers_bdy_u_sumsq_er    = pers_bdy_u_sumsq_er + np.nansum( np.square(
                                                                       target_tend_batch[:,z_dim:2*z_dim,:,:])[bdy_masks[:,z_dim:2*z_dim,:,:]] ) 
          pers_bdy_v_sumsq_er    = pers_bdy_v_sumsq_er + np.nansum( np.square(
                                                                       target_tend_batch[:,2*z_dim:3*z_dim,:,:])[bdy_masks[:,2*z_dim:3*z_dim,:,:]] ) 
          pers_bdy_eta_sumsq_er  = pers_bdy_eta_sumsq_er + np.nansum( np.square(
                                                                       target_tend_batch[:,3*z_dim,:,:])[bdy_masks[:,3*z_dim,:,:]] ) 
          pers_bdy_sumsq_er      = pers_bdy_temp_sumsq_er + pers_bdy_u_sumsq_er + pers_bdy_v_sumsq_er + pers_bdy_eta_sumsq_er


          nonbdy_temp_sumsq_er = nonbdy_temp_sumsq_er + np.nansum( np.square(pred_tend_batch[:,:z_dim,:,:] - 
                                                                       target_tend_batch[:,:z_dim,:,:])[~bdy_masks[:,:z_dim,:,:]] ) 
          nonbdy_u_sumsq_er    = nonbdy_u_sumsq_er + np.nansum( np.square(pred_tend_batch[:,z_dim:2*z_dim,:,:] - 
                                                                       target_tend_batch[:,z_dim:2*z_dim,:,:])[~bdy_masks[:,z_dim:2*z_dim,:,:]] ) 
          nonbdy_v_sumsq_er    = nonbdy_v_sumsq_er + np.nansum( np.square(pred_tend_batch[:,2*z_dim:3*z_dim,:,:] - 
                                                                       target_tend_batch[:,2*z_dim:3*z_dim,:,:])[~bdy_masks[:,2*z_dim:3*z_dim,:,:]] )
          nonbdy_eta_sumsq_er  = nonbdy_eta_sumsq_er + np.nansum( np.square(pred_tend_batch[:,3*z_dim,:,:] -
                                                                       target_tend_batch[:,3*z_dim,:,:])[~bdy_masks[:,3*z_dim,:,:]] )
          nonbdy_sumsq_er      = nonbdy_temp_sumsq_er + nonbdy_u_sumsq_er + nonbdy_v_sumsq_er + nonbdy_eta_sumsq_er
          pers_nonbdy_temp_sumsq_er = pers_nonbdy_temp_sumsq_er + np.nansum( np.square(
                                                                       target_tend_batch[:,:z_dim,:,:])[~bdy_masks[:,:z_dim,:,:]] ) 
          pers_nonbdy_u_sumsq_er    = pers_nonbdy_u_sumsq_er + np.nansum( np.square(
                                                                       target_tend_batch[:,z_dim:2*z_dim,:,:])[~bdy_masks[:,z_dim:2*z_dim,:,:]] ) 
          pers_nonbdy_v_sumsq_er    = pers_nonbdy_v_sumsq_er + np.nansum( np.square(
                                                                       target_tend_batch[:,2*z_dim:3*z_dim,:,:])[~bdy_masks[:,2*z_dim:3*z_dim,:,:]] )
          pers_nonbdy_eta_sumsq_er  = pers_nonbdy_eta_sumsq_er + np.nansum( np.square(
                                                                       target_tend_batch[:,3*z_dim,:,:])[~bdy_masks[:,3*z_dim,:,:]] )
          pers_nonbdy_sumsq_er      = pers_nonbdy_temp_sumsq_er + pers_nonbdy_u_sumsq_er + pers_nonbdy_v_sumsq_er + pers_nonbdy_eta_sumsq_er


          mean_targets_tend             = mean_targets_tend             + np.nansum(target_tend_batch, axis=0)
          mean_temp_targets_tend        = mean_temp_targets_tend        + np.nansum(target_tend_batch[:,:z_dim,:,:])
          mean_u_targets_tend           = mean_u_targets_tend           + np.nansum(target_tend_batch[:,z_dim:2*z_dim,:,:])
          mean_v_targets_tend           = mean_v_targets_tend           + np.nansum(target_tend_batch[:,2*z_dim:3*z_dim,:,:])
          mean_eta_targets_tend         = mean_eta_targets_tend         + np.nansum(target_tend_batch[:,3*z_dim,:,:])
          mean_bdy_temp_targets_tend    = mean_bdy_temp_targets_tend    + np.nansum(target_tend_batch[:,:z_dim,:,:][bdy_masks[:,:z_dim,:,:]])
          mean_bdy_u_targets_tend       = mean_bdy_u_targets_tend       + np.nansum(target_tend_batch[:,z_dim:2*z_dim,:,:][bdy_masks[:,z_dim:2*z_dim,:,:]])
          mean_bdy_v_targets_tend       = mean_bdy_v_targets_tend       + np.nansum(target_tend_batch[:,2*z_dim:3*z_dim,:,:][bdy_masks[:,2*z_dim:3*z_dim,:,:]])
          mean_bdy_eta_targets_tend     = mean_bdy_eta_targets_tend     + np.nansum(target_tend_batch[:,3*z_dim,:,:][bdy_masks[:,3*z_dim,:,:]])
          mean_nonbdy_temp_targets_tend = mean_nonbdy_temp_targets_tend + np.nansum(target_tend_batch[:,:z_dim,:,:][~bdy_masks[:,:z_dim,:,:]])
          mean_nonbdy_u_targets_tend    = mean_nonbdy_u_targets_tend    + np.nansum(target_tend_batch[:,z_dim:2*z_dim,:,:][~bdy_masks[:,z_dim:2*z_dim,:,:]])
          mean_nonbdy_v_targets_tend    = mean_nonbdy_v_targets_tend    + np.nansum(target_tend_batch[:,2*z_dim:3*z_dim,:,:][~bdy_masks[:,2*z_dim:3*z_dim,:,:]])
          mean_nonbdy_eta_targets_tend  = mean_nonbdy_eta_targets_tend  + np.nansum(target_tend_batch[:,3*z_dim,:,:][~bdy_masks[:,3*z_dim,:,:]])

          count_3d = count_3d + bdy_masks.shape[0]*bdy_masks.shape[1]*bdy_masks.shape[2]*bdy_masks.shape[3]
          count_2d = count_2d + bdy_masks.shape[0]*bdy_masks.shape[2]*bdy_masks.shape[3]
          bdy_count_3d = bdy_count_3d + np.sum(bdy_masks[:,:,:,:])
          bdy_count_2d = bdy_count_2d + np.sum(bdy_masks[:,0,:,:])
          nonbdy_count_3d = nonbdy_count_3d + np.sum(~bdy_masks[:,:,:,:])
          nonbdy_count_2d = nonbdy_count_2d + np.sum(~bdy_masks[:,0,:,:])

          RF_batch_no = RF_batch_no+1
          count_samples = count_samples + input_batch.shape[0]

   no_samples = min(no_samples, count_samples) # Reduce no_samples if this is bigger than total samples!
  
   spatial_rms_error = np.sqrt( np.divide(spatial_sumsq_er, count_samples) )
   pers_spatial_rms_error = np.sqrt( np.divide(pers_spatial_sumsq_er, count_samples) )

   rms_error = np.sqrt( np.divide(sumsq_er, count_3d*3+count_2d) )
   pers_rms_error = np.sqrt( np.divide(pers_sumsq_er, count_3d*3+count_2d) )
   temp_rms_error = np.sqrt( np.divide(temp_sumsq_er, count_3d) )
   u_rms_error    = np.sqrt( np.divide(u_sumsq_er, count_3d) )
   v_rms_error    = np.sqrt( np.divide(v_sumsq_er, count_3d) )
   eta_rms_error  = np.sqrt( np.divide(eta_sumsq_er, count_2d) )
   pers_temp_rms_error = np.sqrt( np.divide(pers_temp_sumsq_er, count_3d) )
   pers_u_rms_error    = np.sqrt( np.divide(pers_u_sumsq_er, count_3d) )
   pers_v_rms_error    = np.sqrt( np.divide(pers_v_sumsq_er, count_3d) )
   pers_eta_rms_error  = np.sqrt( np.divide(pers_eta_sumsq_er, count_2d) )

   bdy_rms_error = np.sqrt( np.divide(bdy_sumsq_er, count_3d*3+count_2d) )
   pers_bdy_rms_error = np.sqrt( np.divide(pers_bdy_sumsq_er, count_3d*3+count_2d) )
   bdy_temp_rms_error = np.sqrt( np.divide(bdy_temp_sumsq_er, bdy_count_3d) )
   bdy_u_rms_error    = np.sqrt( np.divide(bdy_u_sumsq_er, bdy_count_3d) )
   bdy_v_rms_error    = np.sqrt( np.divide(bdy_v_sumsq_er, bdy_count_3d) )
   bdy_eta_rms_error  = np.sqrt( np.divide(bdy_eta_sumsq_er, bdy_count_2d) )
   pers_bdy_temp_rms_error = np.sqrt( np.divide(pers_bdy_temp_sumsq_er, bdy_count_3d) )
   pers_bdy_u_rms_error    = np.sqrt( np.divide(pers_bdy_u_sumsq_er, bdy_count_3d) )
   pers_bdy_v_rms_error    = np.sqrt( np.divide(pers_bdy_v_sumsq_er, bdy_count_3d) )
   pers_bdy_eta_rms_error  = np.sqrt( np.divide(pers_bdy_eta_sumsq_er, bdy_count_2d) )

   nonbdy_rms_error = np.sqrt( np.divide(nonbdy_sumsq_er, count_3d*3+count_2d) )
   pers_nonbdy_rms_error = np.sqrt( np.divide(pers_nonbdy_sumsq_er, count_3d*3+count_2d) )
   nonbdy_temp_rms_error = np.sqrt( np.divide(nonbdy_temp_sumsq_er, (count_3d-bdy_count_3d)) )
   nonbdy_u_rms_error    = np.sqrt( np.divide(nonbdy_u_sumsq_er, (count_3d-bdy_count_3d)) )
   nonbdy_v_rms_error    = np.sqrt( np.divide(nonbdy_v_sumsq_er, (count_3d-bdy_count_3d)) )
   nonbdy_eta_rms_error  = np.sqrt( np.divide(nonbdy_eta_sumsq_er, (count_2d-bdy_count_2d)) )
   pers_nonbdy_temp_rms_error = np.sqrt( np.divide(pers_nonbdy_temp_sumsq_er, (count_3d-bdy_count_3d)) )
   pers_nonbdy_u_rms_error    = np.sqrt( np.divide(pers_nonbdy_u_sumsq_er, (count_3d-bdy_count_3d)) )
   pers_nonbdy_v_rms_error    = np.sqrt( np.divide(pers_nonbdy_v_sumsq_er, (count_3d-bdy_count_3d)) )
   pers_nonbdy_eta_rms_error  = np.sqrt( np.divide(pers_nonbdy_eta_sumsq_er, (count_2d-bdy_count_2d)) )

   extbdy_temp_rms_error = np.sqrt( np.divide(extbdy_temp_sumsq_er, extbdy_count_3d) )
   extbdy_u_rms_error    = np.sqrt( np.divide(extbdy_u_sumsq_er, extbdy_count_3d) )
   extbdy_v_rms_error    = np.sqrt( np.divide(extbdy_v_sumsq_er, extbdy_count_3d) )
   extbdy_eta_rms_error  = np.sqrt( np.divide(extbdy_eta_sumsq_er, extbdy_count_2d) )
   pers_extbdy_temp_rms_error = np.sqrt( np.divide(pers_extbdy_temp_sumsq_er, extbdy_count_3d) )
   pers_extbdy_u_rms_error    = np.sqrt( np.divide(pers_extbdy_u_sumsq_er, extbdy_count_3d) )
   pers_extbdy_v_rms_error    = np.sqrt( np.divide(pers_extbdy_v_sumsq_er, extbdy_count_3d) )
   pers_extbdy_eta_rms_error  = np.sqrt( np.divide(pers_extbdy_eta_sumsq_er, extbdy_count_2d) )

   nonextbdy_temp_rms_error = np.sqrt( np.divide(nonextbdy_temp_sumsq_er, (count_3d-extbdy_count_3d)) )
   nonextbdy_u_rms_error    = np.sqrt( np.divide(nonextbdy_u_sumsq_er, (count_3d-extbdy_count_3d)) )
   nonextbdy_v_rms_error    = np.sqrt( np.divide(nonextbdy_v_sumsq_er, (count_3d-extbdy_count_3d)) )
   nonextbdy_eta_rms_error  = np.sqrt( np.divide(nonextbdy_eta_sumsq_er, (count_2d-extbdy_count_2d)) )
   pers_nonextbdy_temp_rms_error = np.sqrt( np.divide(pers_nonextbdy_temp_sumsq_er, (count_3d-extbdy_count_3d)) )
   pers_nonextbdy_u_rms_error    = np.sqrt( np.divide(pers_nonextbdy_u_sumsq_er, (count_3d-extbdy_count_3d)) )
   pers_nonextbdy_v_rms_error    = np.sqrt( np.divide(pers_nonextbdy_v_sumsq_er, (count_3d-extbdy_count_3d)) )
   pers_nonextbdy_eta_rms_error  = np.sqrt( np.divide(pers_nonextbdy_eta_sumsq_er, (count_2d-extbdy_count_2d)) )

   # mask rms error with nans (cuurently masked with 0)
   spatial_rms_error = np.where( out_masks[0]==1, spatial_rms_error, np.nan)
   #mean_targets_tend = np.where( out_masks[0]==1, mean_targets_tend, np.nan)

   mean_targets_tend = np.divide(mean_targets_tend, count_samples)
   mean_temp_targets_tend = mean_temp_targets_tend/count_3d
   mean_u_targets_tend = mean_u_targets_tend/count_3d
   mean_v_targets_tend = mean_v_targets_tend/count_3d
   mean_eta_targets_tend = mean_eta_targets_tend/count_2d
   mean_bdy_temp_targets_tend = mean_bdy_temp_targets_tend/count_3d
   mean_bdy_u_targets_tend = mean_bdy_u_targets_tend/count_3d
   mean_bdy_v_targets_tend = mean_bdy_v_targets_tend/count_3d
   mean_bdy_eta_targets_tend = mean_bdy_eta_targets_tend/count_2d
   mean_nonbdy_temp_targets_tend = mean_nonbdy_temp_targets_tend/count_3d
   mean_nonbdy_u_targets_tend = mean_nonbdy_u_targets_tend/count_3d
   mean_nonbdy_v_targets_tend = mean_nonbdy_v_targets_tend/count_3d
   mean_nonbdy_eta_targets_tend = mean_nonbdy_eta_targets_tend/count_2d
   mean_extbdy_temp_targets_tend = mean_extbdy_temp_targets_tend/count_3d
   mean_extbdy_u_targets_tend = mean_extbdy_u_targets_tend/count_3d
   mean_extbdy_v_targets_tend = mean_extbdy_v_targets_tend/count_3d
   mean_extbdy_eta_targets_tend = mean_extbdy_eta_targets_tend/count_2d
   mean_nonextbdy_temp_targets_tend = mean_nonextbdy_temp_targets_tend/count_3d
   mean_nonextbdy_u_targets_tend = mean_nonextbdy_u_targets_tend/count_3d
   mean_nonextbdy_v_targets_tend = mean_nonextbdy_v_targets_tend/count_3d
   mean_nonextbdy_eta_targets_tend = mean_nonextbdy_eta_targets_tend/count_2d

   text_filename = '../../../Channel_nn_Outputs/'+model_name+'/STATS/'+model_name+'_'+str(no_epochs)+'epochs_StatsOutput_'+file_append+'.txt'
   with open(text_filename, "w") as text_file:
      text_file.write('\n')
      text_file.write('\n')
      text_file.write('count_samples ; %s \n' % count_samples)
      text_file.write('count_3d : %s \n' % count_3d)
      text_file.write('count_2d ; %s \n' % count_2d)
      text_file.write('bdy_count_3d ; %s \n' % bdy_count_3d)
      text_file.write('bdy_count_2d ; %s \n' % bdy_count_2d)
      text_file.write('nonbdy_count_3d ; %s \n' % nonbdy_count_3d)
      text_file.write('nonbdy_count_2d ; %s \n' % nonbdy_count_2d)
      text_file.write('extbdy_count_3d ; %s \n' % extbdy_count_3d)
      text_file.write('extbdy_count_2d ; %s \n' % extbdy_count_2d)
      text_file.write('nonextbdy_count_3d ; %s \n' % nonextbdy_count_3d)
      text_file.write('nonextbdy_count_2d ; %s \n' % nonextbdy_count_2d)

      text_file.write('\n')
      text_file.write('mean_temp_targets_tend ; %s \n' % mean_temp_targets_tend)
      text_file.write('mean_eta_targets_tend ; %s \n' % mean_eta_targets_tend)
      text_file.write('mean_u_targets_tend ; %s \n' % mean_u_targets_tend)
      text_file.write('mean_v_targets_tend ; %s \n' % mean_v_targets_tend)

      text_file.write('\n')
      text_file.write('\n')
      text_file.write('rms_error ; %s \n' % rms_error)
      text_file.write('pers_rms_error ; %s \n' % pers_rms_error)
      text_file.write('\n')
      text_file.write('temp_rms_error ; %s \n' % temp_rms_error)
      text_file.write('eta_rms_error  ; %s \n' % eta_rms_error)
      text_file.write('u_rms_error    ; %s \n' % u_rms_error)
      text_file.write('v_rms_error    ; %s \n' % v_rms_error)
      text_file.write('pers_temp_rms_error ; %s \n' % pers_temp_rms_error)
      text_file.write('pers_eta_rms_error  ; %s \n' % pers_eta_rms_error)
      text_file.write('pers_u_rms_error    ; %s \n' % pers_u_rms_error)
      text_file.write('pers_v_rms_error    ; %s \n' % pers_v_rms_error)
      text_file.write('\n')
      text_file.write('norm_rms_error      ; %s \n' % np.divide(rms_error,pers_rms_error))
      text_file.write('norm_temp_rms_error ; %s \n' % np.divide(temp_rms_error,pers_temp_rms_error))
      text_file.write('norm_eta_rms_error  ; %s \n' % np.divide(eta_rms_error,pers_eta_rms_error))
      text_file.write('norm_u_rms_error    ; %s \n' % np.divide(u_rms_error,pers_u_rms_error))
      text_file.write('norm_v_rms_error    ; %s \n' % np.divide(v_rms_error,pers_v_rms_error))

      text_file.write('\n')
      text_file.write('\n')
      text_file.write('bdy_rms_error ; %s \n' % bdy_rms_error)
      text_file.write('pers_bdy_rms_error ; %s \n' % pers_bdy_rms_error)
      text_file.write('\n')
      text_file.write('bdy_temp_rms_error ; %s \n' % bdy_temp_rms_error)
      text_file.write('bdy_eta_rms_error  ; %s \n' % bdy_eta_rms_error)
      text_file.write('bdy_u_rms_error    ; %s \n' % bdy_u_rms_error)
      text_file.write('bdy_v_rms_error    ; %s \n' % bdy_v_rms_error)
      text_file.write('pers_bdy_temp_rms_error ; %s \n' % pers_bdy_temp_rms_error)
      text_file.write('pers_bdy_eta_rms_error  ; %s \n' % pers_bdy_eta_rms_error)
      text_file.write('pers_bdy_u_rms_error    ; %s \n' % pers_bdy_u_rms_error)
      text_file.write('pers_bdy_v_rms_error    ; %s \n' % pers_bdy_v_rms_error)
      text_file.write('\n')
      text_file.write('norm_bdy_rms_error      ; %s \n' % np.divide(bdy_rms_error,pers_bdy_rms_error))
      text_file.write('norm_bdy_temp_rms_error ; %s \n' % np.divide(bdy_temp_rms_error,pers_bdy_temp_rms_error))
      text_file.write('norm_bdy_eta_rms_error  ; %s \n' % np.divide(bdy_eta_rms_error,pers_bdy_eta_rms_error))
      text_file.write('norm_bdy_u_rms_error    ; %s \n' % np.divide(bdy_u_rms_error,pers_bdy_u_rms_error))
      text_file.write('norm_bdy_v_rms_error    ; %s \n' % np.divide(bdy_v_rms_error,pers_bdy_v_rms_error))

      text_file.write('\n')
      text_file.write('\n')
      text_file.write('nonbdy_rms_error ; %s \n' % rms_error)
      text_file.write('pers_nonbdy_rms_error ; %s \n' % pers_rms_error)
      text_file.write('\n')
      text_file.write('nonbdy_temp_rms_error ; %s \n' % nonbdy_temp_rms_error)
      text_file.write('nonbdy_eta_rms_error  ; %s \n' % nonbdy_eta_rms_error)
      text_file.write('nonbdy_u_rms_error    ; %s \n' % nonbdy_u_rms_error)
      text_file.write('nonbdy_v_rms_error    ; %s \n' % nonbdy_v_rms_error)
      text_file.write('pers_nonbdy_temp_rms_error ; %s \n' % pers_nonbdy_temp_rms_error)
      text_file.write('pers_nonbdy_eta_rms_error  ; %s \n' % pers_nonbdy_eta_rms_error)
      text_file.write('pers_nonbdy_u_rms_error    ; %s \n' % pers_nonbdy_u_rms_error)
      text_file.write('pers_nonbdy_v_rms_error    ; %s \n' % pers_nonbdy_v_rms_error)
      text_file.write('\n')
      text_file.write('norm_nonbdy_rms_error      ; %s \n' % np.divide(nonbdy_rms_error,pers_nonbdy_rms_error))
      text_file.write('norm_nonbdy_temp_rms_error ; %s \n' % np.divide(nonbdy_temp_rms_error,pers_nonbdy_temp_rms_error))
      text_file.write('norm_nonbdy_eta_rms_error  ; %s \n' % np.divide(nonbdy_eta_rms_error,pers_nonbdy_eta_rms_error))
      text_file.write('norm_nonbdy_u_rms_error    ; %s \n' % np.divide(nonbdy_u_rms_error,pers_nonbdy_u_rms_error))
      text_file.write('norm_nonbdy_v_rms_error    ; %s \n' % np.divide(nonbdy_v_rms_error,pers_nonbdy_v_rms_error))

      #text_file.write('\n')
      #text_file.write('extbdy_temp_rms_error ; %s \n' % extbdy_temp_rms_error)
      #text_file.write('extbdy_eta_rms_error  ; %s \n' % extbdy_eta_rms_error)
      #text_file.write('extbdy_u_rms_error    ; %s \n' % extbdy_u_rms_error)
      #text_file.write('extbdy_v_rms_error    ; %s \n' % extbdy_v_rms_error)
      #text_file.write('pers_extbdy_temp_rms_error ; %s \n' % pers_extbdy_temp_rms_error)
      #text_file.write('pers_extbdy_eta_rms_error  ; %s \n' % pers_extbdy_eta_rms_error)
      #text_file.write('pers_extbdy_u_rms_error    ; %s \n' % pers_extbdy_u_rms_error)
      #text_file.write('pers_extbdy_v_rms_error    ; %s \n' % pers_extbdy_v_rms_error)

      #text_file.write('\n')
      #text_file.write('nonextbdy_temp_rms_error ; %s \n' % nonextbdy_temp_rms_error)
      #text_file.write('nonextbdy_eta_rms_error  ; %s \n' % nonextbdy_eta_rms_error)
      #text_file.write('nonextbdy_u_rms_error    ; %s \n' % nonextbdy_u_rms_error)
      #text_file.write('nonextbdy_v_rms_error    ; %s \n' % nonextbdy_v_rms_error)
      #text_file.write('pers_nonextbdy_temp_rms_error ; %s \n' % pers_nonextbdy_temp_rms_error)
      #text_file.write('pers_nonextbdy_eta_rms_error  ; %s \n' % pers_nonextbdy_eta_rms_error)
      #text_file.write('pers_nonextbdy_u_rms_error    ; %s \n' % pers_nonextbdy_u_rms_error)
      #text_file.write('pers_nonextbdy_v_rms_error    ; %s \n' % pers_nonextbdy_v_rms_error)

      text_file.write('\n')
      text_file.write('normalised_temp_rms_error ; %s \n' % np.divide(temp_rms_error,mean_temp_targets_tend))
      text_file.write('normalised_eta_rms_error  ; %s \n' % np.divide(eta_rms_error,mean_eta_targets_tend))
      text_file.write('normalised_u_rms_error    ; %s \n' % np.divide(u_rms_error,mean_u_targets_tend))
      text_file.write('normalised_v_rms_error    ; %s \n' % np.divide(v_rms_error,mean_v_targets_tend))
      text_file.write('pers_normalised_temp_rms_error ; %s \n' % np.divide(pers_temp_rms_error,mean_temp_targets_tend))
      text_file.write('pers_normalised_eta_rms_error  ; %s \n' % np.divide(pers_eta_rms_error,mean_eta_targets_tend))
      text_file.write('pers_normalised_u_rms_error    ; %s \n' % np.divide(pers_u_rms_error,mean_u_targets_tend))
      text_file.write('pers_normalised_v_rms_error    ; %s \n' % np.divide(pers_v_rms_error,mean_v_targets_tend))

      text_file.write('\n')
      text_file.write('normalised_bdy_temp_rms_error ; %s \n' % np.divide(bdy_temp_rms_error,mean_bdy_temp_targets_tend))
      text_file.write('normalised_bdy_eta_rms_error  ; %s \n' % np.divide(bdy_eta_rms_error,mean_bdy_eta_targets_tend))
      text_file.write('normalised_bdy_u_rms_error    ; %s \n' % np.divide(bdy_u_rms_error,mean_bdy_u_targets_tend))
      text_file.write('normalised_bdy_v_rms_error    ; %s \n' % np.divide(bdy_v_rms_error,mean_bdy_v_targets_tend))
      text_file.write('pers_normalised_bdy_temp_rms_error ; %s \n' % np.divide(pers_bdy_temp_rms_error,mean_bdy_temp_targets_tend))
      text_file.write('pers_normalised_bdy_eta_rms_error  ; %s \n' % np.divide(pers_bdy_eta_rms_error,mean_bdy_eta_targets_tend))
      text_file.write('pers_normalised_bdy_u_rms_error    ; %s \n' % np.divide(pers_bdy_u_rms_error,mean_bdy_u_targets_tend))
      text_file.write('pers_normalised_bdy_v_rms_error    ; %s \n' % np.divide(pers_bdy_v_rms_error,mean_bdy_v_targets_tend))

      text_file.write('\n')
      text_file.write('normalised_nonbdy_temp_rms_error ; %s \n' % np.divide(nonbdy_temp_rms_error,mean_nonbdy_temp_targets_tend))
      text_file.write('normalised_nonbdy_eta_rms_error  ; %s \n' % np.divide(nonbdy_eta_rms_error,mean_nonbdy_eta_targets_tend))
      text_file.write('normalised_nonbdy_u_rms_error    ; %s \n' % np.divide(nonbdy_u_rms_error,mean_nonbdy_u_targets_tend))
      text_file.write('normalised_nonbdy_v_rms_error    ; %s \n' % np.divide(nonbdy_v_rms_error,mean_nonbdy_v_targets_tend))
      text_file.write('pers_normalised_nonbdy_temp_rms_error ; %s \n' % np.divide(pers_nonbdy_temp_rms_error,mean_nonbdy_temp_targets_tend))
      text_file.write('pers_normalised_nonbdy_eta_rms_error  ; %s \n' % np.divide(pers_nonbdy_eta_rms_error,mean_nonbdy_eta_targets_tend))
      text_file.write('pers_normalised_nonbdy_u_rms_error    ; %s \n' % np.divide(pers_nonbdy_u_rms_error,mean_nonbdy_u_targets_tend))
      text_file.write('pers_normalised_nonbdy_v_rms_error    ; %s \n' % np.divide(pers_nonbdy_v_rms_error,mean_nonbdy_v_targets_tend))

      #text_file.write('\n')
      #text_file.write('normalised_extbdy_temp_rms_error ; %s \n' % extbdy_temp_rms_error/mean_extbdy_temp_targets_tend)
      #text_file.write('normalised_extbdy_eta_rms_error  ; %s \n' % extbdy_eta_rms_error/mean_extbdy_eta_targets_tend)
      #text_file.write('normalised_extbdy_u_rms_error    ; %s \n' % extbdy_u_rms_error/mean_extbdy_u_targets_tend)
      #text_file.write('normalised_extbdy_v_rms_error    ; %s \n' % extbdy_v_rms_error/mean_extbdy_v_targets_tend)
      #text_file.write('pers_normalised_extbdy_temp_rms_error ; %s \n' % pers_extbdy_temp_rms_error/mean_extbdy_temp_targets_tend)
      #text_file.write('pers_normalised_extbdy_eta_rms_error  ; %s \n' % pers_extbdy_eta_rms_error/mean_extbdy_eta_targets_tend)
      #text_file.write('pers_normalised_extbdy_u_rms_error    ; %s \n' % pers_extbdy_u_rms_error/mean_extbdy_u_targets_tend)
      #text_file.write('pers_normalised_extbdy_v_rms_error    ; %s \n' % pers_extbdy_v_rms_error/mean_extbdy_v_targets_tend)

      #text_file.write('\n')
      #text_file.write('normalised_nonextbdy_temp_rms_error ; %s \n' % nonextbdy_temp_rms_error/mean_nonextbdy_temp_targets_tend)
      #text_file.write('normalised_nonextbdy_eta_rms_error  ; %s \n' % nonextbdy_eta_rms_error/mean_nonextbdy_eta_targets_tend)
      #text_file.write('normalised_nonextbdy_u_rms_error    ; %s \n' % nonextbdy_u_rms_error/mean_nonextbdy_u_targets_tend)
      #text_file.write('normalised_nonextbdy_v_rms_error    ; %s \n' % nonextbdy_v_rms_error/mean_nonextbdy_v_targets_tend)
      #text_file.write('pers_normalised_nonextbdy_temp_rms_error ; %s \n' % pers_nonextbdy_temp_rms_error/mean_nonextbdy_temp_targets_tend)
      #text_file.write('pers_normalised_nonextbdy_eta_rms_error  ; %s \n' % pers_nonextbdy_eta_rms_error/mean_nonextbdy_eta_targets_tend)
      #text_file.write('pers_normalised_nonextbdy_u_rms_error    ; %s \n' % pers_nonextbdy_u_rms_error/mean_nonextbdy_u_targets_tend)
      #text_file.write('pers_normalised_nonextbdy_v_rms_error    ; %s \n' % pers_nonextbdy_v_rms_error/mean_nonextbdy_v_targets_tend)

   # Set up netcdf files
   nc_filename = '../../../Channel_nn_Outputs/'+model_name+'/STATS/'+model_name+'_'+str(no_epochs)+'epochs_StatsOutput_'+file_append+'.nc'
   nc_file = nc4.Dataset(nc_filename,'w', format='NETCDF4') #'w' stands for write
   # Create Dimensions
   z_dim = 38  # Hard coded...perhaps should change...?
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

   nc_TrueTempTend = nc_file.createVariable( 'True_Temp_tend'  , 'f4', ('T', 'Z', 'Y', 'X') )
   nc_TrueUTend    = nc_file.createVariable( 'True_U_tend'     , 'f4', ('T', 'Z', 'Y', 'X') )
   nc_TrueVTend    = nc_file.createVariable( 'True_V_tend'     , 'f4', ('T', 'Z', 'Y', 'X') )
   nc_TrueEtaTend  = nc_file.createVariable( 'True_Eta_tend'   , 'f4', ('T', 'Y', 'X')      )

   nc_TempMask   = nc_file.createVariable( 'Temp_Mask'  , 'f4', ('Z', 'Y', 'X') )
   nc_UMask      = nc_file.createVariable( 'U_Mask'     , 'f4', ('Z', 'Y', 'X') )
   nc_VMask      = nc_file.createVariable( 'V_Mask'     , 'f4', ('Z', 'Y', 'X') )
   nc_EtaMask    = nc_file.createVariable( 'Eta_Mask'   , 'f4', ('Y', 'X')      )

   nc_PredTemp   = nc_file.createVariable( 'Pred_Temp'  , 'f4', ('T', 'Z', 'Y', 'X') )
   nc_PredU      = nc_file.createVariable( 'Pred_U'     , 'f4', ('T', 'Z', 'Y', 'X') )
   nc_PredV      = nc_file.createVariable( 'Pred_V'     , 'f4', ('T', 'Z', 'Y', 'X') ) 
   nc_PredEta    = nc_file.createVariable( 'Pred_Eta'   , 'f4', ('T', 'Y', 'X')      )

   nc_PredTempTend = nc_file.createVariable( 'Pred_Temp_tend'  , 'f4', ('T', 'Z', 'Y', 'X') )
   nc_PredUTend    = nc_file.createVariable( 'Pred_U_tend'     , 'f4', ('T', 'Z', 'Y', 'X') )
   nc_PredVTend    = nc_file.createVariable( 'Pred_V_tend'     , 'f4', ('T', 'Z', 'Y', 'X') ) 
   nc_PredEtaTend  = nc_file.createVariable( 'Pred_Eta_tend'   , 'f4', ('T', 'Y', 'X')      )

   nc_TempErrors = nc_file.createVariable( 'Temp_Errors', 'f4', ('T', 'Z', 'Y', 'X') )
   nc_UErrors    = nc_file.createVariable( 'U_Errors'   , 'f4', ('T', 'Z', 'Y', 'X') )
   nc_VErrors    = nc_file.createVariable( 'V_Errors'   , 'f4', ('T', 'Z', 'Y', 'X') )
   nc_EtaErrors  = nc_file.createVariable( 'Eta_Errors' , 'f4', ('T', 'Y', 'X')      )

   nc_TempErrorsTend = nc_file.createVariable( 'Temp_Errors_tend', 'f4', ('T', 'Z', 'Y', 'X') )
   nc_UErrorsTend    = nc_file.createVariable( 'U_Errors_tend'   , 'f4', ('T', 'Z', 'Y', 'X') )
   nc_VErrorsTend    = nc_file.createVariable( 'V_Errors_tend'   , 'f4', ('T', 'Z', 'Y', 'X') )
   nc_EtaErrorsTend  = nc_file.createVariable( 'Eta_Errors_tend' , 'f4', ('T', 'Y', 'X')      )

   nc_TempRMS    = nc_file.createVariable( 'Temp_RMS'   , 'f4', ('Z', 'Y', 'X')      )
   nc_U_RMS      = nc_file.createVariable( 'U_RMS'      , 'f4', ('Z', 'Y', 'X')      )
   nc_V_RMS      = nc_file.createVariable( 'V_RMS'      , 'f4', ('Z', 'Y', 'X')      )
   nc_EtaRMS     = nc_file.createVariable( 'Eta_RMS'    , 'f4', ('Y', 'X')           )

   nc_Scaled_TempRMS = nc_file.createVariable( 'ScaledTempRMS' , 'f4', ('Z', 'Y', 'X')      )
   nc_Scaled_U_RMS   = nc_file.createVariable( 'Scaled_U_RMS'  , 'f4', ('Z', 'Y', 'X')      )
   nc_Scaled_V_RMS   = nc_file.createVariable( 'Scaled_V_RMS'  , 'f4', ('Z', 'Y', 'X')      )
   nc_Scaled_EtaRMS  = nc_file.createVariable( 'ScaledEtaRMS'  , 'f4', ('Y', 'X')           )

   nc_MeanObsTempTend = nc_file.createVariable( 'MeanTempTend' , 'f4', ('Z', 'Y', 'X')      )
   nc_MeanObsUTend    = nc_file.createVariable( 'MeanUTend'    , 'f4', ('Z', 'Y', 'X')      )
   nc_MeanObsVTend    = nc_file.createVariable( 'MeanVTend'    , 'f4', ('Z', 'Y', 'X')      )
   nc_MeanObsEtaTend  = nc_file.createVariable( 'MeanEtaTend'  , 'f4', ('Y', 'X')           )

   nc_Normalised_Temp_RMS = nc_file.createVariable( 'NormTempRMS' , 'f4', ('Z', 'Y', 'X')      )
   nc_Normalised_U_RMS    = nc_file.createVariable( 'NormURMS'    , 'f4', ('Z', 'Y', 'X')      )
   nc_Normalised_V_RMS    = nc_file.createVariable( 'NormVRMS'    , 'f4', ('Z', 'Y', 'X')      )
   nc_Normalised_Eta_RMS  = nc_file.createVariable( 'NormEtaRMS'  , 'f4', ('Y', 'X')           )

   # Fill variables
   nc_T[:] = np.arange(no_samples)
   nc_Z[:] = da_Z.values
   nc_Y[:] = da_Y.values[:y_dim_used]
   nc_X[:] = da_X.values

   if dimension=='2d':
      nc_TrueTemp[:,:,:,:] = nc_targets_fld[:no_samples,0:z_dim,:,:] 
      nc_TrueU[:,:,:,:]    = nc_targets_fld[:no_samples,1*z_dim:2*z_dim,:,:]
      nc_TrueV[:,:,:,:]    = nc_targets_fld[:no_samples,2*z_dim:3*z_dim,:,:]
      nc_TrueEta[:,:,:]    = nc_targets_fld[:no_samples,3*z_dim,:,:]

      nc_TrueTempTend[:,:,:,:] = nc_targets_tend[:no_samples,0:z_dim,:,:] 
      nc_TrueUTend[:,:,:,:]    = nc_targets_tend[:no_samples,1*z_dim:2*z_dim,:,:]
      nc_TrueVTend[:,:,:,:]    = nc_targets_tend[:no_samples,2*z_dim:3*z_dim,:,:]
      nc_TrueEtaTend[:,:,:]    = nc_targets_tend[:no_samples,3*z_dim,:,:]

      nc_TempMask[:,:,:] = out_masks[0,0:z_dim,:,:]
      nc_UMask[:,:,:]    = out_masks[0,1*z_dim:2*z_dim,:,:]
      nc_VMask[:,:,:]    = out_masks[0,2*z_dim:3*z_dim,:,:]
      nc_EtaMask[:,:]    = out_masks[0,3*z_dim,:,:]

      nc_PredTemp[:,:,:,:] = nc_predictions_fld[:no_samples,0:z_dim,:,:]
      nc_PredU[:,:,:,:]    = nc_predictions_fld[:no_samples,1*z_dim:2*z_dim,:,:]
      nc_PredV[:,:,:,:]    = nc_predictions_fld[:no_samples,2*z_dim:3*z_dim,:,:]
      nc_PredEta[:,:,:]    = nc_predictions_fld[:no_samples,3*z_dim,:,:]
 
      nc_PredTempTend[:,:,:,:] = nc_predictions_tend[:no_samples,0:z_dim,:,:]
      nc_PredUTend[:,:,:,:]    = nc_predictions_tend[:no_samples,1*z_dim:2*z_dim,:,:]
      nc_PredVTend[:,:,:,:]    = nc_predictions_tend[:no_samples,2*z_dim:3*z_dim,:,:]
      nc_PredEtaTend[:,:,:]    = nc_predictions_tend[:no_samples,3*z_dim,:,:]
 
      nc_TempErrors[:,:,:,:] = ( nc_predictions_fld[:no_samples,0:z_dim,:,:]
                                 - nc_targets_fld[:no_samples,0:z_dim,:,:] )
      nc_UErrors[:,:,:,:]    = ( nc_predictions_fld[:no_samples,1*z_dim:2*z_dim,:,:]
                                 - nc_targets_fld[:no_samples,1*z_dim:2*z_dim,:,:] )
      nc_VErrors[:,:,:,:]    = ( nc_predictions_fld[:no_samples,2*z_dim:3*z_dim,:,:] 
                                 - nc_targets_fld[:no_samples,2*z_dim:3*z_dim,:,:] )
      nc_EtaErrors[:,:,:]    = ( nc_predictions_fld[:no_samples,3*z_dim,:,:] 
                                 - nc_targets_fld[:no_samples,3*z_dim,:,:] )

      nc_TempErrorsTend[:,:,:,:] = ( nc_predictions_tend[:no_samples,0:z_dim,:,:]
                                     - nc_targets_tend[:no_samples,0:z_dim,:,:] )
      nc_UErrorsTend[:,:,:,:]    = ( nc_predictions_tend[:no_samples,1*z_dim:2*z_dim,:,:]
                                     - nc_targets_tend[:no_samples,1*z_dim:2*z_dim,:,:] )
      nc_VErrorsTend[:,:,:,:]    = ( nc_predictions_tend[:no_samples,2*z_dim:3*z_dim,:,:] 
                                     - nc_targets_tend[:no_samples,2*z_dim:3*z_dim,:,:] )
      nc_EtaErrorsTend[:,:,:]    = ( nc_predictions_tend[:no_samples,3*z_dim,:,:] 
                                     - nc_targets_tend[:no_samples,3*z_dim,:,:] )

      nc_TempRMS[:,:,:] = spatial_rms_error[0:z_dim,:,:]
      nc_U_RMS[:,:,:]   = spatial_rms_error[1*z_dim:2*z_dim,:,:] 
      nc_V_RMS[:,:,:]   = spatial_rms_error[2*z_dim:3*z_dim,:,:]
      nc_EtaRMS[:,:]    = spatial_rms_error[3*z_dim,:,:]

      # interp MITgcm Std values onto T grid for use in scaling
      MITgcm_StdUVel = ( MITgcm_stats_ds['StdUVel'].values[:,:,:1]+MITgcm_stats_ds['StdUVel'].values[:,:,:-1] ) / 2.
      MITgcm_StdVVel = ( MITgcm_stats_ds['StdVVel'].values[:,1:,:]+MITgcm_stats_ds['StdVVel'].values[:,:-1,:] ) / 2.

      nc_Scaled_TempRMS[:,:,:] = spatial_rms_error[0:z_dim,:,:] / MITgcm_stats_ds['StdTemp'].values
      nc_Scaled_U_RMS[:,:,:]   = spatial_rms_error[1*z_dim:2*z_dim,:,:] / MITgcm_StdUVel
      nc_Scaled_V_RMS[:,:,:]   = spatial_rms_error[2*z_dim:3*z_dim,:,:] / MITgcm_StdVVel
      nc_Scaled_EtaRMS[:,:]    = spatial_rms_error[3*z_dim,:,:] / MITgcm_stats_ds['StdEta'].values

      nc_MeanObsTempTend[:,:,:] = mean_targets_tend[0:z_dim,:,:]
      nc_MeanObsUTend[:,:,:]    = mean_targets_tend[1*z_dim:2*z_dim,:,:]
      nc_MeanObsVTend[:,:,:]    = mean_targets_tend[2*z_dim:3*z_dim,:,:]
      nc_MeanObsEtaTend[:,:]    = mean_targets_tend[3*z_dim,:,:]

      nc_Normalised_Temp_RMS[:,:,:] = spatial_rms_error[0:z_dim,:,:] / np.nanmean(mean_targets_tend[0:z_dim,:,:])
      nc_Normalised_U_RMS[:,:,:]    = spatial_rms_error[1*z_dim:2*z_dim,:,:] / np.nanmean(mean_targets_tend[1*z_dim:2*z_dim,:,:])
      nc_Normalised_V_RMS[:,:,:]    = spatial_rms_error[2*z_dim:3*z_dim,:,:] / np.nanmean(mean_targets_tend[2*z_dim:3*z_dim,:,:])
      nc_Normalised_Eta_RMS[:,:]    = spatial_rms_error[3*z_dim,:,:] / np.nanmean(mean_targets_tend[3*z_dim,:,:])

   elif dimension=='3d':
      nc_TrueTemp[:,:,:,:] = nc_targets_fld[:no_samples,0,:,:,:] 
      nc_TrueU[:,:,:,:]    = nc_targets_fld[:no_samples,1,:,:,:]
      nc_TrueV[:,:,:,:]    = nc_targets_fld[:no_samples,2,:,:,:]
      nc_TrueEta[:,:,:]    = nc_targets_fld[:no_samples,3,0,:,:]

      nc_TrueTempTend[:,:,:,:] = nc_targets_tend[:no_samples,0,:,:,:] 
      nc_TrueUTend[:,:,:,:]    = nc_targets_tend[:no_samples,1,:,:,:]
      nc_TrueVTend[:,:,:,:]    = nc_targets_tend[:no_samples,2,:,:,:]
      nc_TrueEtaTend[:,:,:]    = nc_targets_tend[:no_samples,3,0,:,:]

      nc_TempMask[:,:,:] = out_masks[0,0,:,:,:]
      nc_UMask[:,:,:]    = out_masks[0,1,:,:,:]
      nc_VMask[:,:,:]    = out_masks[0,2,:,:,:]
      nc_EtaMask[:,:]    = out_masks[0,3,0,:,:]

      nc_PredTemp[:,:,:,:] = nc_predictions_fld[:no_samples,0,:,:,:]
      nc_PredU[:,:,:,:]    = nc_predictions_fld[:no_samples,1,:,:,:]
      nc_PredV[:,:,:,:]    = nc_predictions_fld[:no_samples,2,:,:,:]
      nc_PredEta[:,:,:]    = nc_predictions_fld[:no_samples,3,0,:,:]
 
      nc_PredTempTend[:,:,:,:] = nc_predictions_tend[:no_samples,0,:,:,:]
      nc_PredUTend[:,:,:,:]    = nc_predictions_tend[:no_samples,1,:,:,:]
      nc_PredVTend[:,:,:,:]    = nc_predictions_tend[:no_samples,2,:,:,:]
      nc_PredEtaTend[:,:,:]    = nc_predictions_tend[:no_samples,3,0,:,:]
 
      nc_TempErrors[:,:,:,:] = ( nc_predictions_fld[:no_samples,0,:,:,:]
                                 - nc_targets_fld[:no_samples,0,:,:,:] )
      nc_UErrors[:,:,:,:]    = ( nc_predictions_fld[:no_samples,1,:,:,:]
                                 - nc_targets_fld[:no_samples,1,:,:,:] )
      nc_VErrors[:,:,:,:]    = ( nc_predictions_fld[:no_samples,2,:,:,:] 
                                 - nc_targets_fld[:no_samples,2,:,:,:] )
      nc_EtaErrors[:,:,:]    = ( nc_predictions_fld[:no_samples,3,0,:,:] 
                                 - nc_targets_fld[:no_samples,3,0,:,:] )

      nc_TempRMS[:,:,:] = spatial_rms_error[0,:,:,:]
      nc_U_RMS[:,:,:]   = spatial_rms_error[1,:,:,:] 
      nc_V_RMS[:,:,:]   = spatial_rms_error[2,:,:,:]
      nc_EtaRMS[:,:]    = spatial_rms_error[3,0,:,:]


def IterativelyPredict(model_name, model_style, MITgcm_filename, Iterate_Dataset, h, start, for_len, no_epochs, y_dim_used,
                       land, dimension, histlen, landvalues, iterate_method, iterate_smooth, smooth_steps, norm_method,
                       channel_dim, mean_std_file, for_subsample, no_phys_in_channels, no_out_channels, seed, multipredmethod):
   logging.info('Iterating')

   landvalues = torch.tensor(landvalues)
   output_count = 0
  
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
   da_Y = MITgcm_ds['Y']
   if land == 'ExcLand':
      da_Y = da_Y[3:101]
   da_Z = MITgcm_ds['Zmd000038'] 

   z_dim = da_Z.shape[0]

   inputs_mean, inputs_std, inputs_range, targets_mean, targets_std, targets_range = \
                                   ReadMeanStd(mean_std_file, dimension, no_phys_in_channels, no_out_channels, da_Z.shape[0], seed)
   
   # Read in data from MITgcm dataset (take 0th entry, as others are target, masks etc) and save as first entry in both arrays
   input_sample, target_sample, extra_fluxes_sample, Tmask, Umask, Vmask, out_masks, bdy_masks = Iterate_Dataset.__getitem__(start)

   # Give extra dimension at front (number of samples - here 1)
   input_sample = input_sample.unsqueeze(0).float()
   Tmask = Tmask.unsqueeze(0).float()
   Umask = Umask.unsqueeze(0).float()
   Vmask = Vmask.unsqueeze(0).float()
   if model_style == 'ConvLSTM' or model_style == 'UNetConvLSTM':
      iterated_fields = rr.RF_DeNormalise(Iterate_Dataset.__getitem__(start)[0][:histlen,:3*z_dim+1,:,:],
                                          inputs_mean[:3*z_dim+1], inputs_std[:3*z_dim+1], inputs_range[:3*z_dim+1],
                                          dimension, norm_method, seed)
      out_iterated_fields = rr.RF_DeNormalise(Iterate_Dataset.__getitem__(start)[0][:histlen:for_subsample,:3*z_dim+1,:,:],
                                          inputs_mean[:3*z_dim+1], inputs_std[:3*z_dim+1], inputs_range[:3*z_dim+1],
                                          dimension, norm_method, seed)
      MITgcm_data = Iterate_Dataset.__getitem__(start)[0][:histlen:for_subsample,:3*z_dim+1,:,:]
   elif dimension == '2d':
      out_iterated_fields = rr.RF_DeNormalise(Iterate_Dataset.__getitem__(start)[0][:3*z_dim+1,:,:].unsqueeze(0),
                                          inputs_mean[:3*z_dim+1], inputs_std[:3*z_dim+1], inputs_range[:3*z_dim+1], 
                                          dimension, norm_method, seed)
      iterated_fields = rr.RF_DeNormalise(Iterate_Dataset.__getitem__(start)[0][:3*z_dim+1,:,:].unsqueeze(0),
                                          inputs_mean[:3*z_dim+1], inputs_std[:3*z_dim+1], inputs_range[:3*z_dim+1], 
                                          dimension, norm_method, seed)
      MITgcm_data = Iterate_Dataset.__getitem__(start)[0][:3*z_dim+1,:,:].unsqueeze(0)
      for histtime in range(1,histlen):
         iterated_fields = torch.cat( (iterated_fields,
                            rr.RF_DeNormalise(Iterate_Dataset.__getitem__(start)[0]
                                                 [histtime*no_phys_in_channels:histtime*no_phys_in_channels+3*z_dim+1,:,:].unsqueeze(0),
                                              inputs_mean[:3*z_dim+1], inputs_std[:3*z_dim+1], inputs_range[:3*z_dim+1],
                                              dimension, norm_method, seed) ),
                            axis = 0)
         MITgcm_data = torch.cat( ( MITgcm_data,
                                  Iterate_Dataset.__getitem__(start)[0]
                                     [histtime*no_phys_in_channels:histtime*no_phys_in_channels+3*z_dim+1,:,:].unsqueeze(0) ),
                                  axis=0 )
         if ( histtime%for_subsample == 0 ):
            out_iterated_fields = torch.cat( (out_iterated_fields,
                               rr.RF_DeNormalise(Iterate_Dataset.__getitem__(start)[0]
                                                    [histtime*no_phys_in_channels:histtime*no_phys_in_channels+3*z_dim+1,:,:].unsqueeze(0),
                                                 inputs_mean[:3*z_dim+1], inputs_std[:3*z_dim+1], inputs_range[:3*z_dim+1],
                                                 dimension, norm_method, seed) ),
                               axis = 0)
   elif dimension == '3d':
      iterated_fields = rr.RF_DeNormalise(Iterate_Dataset.__getitem__(start)[0][:3*z_dim+1,:,:,:].unsqueeze(0),
                                          inputs_mean[:3*z_dim+1], inputs_std[:3*z_dim+1], inputs_range[:3*z_dim+1],
                                          dimension, norm_method, seed)
      MITgcm_data = Iterate_Dataset.__getitem__(start)[0][:3*z_dim+1,:,:,:].unsqueeze(0)
      for histtime in range(1,histlen):
         iterated_fields = torch.cat( (iterated_fields,
                             rr.RF_DeNormalise(Iterate_Dataset.__getitem__(start)[0]
                                                  [histtime*no_phys_in_channels:histtime*no_phys_in_channels+3*z_dim+1,:,:,:].unsqueeze(0),
                                               inputs_mean[:3*z_dim+1], inputs_std[:3*z_dim+1], inputs_range[:3*z_dim+1],
                                               dimension, norm_method, seed) ),
                             axis = 0)
         MITgcm_data = torch.cat( (MITgcm_data, 
                                 Iterate_Dataset.__getitem__(start)[0]
                                    [histtime*no_phys_in_channels:histtime*no_phys_in_channels+3*z_dim+1,:,:].unsqueeze(0) ),
                                 axis=0)
         if ( histtime%for_subsample == 0 ):
            out_iterated_fields = torch.cat( (out_iterated_fields,
                                rr.RF_DeNormalise(Iterate_Dataset.__getitem__(start)[0]
                                                     [histtime*no_phys_in_channels:histtime*no_phys_in_channels+3*z_dim+1,:,:,:].unsqueeze(0),
                                                  inputs_mean[:3*z_dim+1], inputs_std[:3*z_dim+1], inputs_range[:3*z_dim+1],
                                                  dimension, norm_method, seed) ),
                                axis = 0)
   predicted_increments = torch.zeros(target_sample.shape).unsqueeze(0)[:,:no_out_channels,:,:]
   true_increments = torch.zeros(target_sample.shape).unsqueeze(0)[:,:no_out_channels,:,:]

   # Make iterative forecast, saving predictions as we go, and pull out MITgcm data at correct time steps
   with torch.no_grad():
      for fortime in range(start+histlen, start+for_len):
         print(fortime)
         # Make prediction, for both multi model ensemble, or single model predictions
         if multipredmethod == 'random':
            rand_model = random.randrange(len(h))
            print('rand_model '+str(rand_model))
            predicted = h[rand_model]( torch.cat( (input_sample, Tmask, Umask, Vmask), dim=channel_dim) ).cpu().detach()
         elif multipredmethod == 'average':
            predicted = h[0]( torch.cat((input_sample, Tmask, Umask, Vmask), dim=channel_dim)).cpu().detach()
            for i in range(1, len(h)):
               predicted = predicted + \
                           h[i]( torch.cat((input_sample, Tmask, Umask, Vmask), dim=channel_dim)).cpu().detach()
            predicted = predicted / len(h)
         elif multipredmethod == 'singlepred':
            if land == 'ExcLand':
               predicted = h( input_sample ).cpu().detach()
            else:
               predicted = h( torch.cat( (input_sample, Tmask, Umask, Vmask), dim=channel_dim ) ).cpu().detach()

         # Denormalise 
         predicted = rr.RF_DeNormalise(predicted, targets_mean, targets_std, targets_range, dimension, norm_method, seed)
         if ( fortime%for_subsample == 0 ):
            print('    '+str(fortime))
            predicted_increments = torch.cat( (predicted_increments, predicted), axis=0)

         # Calculate next field, by combining prediction (increment) with field
         if iterate_method == 'simple':
            next_field = iterated_fields[-1] + predicted
         elif iterate_method == 'AB2':
            if fortime == start+histlen:
               if dimension == '2d':
                  next_field = rr.RF_DeNormalise(Iterate_Dataset.__getitem__(fortime)[0][:3*z_dim+1,:,:].unsqueeze(0),
                                                 inputs_mean[:3*z_dim+1], inputs_std[:3*z_dim+1], inputs_range[:3*z_dim+1],
                                                 dimension, norm_method, seed)
               elif dimension == '3d':
                  next_field = rr.RF_DeNormalise(Iterate_Dataset.__getitem__(fortime)[0][:4,:,:,:].unsqueeze(0),
                                                 inputs_mean[:4], inputs_std[:4], inputs_range[:4],
                                                 dimension, norm_method, seed)
            else: 
               next_field = iterated_fields[-1] + (3./2.) * predicted - (1./2.) * old_predicted
            old_predicted = predicted
         elif iterate_method == 'AB3':
            if fortime == start+histlen:
               if dimension == '2d':
                  next_field = rr.RF_DeNormalise(Iterate_Dataset.__getitem__(fortime)[0][:3*z_dim+1,:,:].unsqueeze(0),
                                                 inputs_mean[:3*z_dim+1], inputs_std[:3*z_dim+1], inputs_range[:3*z_dim+1],
                                                 dimension, norm_method, seed)
               elif dimension == '3d':
                  next_field = rr.RF_DeNormalise(Iterate_Dataset.__getitem__(fortime)[0][:4,:,:,:].unsqueeze(0),
                                                 inputs_mean[:4], inputs_std[:4], inputs_range[:4],
                                                 dimension, norm_method, seed)
               old_predicted = predicted
            elif fortime == start+histlen+1:
               next_field = iterated_fields[-1] + (3./2.+.1) * predicted - (1./2.+.1) * old_predicted
               old2_predicted = old_predicted
               old_predicted = predicted
            else:
               next_field = iterated_fields[-1] + 23./12. * predicted - 16./12. * old_predicted + 5./12. * old2_predicted
               old2_predicted = old_predicted
               old_predicted = predicted
         # RK4 not set up to run with flux inputs - would need to work out what to pass in for fluxes....
         #elif iterate_method == 'RK4':
         #   if dimension == '2d':
         #      k1 = predicted
         #      # (k1)/2 added to state, masked, then normalised, and finally masks catted on. And then passed through neural net)
         #      k2 = h( torch.cat( (
         #                 rr.RF_Normalise( 
         #                     torch.where( out_masks==1, ( iterated_fields[-1,:,:,:] + (k1)/2. ), landvalues.unsqueeze(1).unsqueeze(2) ),
         #                     inputs_mean[:3*z_dim+1], inputs_std[:3*z_dim+1], inputs_range[:3*z_dim+1], dimension, norm_method), 
         #                 masks),axis=channel_dim )
         #            ).cpu().detach()
         #      k3 = h( torch.cat( (
         #                 rr.RF_Normalise( 
         #                     torch.where( out_masks==1, ( iterated_fields[-1,:,:,:] + (k2)/2. ), landvalues.unsqueeze(1).unsqueeze(2) ),
         #                     inputs_mean, inputs_std, inputs_range, dimension, norm_method), 
         #                 masks),axis=channel_dim )
         #            ).cpu().detach()
         #      k4 = h( torch.cat( (
         #                 rr.RF_Normalise( 
         #                    torch.where( out_masks==1, ( iterated_fields[-1,:,:,:] + (k3)    ), landvalues.unsqueeze(1).unsqueeze(2) ),
         #                    inputs_mean, inputs_std, inputs_range, dimension, norm_method), 
         #                 masks),axis=channel_dim )
         #            ).cpu().detach()
         #      next_field = iterated_fields[-1,:,:,:] + (1./6.) * ( k1 + 2*k2 +2*k3 + k4 )         
         #   elif dimension == '3d':
         #      k1 = predicted
         #      # (k1)/2 added to state, masked, then normalised, and finally masks catted on. And then passed through neural net)
         #      k2 = h( torch.cat( (
         #                 rr.RF_Normalise( 
         #                    torch.where( out_masks==1, ( iterated_fields[-1,:,:,:,:] + (k1)/2. ), landvalues.unsqueeze(1).unsqueeze(2).unsqueeze(3) ),
         #                    inputs_mean, inputs_std, inputs_range, dimension, norm_method), 
         #                 masks),axis=channel_dim )
         #            ).cpu().detach()
         #      k3 = h( torch.cat( (
         #                 rr.RF_Normalise( 
         #                    torch.where( out_masks==1, ( iterated_fields[-1,:,:,:,:] + (k2)/2. ), landvalues.unsqueeze(1).unsqueeze(2).unsqueeze(3) ),
         #                    inputs_mean, inputs_std, inputs_range, dimension, norm_method), 
         #                 masks),axis=channel_dim )
         #            ).cpu().detach()
         #      k4 = h( torch.cat( (
         #                 rr.RF_Normalise( 
         #                    torch.where( out_masks==1, ( iterated_fields[-1,:,:,:,:] + (k3)    ), landvalues.unsqueeze(1).unsqueeze(2).unsqueeze(3) ),
         #                    inputs_mean, inputs_std, inputs_range, dimension, norm_method), 
         #                 masks),axis=channel_dim )
         #            ).cpu().detach()
         #      next_field = iterated_fields[-1,:,:,:,:] + (1./6.) * ( k1 + 2*k2 +2*k3 + k4 )         
         else:
            sys.exit('no suitable iteration method chosen')

         # Mask field
         if dimension == '2d':
            next_field = torch.where(out_masks==1, next_field, landvalues[:3*z_dim+1].unsqueeze(1).unsqueeze(2) )
         if dimension == '3d':
            next_field = torch.where( out_masks==1, next_field, landvalues[:3*z_dim+1].unsqueeze(1).unsqueeze(2).unsqueeze(3) )

         # Smooth field 
         if iterate_smooth != 0:
            for channel in range(next_field.shape[1]):
               filter = gcm_filters.Filter( filter_scale=iterate_smooth, dx_min=10, filter_shape=gcm_filters.FilterShape.GAUSSIAN,
                                            grid_type=gcm_filters.GridType.REGULAR_WITH_LAND,
                                            grid_vars={'wet_mask': xr.DataArray(out_masks[channel,:,:], dims=['y','x'])},
                                            n_steps=smooth_steps )
               next_field[0,channel,:,:] = torch.from_numpy( filter.apply( xr.DataArray(next_field[0,channel,:,:], dims=['y', 'x'] ),
                                                             dims=['y', 'x']).values )

         iterated_fields = torch.cat( ( iterated_fields, next_field ), axis=0 )
         iterated_fields = iterated_fields[-2:]
         if ( fortime%for_subsample == 0 ):
            print('    '+str(fortime))
            print('    '+str(output_count))
            output_count = output_count+1
            # Cat prediction onto existing fields
            out_iterated_fields = torch.cat( ( out_iterated_fields, next_field ), axis=0 )
            # Get MITgcm data for relevant step, take first set for histlen>1, as looking at relevant time step
            if model_style == 'ConvLSTM' or model_style == 'UNetConvLSTM':
               MITgcm_data = torch.cat( ( MITgcm_data, Iterate_Dataset.__getitem__(fortime)[0][:1,:3*z_dim+1,:,:] ), axis=0)
               true_increments = torch.cat( ( true_increments, 
                                             Iterate_Dataset.__getitem__(fortime+1)[0][0:1,:3*z_dim+1,:,:]
                                             - Iterate_Dataset.__getitem__(fortime)[0][0:1,:3*z_dim+1,:,:] ), axis=0)
            elif dimension == '2d':
               MITgcm_data = torch.cat( ( MITgcm_data, Iterate_Dataset.__getitem__(fortime)[0][:3*z_dim+1,:,:].unsqueeze(0) ),
                                          axis=0 )
               true_increments = torch.cat( ( true_increments,
                                              Iterate_Dataset.__getitem__(fortime+1)[0][:3*z_dim+1,:,:].unsqueeze(0)
                                              - Iterate_Dataset.__getitem__(fortime)[0][:3*z_dim+1,:,:].unsqueeze(0)
                                            ),
                                           axis=0 )
            elif dimension == '3d':
               MITgcm_data = torch.cat( ( MITgcm_data, Iterate_Dataset.__getitem__(fortime)[0][:4,:,:,:].unsqueeze(0) ),
                                        axis=0 )
               true_increments = torch.cat( ( true_increments,
                                              Iterate_Dataset.__getitem__(fortime+1)[0][:4,:,:,:].unsqueeze(0)
                                              - Iterate_Dataset.__getitem__(fortime)[0][:4,:,:,:].unsqueeze(0)
                                            ),
                                           axis=0 )

         # Re-Normalise next field ready to use as inputs
         next_field = rr.RF_Normalise(next_field, inputs_mean[:3*z_dim+1], inputs_std[:3*z_dim+1], inputs_range[:3*z_dim+1],
                                      dimension, norm_method, seed)

         # Prep new input sample
         if model_style == 'ConvLSTM' or model_style == 'UNetConvLSTM':
            # shuffle everything up one time space
            for histtime in range(histlen-1):
               input_sample[0,histtime,:,:,:] = (input_sample[0,histtime+1,:,:,:])
            # Add predicted field, and fluxes for latest time field
            input_sample[0,histlen-1,:,:,:] = torch.cat( (next_field[0,:,:,:], 
                                                          Iterate_Dataset.__getitem__(fortime)[0][-1,3*z_dim+1:,:,:]),
                                                          axis=0)
         elif dimension == '2d':
            # shuffle everything up one time space
            for histtime in range(histlen-1):
               input_sample[0,histtime*(no_phys_in_channels):(histtime+1)*(no_phys_in_channels),:,:] = \
                                 input_sample[0,(histtime+1)*(no_phys_in_channels):(histtime+2)*(no_phys_in_channels),:,:]
            # Add predicted field, and fluxes for latest time field
            input_sample[0,(histlen-1)*(no_phys_in_channels):,:,:] = \
                            torch.cat( (next_field[0,:,:,:],
                                        Iterate_Dataset.__getitem__(fortime)[0][(histlen-1)*no_phys_in_channels+3*z_dim+1:,:,:]),
                                        axis=0)
         elif dimension == '3d':
            for histtime in range(histlen-1):
               input_sample[0,histtime*no_phys_in_channels:(histtime+1)*no_phys_in_channels,:,:,:] = \
                                 input_sample[0,(histtime+1)*no_phys_in_channels:(histtime+2)*no_phys_in_channels,:,:,:]
            input_sample[0,(histlen-1)*no_phys_in_channels:,:,:] = \
                            torch.cat( (next_field,
                                        Iterate_Dataset.__getitem__(fortime)[0][(histlen-1)*no_phys_in_channels+3*z_dim+1:,:,:,:]),
                                        axis=0)
   print('all done with iterating...')

   ## Denormalise MITgcm Data 
   MITgcm_data          = rr.RF_DeNormalise(MITgcm_data, inputs_mean[:3*z_dim+1], inputs_std[:3*z_dim+1], inputs_range[:3*z_dim+1],
                                            dimension, norm_method, seed) 

   print(MITgcm_data.shape)
 
   # Set up netcdf files to write to
   nc_filename = '../../../Channel_nn_Outputs/'+model_name+'/ITERATED_FORECAST/'+ \
                  model_name+'_'+str(no_epochs)+'epochs_'+iterate_method+'_smth'+str(iterate_smooth)+'stps'+str(smooth_steps)  \
                  +'_Forlen'+str(for_len)+'.nc'
   nc_file = nc4.Dataset(nc_filename,'w', format='NETCDF4') #'w' stands for write
   # Create Dimensions
   print(for_len/for_subsample)
   nc_file.createDimension('T', np.ceil(for_len/for_subsample))
   nc_file.createDimension('Tm', np.ceil(for_len/for_subsample)-histlen+1)
   nc_file.createDimension('Z', da_Z.values.shape[0])
   nc_file.createDimension('Y', y_dim_used)
   nc_file.createDimension('X', da_X.values.shape[0])

   # Create variables
   nc_T = nc_file.createVariable('T', 'i4', 'T')
   print(nc_T.shape)
   nc_Tm = nc_file.createVariable('Tm', 'i4', 'Tm')
   nc_Z = nc_file.createVariable('Z', 'i4', 'Z')
   nc_Y = nc_file.createVariable('Y', 'i4', 'Y')  
   nc_X = nc_file.createVariable('X', 'i4', 'X')

   nc_TrueTemp   = nc_file.createVariable( 'True_Temp'  , 'f4', ('T', 'Z', 'Y', 'X') )
   nc_TrueU      = nc_file.createVariable( 'True_U'     , 'f4', ('T', 'Z', 'Y', 'X') )
   nc_TrueV      = nc_file.createVariable( 'True_V'     , 'f4', ('T', 'Z', 'Y', 'X') )
   nc_TrueEta    = nc_file.createVariable( 'True_Eta'   , 'f4', ('T', 'Y', 'X')      )

   nc_TrueTempInc = nc_file.createVariable( 'Tr_TInc'  , 'f4', ('Tm', 'Z', 'Y', 'X') )
   nc_TrueUInc    = nc_file.createVariable( 'Tr_UInc'     , 'f4', ('Tm', 'Z', 'Y', 'X') )
   nc_TrueVInc    = nc_file.createVariable( 'Tr_VInc'     , 'f4', ('Tm', 'Z', 'Y', 'X') )
   nc_TrueEtaInc  = nc_file.createVariable( 'Tr_EInc'   , 'f4', ('Tm', 'Y', 'X')      )

   nc_PredTempInc= nc_file.createVariable( 'Pr_T_Inc'  , 'f4', ('Tm', 'Z', 'Y', 'X') )
   nc_PredUInc   = nc_file.createVariable( 'Pr_U_Inc'     , 'f4', ('Tm', 'Z', 'Y', 'X') )
   nc_PredVInc   = nc_file.createVariable( 'Pr_V_Inc'     , 'f4', ('Tm', 'Z', 'Y', 'X') ) 
   nc_PredEtaInc = nc_file.createVariable( 'Pr_E_Inc'   , 'f4', ('Tm', 'Y', 'X')      )

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
   print(np.ceil(for_len/for_subsample))
   print(nc_T.shape)
   print(np.arange(5))
   print(np.arange(np.ceil(for_len/for_subsample)).shape)
   nc_T[:] = np.arange(np.ceil(for_len/for_subsample))
   nc_Z[:] = da_Z.values
   nc_Y[:] = da_Y.values[:y_dim_used]
   nc_X[:] = da_X.values

   # Fill variables
   if dimension == '2d':
      nc_TrueTemp[:,:,:,:] = MITgcm_data[:,0:z_dim,:,:] 
      nc_TrueU[:,:,:,:]    = MITgcm_data[:,1*z_dim:2*z_dim,:,:]
      nc_TrueV[:,:,:,:]    = MITgcm_data[:,2*z_dim:3*z_dim,:,:]
      nc_TrueEta[:,:,:]    = MITgcm_data[:,3*z_dim,:,:]

      nc_PredTempInc[:,:,:,:] = predicted_increments[:,0:z_dim,:,:]
      nc_PredUInc[:,:,:,:]    = predicted_increments[:,1*z_dim:2*z_dim,:,:]
      nc_PredVInc[:,:,:,:]    = predicted_increments[:,2*z_dim:3*z_dim,:,:]
      nc_PredEtaInc[:,:,:]    = predicted_increments[:,3*z_dim,:,:]
      
      nc_PredTempFld[:,:,:,:] = out_iterated_fields[:,0:z_dim,:,:]
      nc_PredUFld[:,:,:,:]    = out_iterated_fields[:,1*z_dim:2*z_dim,:,:]
      nc_PredVFld[:,:,:,:]    = out_iterated_fields[:,2*z_dim:3*z_dim,:,:]
      nc_PredEtaFld[:,:,:]    = out_iterated_fields[:,3*z_dim,:,:]
      
      #nc_TrueTempInc[:,:,:,:] = MITgcm_data[histlen:,0:z_dim,:,:] - MITgcm_data[histlen-1:-1,0:z_dim,:,:] 
      #nc_TrueUInc[:,:,:,:]    = MITgcm_data[histlen:,1*z_dim:2*z_dim,:,:] - MITgcm_data[histlen-1:-1,1*z_dim:2*z_dim,:,:]
      #nc_TrueVInc[:,:,:,:]    = MITgcm_data[histlen:,2*z_dim:3*z_dim,:,:] - MITgcm_data[histlen-1:-1,2*z_dim:3*z_dim,:,:]
      #nc_TrueEtaInc[:,:,:]    = MITgcm_data[histlen:,3*z_dim,:,:] - MITgcm_data[histlen-1:-1,3*z_dim,:,:]
      nc_TrueTempInc[:,:,:,:] = true_increments[:,0:z_dim,:,:]
      nc_TrueUInc[:,:,:,:]    = true_increments[:,1*z_dim:2*z_dim,:,:]
      nc_TrueVInc[:,:,:,:]    = true_increments[:,2*z_dim:3*z_dim,:,:]
      nc_TrueEtaInc[:,:,:]    = true_increments[:,3*z_dim,:,:]

      nc_TempErrors[:,:,:,:] = out_iterated_fields[:,0:z_dim,:,:] - MITgcm_data[:,0:z_dim,:,:]
      nc_UErrors[:,:,:,:]    = out_iterated_fields[:,1*z_dim:2*z_dim,:,:] - MITgcm_data[:,1*z_dim:2*z_dim,:,:]
      nc_VErrors[:,:,:,:]    = out_iterated_fields[:,2*z_dim:3*z_dim,:,:] - MITgcm_data[:,2*z_dim:3*z_dim,:,:]
      nc_EtaErrors[:,:,:]    = out_iterated_fields[:,3*z_dim,:,:] - MITgcm_data[:,3*z_dim,:,:]

      nc_TempMask[:,:,:] = out_masks[0:z_dim,:,:]
      nc_UMask[:,:,:]    = out_masks[1*z_dim:2*z_dim,:,:]
      nc_VMask[:,:,:]    = out_masks[2*z_dim:3*z_dim,:,:]
      nc_EtaMask[:,:]    = out_masks[3*z_dim,:,:]

   elif dimension == '3d':
      nc_TrueTemp[:,:,:,:] = MITgcm_data[:,0,:,:,:] 
      nc_TrueU[:,:,:,:]    = MITgcm_data[:,1,:,:,:]
      nc_TrueV[:,:,:,:]    = MITgcm_data[:,2,:,:,:]
      nc_TrueEta[:,:,:]    = MITgcm_data[:,3,0,:,:]

      nc_PredTempFld[:,:,:,:] = out_iterated_fields[:,0,:,:,:]
      nc_PredUFld[:,:,:,:]    = out_iterated_fields[:,1,:,:,:]
      nc_PredVFld[:,:,:,:]    = out_iterated_fields[:,2,:,:,:]
      nc_PredEtaFld[:,:,:]    = out_iterated_fields[:,3,0,:,:]
      
      nc_TempErrors[:,:,:,:] = out_iterated_fields[:,0,:,:,:] - MITgcm_data[:,0,:,:,:]
      nc_UErrors[:,:,:,:]    = out_iterated_fields[:,1,:,:,:] - MITgcm_data[:,1,:,:,:]
      nc_VErrors[:,:,:,:]    = out_iterated_fields[:,2,:,:,:] - MITgcm_data[:,2,:,:,:]
      nc_EtaErrors[:,:,:]    = out_iterated_fields[:,3,0,:,:] - MITgcm_data[:,3,0,:,:]

      nc_TempMask[:,:,:] = out_masks[0,:,:,:]
      nc_UMask[:,:,:]    = out_masks[1,:,:,:]
      nc_VMask[:,:,:]    = out_masks[2,:,:,:]
      nc_EtaMask[:,:]    = out_masks[3,0,:,:]

