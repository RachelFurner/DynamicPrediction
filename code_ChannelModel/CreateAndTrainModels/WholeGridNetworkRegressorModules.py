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
import pickle

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

def my_loss(output, target):
   loss = torch.mean( (output - target)**2 )
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

       RF_batch_no = 0

       h.train(True)

       for input_batch, target_batch, extrafluxes_batch, masks, out_masks in train_loader:

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
                       predicted_batch = h( torch.cat((input_batch, masks), dim=channel_dim) ) * out_masks.to(device, non_blocking=True)
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
                                                       h( torch.cat((input_batch, masks), dim=channel_dim) ) * out_masks.to(device, non_blocking=True)
                       for hist_time in range(histlen-1):
                          input_batch[:, hist_time*no_phys_in_channels:(hist_time+1)*no_phys_in_channels, :, :] =   \
                                    input_batch[:, (hist_time+1)*no_phys_in_channels:(hist_time+2)*no_phys_in_channels, :, :]
                       input_batch[:, (histlen-1)*no_phys_in_channels:, :, :] =  \
                              torch.cat( ( (input_batch[:,(histlen-1)*no_phys_in_channels:histlen*no_phys_in_channels-2,:,:].to(device, non_blocking=True)
                                            + predicted_batch[:,roll_time*no_out_channels:(roll_time+1)*no_out_channels,:,:]),
                                          extrafluxes_batch[:,roll_time*2:roll_time*2+2,:,:].to(device, non_blocking=True) ), dim=1 )
      	         
                 # Calculate and update loss values
                 
                 loss = my_loss(predicted_batch, target_batch).double()
                 losses['train'][-1] = losses['train'][-1] + loss.item() * input_batch.shape[0]
                 torch.cuda.empty_cache()
      
                 if model_style == 'ConvLSTM' or model_style == 'UNetConvLSTM':
                    if dimension == '2d': 
                       losses['train_Temp'][-1] = losses['train_Temp'][-1] + \
                                                  my_loss( predicted_batch[:,:z_dim,:,:],
      	                			    target_batch[:,:z_dim,:,:],
                                                  ).item() * input_batch.shape[0]
                       losses['train_U'][-1] = losses['train_U'][-1] + \
                                               my_loss( predicted_batch[:,z_dim:2*z_dim,:,:],
                                               target_batch[:,z_dim:2*z_dim,:,:],
                                               ).item() * input_batch.shape[0]
                       losses['train_V'][-1] = losses['train_V'][-1] + \
                                               my_loss( predicted_batch[:,2*z_dim:3*z_dim,:,:],
                                               target_batch[:,2*z_dim:3*z_dim,:,:],
                                               ).item() * input_batch.shape[0]
                       losses['train_Eta'][-1] = losses['train_Eta'][-1] + \
                                                 my_loss( predicted_batch[:,3*z_dim,:,:],
                                                 target_batch[:,3*z_dim,:,:],
                                                 ).item() * input_batch.shape[0]
                 else:  
                    if dimension == '2d': 
                       losses['train_Temp'][-1] = losses['train_Temp'][-1] + \
                                                  my_loss( predicted_batch[:,:z_dim,:,:],
                                                  target_batch[:,:z_dim,:,:],
                                                  ).item() * input_batch.shape[0]
                       losses['train_U'][-1] = losses['train_U'][-1] + \
                                               my_loss( predicted_batch[:,z_dim:2*z_dim,:,:],
                                               target_batch[:,z_dim:2*z_dim,:,:],
                                               ).item() * input_batch.shape[0]
                       losses['train_V'][-1] = losses['train_V'][-1] + \
                                               my_loss( predicted_batch[:,2*z_dim:3*z_dim,:,:],
                                               target_batch[:,2*z_dim:3*z_dim,:,:],
                                               ).item() * input_batch.shape[0]
                       losses['train_Eta'][-1] = losses['train_Eta'][-1] + \
                                                 my_loss( predicted_batch[:,3*z_dim,:,:],
                                                 target_batch[:,3*z_dim,:,:],
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
           del extrafluxes_batch
           del masks
           del out_masks
           gc.collect()
           torch.cuda.empty_cache()
      
           RF_batch_no = RF_batch_no + 1

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
 
       if wandb:
          wandb.log({"train loss" : losses['train'][-1]/no_tr_samples})
       losses['train'][-1] = losses['train'][-1] / no_tr_samples
       losses['train_Temp'][-1] = losses['train_Temp'][-1] / no_tr_samples
       losses['train_U'][-1] = losses['train_U'][-1] / no_tr_samples
       losses['train_V'][-1] = losses['train_V'][-1] / no_tr_samples
       losses['train_Eta'][-1] = losses['train_Eta'][-1] / no_tr_samples
        
       logging.info('epoch {}, training loss {}'.format(epoch, losses['train'][-1])+'\n')

       #### Validation ######
   
       RF_batch_no = 0
       
       h.train(False)

       with torch.no_grad():
          for input_batch, target_batch, extrafluxes_batch, masks, out_masks in val_loader:
   
              target_batch = target_batch.to(device, non_blocking=True)
              # get prediction from the model, given the inputs
              if rolllen == 1:
                 if land == 'ExcLand':
                    predicted_batch = h( input_batch ) * out_masks.to(device, non_blocking=True)
                 else:
                    predicted_batch = h( torch.cat((input_batch, masks), dim=channel_dim) ) * out_masks.to(device, non_blocking=True)
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
                                                     h( torch.cat((input_batch, masks), dim=channel_dim) ) * out_masks.to(device, non_blocking=True)
                    for hist_time in range(histlen-1):
                       input_batch[:, hist_time*no_phys_in_channels:(hist_time+1)*no_phys_in_channels, :, :] =   \
                                 input_batch[:, (hist_time+1)*no_phys_in_channels:(hist_time+2)*no_phys_in_channels, :, :]
                    input_batch[:, (histlen-1)*no_phys_in_channels:, :, :] = \
                              torch.cat( ( (input_batch[:,(histlen-1)*no_phys_in_channels:histlen*no_phys_in_channels-2,:,:].to(device, non_blocking=True)
                                            + predicted_batch[:,roll_time*no_out_channels:(roll_time+1)*no_out_channels,:,:]),
                                          extrafluxes_batch[:,roll_time*2:roll_time*2+2,:,:].to(device, non_blocking=True) ), dim=1 )
          
              # get loss for the predicted_batch output
              losses['val'][-1] = losses['val'][-1] +  \
                                  my_loss( predicted_batch, target_batch ).item() * input_batch.shape[0]
   
              del input_batch
              del target_batch
              del predicted_batch
              gc.collect()
              torch.cuda.empty_cache()
 
              RF_batch_no = RF_batch_no + 1
   
          if wandb:
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
  
   logging.info('End of train model')
   return(losses)

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


def PlotScatter(model_name, dimension, data_loader, h, epoch, title, norm_method, channel_dim, mean_std_file, no_out_channels,
                no_phys_in_channels, z_dim, land, seed):

   logging.info('Making scatter plots')
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

   with torch.no_grad():
      input_batch, target_batch, extrafluxes_batch, masks, out_masks = next(iter(data_loader))
      target_batch = target_batch.double()
      # Make prediction
      if isinstance(h, list):
         predicted_batch = h[0]( torch.cat( (input_batch, masks), dim=channel_dim) ).to('cpu', non_blocking=True).double() 
         for i in range(1, len(h)):
            predicted_batch = predicted_batch + h[i]( torch.cat( (input_batch, masks ), dim=channel_dim) ).to('cpu', non_blocking=True).double()
         predicted_batch = predicted_batch / len(h)
      else:
         if land == 'ExcLand':
            predicted_batch = h( input_batch ).to('cpu', non_blocking=True).double()
         else:
            predicted_batch = h( torch.cat( (input_batch, masks), dim=channel_dim )).to('cpu', non_blocking=True).double()

   # Denormalise and mask, if rolllen>1 (loss function calc over multiple steps) only take first set of predictions
   target_batch = torch.where(out_masks==1, 
                              rr.RF_DeNormalise(target_batch[:, :no_out_channels, :, :],
                                                targets_mean, targets_std, targets_range, dimension, norm_method, seed),
                              np.nan)
   predicted_batch = torch.where(out_masks==1, 
                                 rr.RF_DeNormalise(predicted_batch[:, :no_out_channels, :, :],
                                                   targets_mean, targets_std, targets_range, dimension, norm_method, seed),
                                 np.nan)

   if dimension == '2d':
      # reshape to give each variable its own dimension
      targets = np.zeros((predicted_batch.shape[0], 4, z_dim, target_batch.shape[2], target_batch.shape[3] ))
      predictions = np.zeros((predicted_batch.shape[0], 4, z_dim, target_batch.shape[2], target_batch.shape[3] ))
      for i in range(3):
         targets[:,i,:,:,:]  = target_batch[:,i*z_dim:(i+1)*z_dim,:,:] 
         predictions[:,i,:,:,:] = predicted_batch[:,i*z_dim:(i+1)*z_dim,:,:] 
      targets[:,3,:,:,:] = target_batch[:,3*z_dim:4*z_dim,:,:]
      predictions[:,3,:,:,:] = predicted_batch[:,3*z_dim:4*z_dim,:,:]

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
   ax_temp.set_title('Temperature ('+u'\xb0'+'C)')
   ax_temp.set_xlim( min( np.nanmin(targets[:,0,:,:,:]), np.nanmin(predictions[:,0,:,:,:]) ),
                     max( np.nanmax(targets[:,0,:,:,:]), np.nanmax(predictions[:,0,:,:,:]) ) )
   ax_temp.set_ylim( min( np.nanmin(targets[:,0,:,:,:]), np.nanmin(predictions[:,0,:,:,:]) ),
                     max( np.nanmax(targets[:,0,:,:,:]), np.nanmax(predictions[:,0,:,:,:]) ) )
   ax_temp.plot([bottom, top], [bottom, top], 'k--', lw=1)#, color='black')

   ax_U.set_xlabel('Truth')
   ax_U.set_ylabel('Predictions')
   ax_U.set_title('East-West Velocity (m/s)')
   ax_U.set_xlim( min( np.nanmin(targets[:,1,:,:,:]), np.nanmin(predictions[:,1,:,:,:]) ),
                  max( np.nanmax(targets[:,1,:,:,:]), np.nanmax(predictions[:,1,:,:,:]) ) )
   ax_U.set_ylim( min( np.nanmin(targets[:,1,:,:,:]), np.nanmin(predictions[:,1,:,:,:]) ),
                  max( np.nanmax(targets[:,1,:,:,:]), np.nanmax(predictions[:,1,:,:,:]) ) )
   ax_U.plot([bottom, top], [bottom, top], 'k--', lw=1)#, color='black')

   ax_V.set_xlabel('Truth')
   ax_V.set_ylabel('Predictions')
   ax_V.set_title('North-South Velocity (m/s)')
   ax_V.set_xlim( min( np.nanmin(targets[:,2,:,:,:]), np.nanmin(predictions[:,2,:,:,:]) ),
                  max( np.nanmax(targets[:,2,:,:,:]), np.nanmax(predictions[:,2,:,:,:]) ) )
   ax_V.set_ylim( min( np.nanmin(targets[:,2,:,:,:]), np.nanmin(predictions[:,2,:,:,:]) ),
                  max( np.nanmax(targets[:,2,:,:,:]), np.nanmax(predictions[:,2,:,:,:]) ) )
   ax_V.plot([bottom, top], [bottom, top], 'k--', lw=1)#, color='black')

   ax_Eta.set_xlabel('Truth')
   ax_Eta.set_ylabel('Predictions')
   ax_Eta.set_title('Sea Surface Height (m)')
   ax_Eta.set_xlim( min( np.nanmin(targets[:,3,:,:,:]), np.nanmin(predictions[:,3,:,:,:]) ),
                    max( np.nanmax(targets[:,3,:,:,:]), np.nanmax(predictions[:,3,:,:,:]) ) )
   ax_Eta.set_ylim( min( np.nanmin(targets[:,3,:,:,:]), np.nanmin(predictions[:,3,:,:,:]) ),
                    max( np.nanmax(targets[:,3,:,:,:]), np.nanmax(predictions[:,3,:,:,:]) ) )
   ax_Eta.plot([bottom, top], [bottom, top], 'k--', lw=1)#, color='black')

   plt.tight_layout()

   plt.savefig('../../../Channel_nn_Outputs/'+model_name+'/TRAINING_PLOTS/'+
               model_name+'_scatter_epoch'+str(epoch).rjust(3,'0')+'_'+title+'.png',
               bbox_inches = 'tight', pad_inches = 0.1)
   plt.close()


def plot_training_output(model_name, start_epoch, total_epochs, plot_freq, losses):
   plot_dir = '../../../Channel_nn_Outputs/'+model_name+'/TRAINING_PLOTS/'
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
 
def OutputStats(model_name, model_style, MITgcm_filename, data_loader, h, no_epochs, y_dim_used, dimension, 
                histlen, land, file_append, norm_method, channel_dim, mean_std_file, no_phys_in_channels,
                no_out_channels, MITgcm_stats_filename, seed):
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
  
   no_samples = 50   # We calculate stats over the entire dataset, but only output a subset of the samples based on no_samples
   z_dim = da_Z.shape[0]

   RF_batch_no = 0
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
      for input_batch, target_tend_batch, extrafluxes_batch, masks, out_masks in data_loader:
          target_tend_batch = target_tend_batch.cpu().detach().numpy()

          # get prediction from the model, given the inputs
          if isinstance(h, list):
             predicted_tend_batch = h[0]( torch.cat( (input_batch, masks), dim=channel_dim) ) 
             for i in range(1, len(h)):
                predicted_tend_batch = predicted_tend_batch + h[i]( torch.cat( (input_batch, masks ), dim=channel_dim) )
             predicted_tend_batch = predicted_tend_batch / len(h)
          else:
             if land == 'ExcLand':
                predicted_tend_batch = h( input_batch ) 
             else:
                predicted_tend_batch = h( torch.cat( (input_batch, masks), dim=channel_dim) ) 
          predicted_tend_batch = predicted_tend_batch.cpu().detach().numpy() 

          # Denormalise, if rolllen>1 (loss function calc over multiple steps) only take first step
          target_tend_batch = rr.RF_DeNormalise(target_tend_batch[:, :no_out_channels, :, :], targets_mean, targets_std, targets_range,
                                                dimension, norm_method, seed)
          predicted_tend_batch = rr.RF_DeNormalise(predicted_tend_batch[:, :no_out_channels, :, :], targets_mean, targets_std, targets_range,
                                                   dimension, norm_method, seed)

          # Convert from increments to fields for outputting
          input_batch = input_batch#.cpu().detach().numpy()

          if model_style == 'ConvLSTM' or model_style == 'UNetConvLSTM':
             input_batch[:,-1,:3*z_dim+1,:,:] = rr.RF_DeNormalise( input_batch[:,-1,:3*z_dim+1,:,:],
                                                                   inputs_mean[:3*z_dim+1], inputs_std[:3*z_dim+1], inputs_range[:3*z_dim+1],
                                                                   dimension, norm_method, seed )
             target_fld_batch = input_batch[:,-1,:3*z_dim+1,:,:] + target_tend_batch
             predicted_fld_batch = input_batch[:,-1,:3*z_dim+1,:,:] + predicted_tend_batch
          elif dimension == '2d':
             input_batch[:,(histlen-1)*no_phys_in_channels:(histlen-1)*no_phys_in_channels+3*z_dim+1,:,:] = \
                                    rr.RF_DeNormalise( input_batch[:,(histlen-1)*no_phys_in_channels:(histlen-1)*no_phys_in_channels+3*z_dim+1,:,:],
                                                       inputs_mean[:3*z_dim+1], inputs_std[:3*z_dim+1], inputs_range[:3*z_dim+1],
                                                       dimension, norm_method, seed )
             target_fld_batch = ( input_batch[:,(histlen-1)*no_phys_in_channels:(histlen-1)*no_phys_in_channels+3*z_dim+1,:,:] 
                                  + target_tend_batch )
             predicted_fld_batch = ( input_batch[:,(histlen-1)*no_phys_in_channels:(histlen-1)*no_phys_in_channels+3*z_dim+1,:,:]
                                     + predicted_tend_batch )
          elif dimension == '3d':
             input_batch[:,(histlen-1)*no_phys_in_channels:(histlen-1)*no_phys_in_channels+3*zdim+1,:,:,:] = \
                                      rr.RF_DeNormalise( input_batch[:,(histlen-1)*no_phys_in_channels:(histlen-1)*no_phys_in_channels+3*zdim+1,:,:,:],
                                                         inputs_mean[:3*z_dim+1], inputs_std[:3*z_dim+1], inputs_range[:3*z_dim+1],
                                                         dimension, norm_method, seed )
             target_fld_batch = ( input_batch[:,(histlen-1)*no_phys_in_channels:(histlen-1)*no_phys_in_channels+3*zdim+1,:,:,:]
                                  + target_tend_batch )
             predicted_fld_batch = ( input_batch[:,(histlen-1)*no_phys_in_channels:(histlen-1)*no_phys_in_channels+3*zdim+1,:,:,:]
                                     + predicted_tend_batch )

          # Add summed error of this batch
          if RF_batch_no==0:
             targets_tend     = target_tend_batch
             predictions_tend = predicted_tend_batch 
             targets_fld      = target_fld_batch
             predictions_fld  = predicted_fld_batch 
             summed_sq_error = torch.sum( np.square(predicted_fld_batch-target_fld_batch), dim=0 ) 
          else:
             if count < no_samples:
                targets_tend     = np.concatenate( ( targets_tend, target_tend_batch ), axis=0)
                predictions_tend = np.concatenate( ( predictions_tend, predicted_tend_batch ), axis=0)
                targets_fld      = np.concatenate( ( targets_fld, target_fld_batch ), axis=0)
                predictions_fld  = np.concatenate( ( predictions_fld, predicted_fld_batch ), axis=0)
             summed_sq_error = summed_sq_error + torch.sum( np.square(predicted_fld_batch-target_fld_batch), dim=0 )

          RF_batch_no = RF_batch_no+1
          count = count + input_batch.shape[0]

   no_samples = min(no_samples, count) # Reduce no_samples if this is bigger than total samples!
  
   rms_error = np.sqrt( np.divide(summed_sq_error, count) )
   # mask rms error (for ease of viewing plots), leave for other variables so we can see what's happening
   rms_error = np.where( out_masks[0]==1, rms_error, np.nan)

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

   # Fill variables
   nc_T[:] = np.arange(no_samples)
   nc_Z[:] = da_Z.values
   nc_Y[:] = da_Y.values[:y_dim_used]
   nc_X[:] = da_X.values

   if dimension=='2d':
      nc_TrueTemp[:,:,:,:] = targets_fld[:no_samples,0:z_dim,:,:] 
      nc_TrueU[:,:,:,:]    = targets_fld[:no_samples,1*z_dim:2*z_dim,:,:]
      nc_TrueV[:,:,:,:]    = targets_fld[:no_samples,2*z_dim:3*z_dim,:,:]
      nc_TrueEta[:,:,:]    = targets_fld[:no_samples,3*z_dim,:,:]

      nc_TrueTempTend[:,:,:,:] = targets_tend[:no_samples,0:z_dim,:,:] 
      nc_TrueUTend[:,:,:,:]    = targets_tend[:no_samples,1*z_dim:2*z_dim,:,:]
      nc_TrueVTend[:,:,:,:]    = targets_tend[:no_samples,2*z_dim:3*z_dim,:,:]
      nc_TrueEtaTend[:,:,:]    = targets_tend[:no_samples,3*z_dim,:,:]

      nc_TempMask[:,:,:] = out_masks[0,0:z_dim,:,:]
      nc_UMask[:,:,:]    = out_masks[0,1*z_dim:2*z_dim,:,:]
      nc_VMask[:,:,:]    = out_masks[0,2*z_dim:3*z_dim,:,:]
      nc_EtaMask[:,:]    = out_masks[0,3*z_dim,:,:]

      nc_PredTemp[:,:,:,:] = predictions_fld[:no_samples,0:z_dim,:,:]
      nc_PredU[:,:,:,:]    = predictions_fld[:no_samples,1*z_dim:2*z_dim,:,:]
      nc_PredV[:,:,:,:]    = predictions_fld[:no_samples,2*z_dim:3*z_dim,:,:]
      nc_PredEta[:,:,:]    = predictions_fld[:no_samples,3*z_dim,:,:]
 
      nc_PredTempTend[:,:,:,:] = predictions_tend[:no_samples,0:z_dim,:,:]
      nc_PredUTend[:,:,:,:]    = predictions_tend[:no_samples,1*z_dim:2*z_dim,:,:]
      nc_PredVTend[:,:,:,:]    = predictions_tend[:no_samples,2*z_dim:3*z_dim,:,:]
      nc_PredEtaTend[:,:,:]    = predictions_tend[:no_samples,3*z_dim,:,:]
 
      nc_TempErrors[:,:,:,:] = ( predictions_fld[:no_samples,0:z_dim,:,:]
                                 - targets_fld[:no_samples,0:z_dim,:,:] )
      nc_UErrors[:,:,:,:]    = ( predictions_fld[:no_samples,1*z_dim:2*z_dim,:,:]
                                 - targets_fld[:no_samples,1*z_dim:2*z_dim,:,:] )
      nc_VErrors[:,:,:,:]    = ( predictions_fld[:no_samples,2*z_dim:3*z_dim,:,:] 
                                 - targets_fld[:no_samples,2*z_dim:3*z_dim,:,:] )
      nc_EtaErrors[:,:,:]    = ( predictions_fld[:no_samples,3*z_dim,:,:] 
                                 - targets_fld[:no_samples,3*z_dim,:,:] )

      nc_TempErrorsTend[:,:,:,:] = ( predictions_tend[:no_samples,0:z_dim,:,:]
                                     - targets_tend[:no_samples,0:z_dim,:,:] )
      nc_UErrorsTend[:,:,:,:]    = ( predictions_tend[:no_samples,1*z_dim:2*z_dim,:,:]
                                     - targets_tend[:no_samples,1*z_dim:2*z_dim,:,:] )
      nc_VErrorsTend[:,:,:,:]    = ( predictions_tend[:no_samples,2*z_dim:3*z_dim,:,:] 
                                     - targets_tend[:no_samples,2*z_dim:3*z_dim,:,:] )
      nc_EtaErrorsTend[:,:,:]    = ( predictions_tend[:no_samples,3*z_dim,:,:] 
                                     - targets_tend[:no_samples,3*z_dim,:,:] )

      nc_TempRMS[:,:,:] = rms_error[0:z_dim,:,:]
      nc_U_RMS[:,:,:]   = rms_error[1*z_dim:2*z_dim,:,:] 
      nc_V_RMS[:,:,:]   = rms_error[2*z_dim:3*z_dim,:,:]
      nc_EtaRMS[:,:]    = rms_error[3*z_dim,:,:]

      # interp MITgcm Std values onto T grid for use in scaling
      MITgcm_StdUVel = ( MITgcm_stats_ds['StdUVel'].values[:,:,:1]+MITgcm_stats_ds['StdUVel'].values[:,:,:-1] ) / 2.
      MITgcm_StdVVel = ( MITgcm_stats_ds['StdVVel'].values[:,1:,:]+MITgcm_stats_ds['StdVVel'].values[:,:-1,:] ) / 2.

      nc_Scaled_TempRMS[:,:,:] = rms_error[0:z_dim,:,:] / MITgcm_stats_ds['StdTemp'].values
      nc_Scaled_U_RMS[:,:,:]   = rms_error[1*z_dim:2*z_dim,:,:] / MITgcm_StdUVel
      nc_Scaled_V_RMS[:,:,:]   = rms_error[2*z_dim:3*z_dim,:,:] / MITgcm_StdVVel
      nc_Scaled_EtaRMS[:,:]    = rms_error[3*z_dim,:,:] / MITgcm_stats_ds['StdEta'].values

   elif dimension=='3d':
      nc_TrueTemp[:,:,:,:] = targets_fld[:no_samples,0,:,:,:] 
      nc_TrueU[:,:,:,:]    = targets_fld[:no_samples,1,:,:,:]
      nc_TrueV[:,:,:,:]    = targets_fld[:no_samples,2,:,:,:]
      nc_TrueEta[:,:,:]    = targets_fld[:no_samples,3,0,:,:]

      nc_TrueTempTend[:,:,:,:] = targets_tend[:no_samples,0,:,:,:] 
      nc_TrueUTend[:,:,:,:]    = targets_tend[:no_samples,1,:,:,:]
      nc_TrueVTend[:,:,:,:]    = targets_tend[:no_samples,2,:,:,:]
      nc_TrueEtaTend[:,:,:]    = targets_tend[:no_samples,3,0,:,:]

      nc_TempMask[:,:,:] = out_masks[0,0,:,:,:]
      nc_UMask[:,:,:]    = out_masks[0,1,:,:,:]
      nc_VMask[:,:,:]    = out_masks[0,2,:,:,:]
      nc_EtaMask[:,:]    = out_masks[0,3,0,:,:]

      nc_PredTemp[:,:,:,:] = predictions_fld[:no_samples,0,:,:,:]
      nc_PredU[:,:,:,:]    = predictions_fld[:no_samples,1,:,:,:]
      nc_PredV[:,:,:,:]    = predictions_fld[:no_samples,2,:,:,:]
      nc_PredEta[:,:,:]    = predictions_fld[:no_samples,3,0,:,:]
 
      nc_PredTempTend[:,:,:,:] = predictions_tend[:no_samples,0,:,:,:]
      nc_PredUTend[:,:,:,:]    = predictions_tend[:no_samples,1,:,:,:]
      nc_PredVTend[:,:,:,:]    = predictions_tend[:no_samples,2,:,:,:]
      nc_PredEtaTend[:,:,:]    = predictions_tend[:no_samples,3,0,:,:]
 
      nc_TempErrors[:,:,:,:] = ( predictions_fld[:no_samples,0,:,:,:]
                                 - targets_fld[:no_samples,0,:,:,:] )
      nc_UErrors[:,:,:,:]    = ( predictions_fld[:no_samples,1,:,:,:]
                                 - targets_fld[:no_samples,1,:,:,:] )
      nc_VErrors[:,:,:,:]    = ( predictions_fld[:no_samples,2,:,:,:] 
                                 - targets_fld[:no_samples,2,:,:,:] )
      nc_EtaErrors[:,:,:]    = ( predictions_fld[:no_samples,3,0,:,:] 
                                 - targets_fld[:no_samples,3,0,:,:] )

      nc_TempRMS[:,:,:] = rms_error[0,:,:,:]
      nc_U_RMS[:,:,:]   = rms_error[1,:,:,:] 
      nc_V_RMS[:,:,:]   = rms_error[2,:,:,:]
      nc_EtaRMS[:,:]    = rms_error[3,0,:,:]


def IterativelyPredict(model_name, model_style, MITgcm_filename, Iterate_Dataset, h, start, for_len, no_epochs, y_dim_used,
                       land, dimension, histlen, landvalues, iterate_method, iterate_smooth, smooth_steps, norm_method,
                       channel_dim, mean_std_file, for_subsample, no_phys_in_channels, no_out_channels, seed):
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
   input_sample, target_sample, extra_fluxes_sample, masks, out_masks = Iterate_Dataset.__getitem__(start)

   # Give extra dimension at front (number of samples - here 1)
   input_sample = input_sample.unsqueeze(0).float()
   masks = masks.unsqueeze(0).float()
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
         if isinstance(h, list):
            predicted = h[0]( torch.cat((input_sample, masks), dim=channel_dim)).cpu().detach()
            for i in range(1, len(h)):
               predicted = predicted + \
                           h[i]( torch.cat((input_sample, masks), dim=channel_dim)).cpu().detach()
            predicted = predicted / len(h)
         else:
            if land == 'ExcLand':
               predicted = h( input_sample ).cpu().detach()
            else:
               predicted = h( torch.cat( (input_sample, masks), dim=channel_dim ) ).cpu().detach()

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
               next_field = iterared_fields[-1] + (3./2.+.1) * predicted - (1./2.+.1) * old_predicted
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

