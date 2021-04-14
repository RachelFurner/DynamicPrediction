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
import gc

import multiprocessing as mp
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

#add this routine for memory profiling
def sizeof_fmt(num, suffix='B'):
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)


def TimeCheck(tic, output_file, task):
   toc = time.time()
   print('Finished '+task+' at {:0.4f} seconds'.format(toc - tic)+'\n')
   output_file.write('Finished '+task+' at {:0.4f} seconds'.format(toc - tic)+'\n')

def CalcMeanStd(MeanStd_prefix, MITGCM_filename, train_end_ratio, subsample_rate, dataset_end_index, batch_size, output_file):

   no_depth_levels = 38 ## Note this is hard coded!!

   mean_std_Dataset = rr.MITGCM_Wholefield_Dataset( MITGCM_filename, 0.0, train_end_ratio, subsample_rate, dataset_end_index, transform=None)
   mean_std_loader = data.DataLoader(mean_std_Dataset, batch_size=batch_size, shuffle=False, num_workers=0)
   
   FirstPass = True 
   count = 0.
   batch_no = 0

   #------------------------------------
   # First calculate the mean and range
   #------------------------------------
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

   print('inputs_mean')
   print(inputs_mean)
   print('inputs_range')
   print(inputs_range)

   #-------------------------
   # Next calculate the std 
   #-------------------------
   count = 0.
   for batch_sample in mean_std_loader:
       
       # shape (batch_size, channels, y, x)
       input_batch  = batch_sample['input']

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

   print('inputs_std')
   print(inputs_std)

   #-------------------------

   no_channels = input_batch.shape[1]

   ## Save to file, so can be used to un-normalise when using model to predict
   mean_std_file = '../../../Channel_nn_Outputs/'+MeanStd_prefix+'MeanStd.npz'
   np.savez( mean_std_file, inputs_mean, inputs_std, inputs_range, no_channels)
   
   output_file.write('no_channels ;'+str(no_channels)+'\n')
   print('no_channels ;'+str(no_channels)+'\n')
  
   del batch_sample
   del input_batch

   return no_channels


def ReadMeanStd(MeanStd_prefix, output_file=None):
   mean_std_file = '../../../Channel_nn_Outputs/'+MeanStd_prefix+'MeanStd.npz'
   mean_std_data = np.load(mean_std_file)
   inputs_mean  = mean_std_data['arr_0']
   inputs_std   = mean_std_data['arr_1']
   inputs_range = mean_std_data['arr_2']
   no_channels  = mean_std_data['arr_3']

   if output_file:
      output_file.write('no_channels ;'+str(no_channels)+'\n')
   print('no_channels ;'+str(no_channels)+'\n')

   return(inputs_mean, inputs_std, inputs_range, no_channels)


def TrainModel(model_name, output_file, tic, TEST, no_tr_samples, no_val_samples, plot_freq, save_freq, train_loader, val_loader, h, optimizer, hyper_params, reproducible, seed_value, existing_losses, start_epoch=0):

   no_depth_levels = 38  # Hard coded...perhaps should change...?
   # Set variables to remove randomness and ensure reproducible results
   if reproducible:
      os.environ['PYTHONHASHSEED'] = str(seed_value)
      np.random.seed(seed_value)
      torch.manual_seed(seed_value)
      torch.cuda.manual_seed(seed_value)
      torch.cuda.manual_seed_all(seed_value)
      torch.backends.cudnn.enabled = False
      torch.backends.cudnn.deterministic = True
      torch.backends.cudnn.benchmark = False

   train_loss_epoch = existing_losses['train_loss_epoch']
   train_loss_epoch_Temp = existing_losses['train_loss_epoch_Temp']
   train_loss_epoch_U    = existing_losses['train_loss_epoch_U']
   train_loss_epoch_V    = existing_losses['train_loss_epoch_V']
   train_loss_epoch_Eta  = existing_losses['train_loss_epoch_Eta']
   val_loss_epoch = existing_losses['val_loss_epoch']

   first_pass = True
   torch.cuda.empty_cache()
   
   plot_dir = '../../../Channel_nn_Outputs/'+model_name+'/PLOTS/'
   if not os.path.isdir(plot_dir):
      os.system("mkdir %s" % (plot_dir))
   
   print('')
   print('###### TRAINING MODEL ######')
   print('')
   for epoch in range(start_epoch,hyper_params["num_epochs"]+start_epoch):
   
       print('epoch ; '+str(epoch))
   
       # Training
   
       train_loss_epoch_tmp = 0.0
       train_loss_epoch_Temp_tmp = 0.
       train_loss_epoch_U_tmp = 0.
       train_loss_epoch_V_tmp = 0.
       train_loss_epoch_Eta_tmp = 0.
   
       
       if ( epoch%plot_freq == 0 and epoch !=0 ) or epoch == hyper_params["num_epochs"]+start_epoch-1 :
           fig_temp = plt.figure(figsize=(9,9))
           ax_temp  = fig_temp.add_subplot(111)
           fig_U    = plt.figure(figsize=(9,9))
           ax_U     = fig_U.add_subplot(111)
           fig_V    = plt.figure(figsize=(9,9))
           ax_V     = fig_V.add_subplot(111)
           fig_Eta  = plt.figure(figsize=(9,9))
           ax_Eta   = fig_Eta.add_subplot(111)
           bottom = 0.
           top = 0.

       batch_no = 0
       for batch_sample in train_loader:
           #print('')
           #print('------------------- Training batch number : '+str(batch_no)+'-------------------')
   
           input_batch  = batch_sample['input']
           target_batch = batch_sample['output']
           #print('read in batch_sample')
           # Converting inputs and labels to Variable and send to GPU if available
           if torch.cuda.is_available():
               input_batch = Variable(input_batch.cuda().float())
               target_batch = Variable(target_batch.cuda().float())
           else:
               input_batch = Variable(inputs.float())
               target_batch = Variable(targets.float())
           #print('converted to variable and sent to GPU if using')
           
           h.train(True)
   
           # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
           optimizer.zero_grad()
           #print('set optimizer to zero grad')
       
           # get prediction from the model, given the inputs
           predicted = h(input_batch)
           #print('made prediction')
   
           if TEST and first_pass:
              print('For test run check if output is correct shape, the following two shapes should match!\n')
              print('target_batch.shape for training data: '+str(target_batch.shape)+'\n')
              print('predicted.shape for training data: '+str(predicted.shape)+'\n')
              output_file.write('For test run check if output is correct shape, the following two shapes should match!\n')
              output_file.write('target_batch.shape for training data: '+str(target_batch.shape)+'\n')
              output_file.write('predicted.shape for training data: '+str(predicted.shape)+'\n')
              
           # get loss for the predicted output
           loss = hyper_params["criterion"](predicted, target_batch)
           #print('calculated loss')
           
           # get gradients w.r.t to parameters
           loss.backward()
           #print('calculated loss gradients')
        
           # update parameters
           optimizer.step()
           #print('did optimization')
   
           if TEST and first_pass:  # Recalculate loss post optimizer step to ensure its decreased
              h.train(False)
              post_opt_predicted = h(input_batch)
              post_opt_loss = hyper_params["criterion"](post_opt_predicted, target_batch)
              print('For test run check if loss decreases over a single training step on a batch of data')
              print('loss pre optimization: ' + str(loss.item()) )
              print('loss post optimization: ' + str(post_opt_loss.item()) )
              output_file.write('For test run check if loss decreases over a single training step on a batch of data\n')
              output_file.write('loss pre optimization: ' + str(loss.item()) + '\n')
              output_file.write('loss post optimization: ' + str(post_opt_loss.item()) + '\n')
           
           train_loss_epoch_tmp += loss.item()*input_batch.shape[0]
           train_loss_epoch_Temp_tmp += hyper_params["criterion"](predicted[:,:no_depth_levels,:,:],
                                                               target_batch[:,:no_depth_levels,:,:]).item() * input_batch.shape[0]
           train_loss_epoch_U_tmp += hyper_params["criterion"](predicted[:,no_depth_levels:2*no_depth_levels,:,:],
                                                            target_batch[:,no_depth_levels:2*no_depth_levels,:,:]).item() * input_batch.shape[0]
           train_loss_epoch_V_tmp += hyper_params["criterion"](predicted[:,2*no_depth_levels:3*no_depth_levels,:,:],
                                                            target_batch[:,2*no_depth_levels:3*no_depth_levels,:,:]).item() * input_batch.shape[0]
           train_loss_epoch_Eta_tmp += hyper_params["criterion"](predicted[:,3*no_depth_levels,:,:],
                                                              target_batch[:,3*no_depth_levels,:,:]).item() * input_batch.shape[0]
           #print('Added loss to epoch tmp')
   
           if TEST: 
              print('train_loss_epoch_tmp; '+str(train_loss_epoch_tmp))
   
           if ( epoch%plot_freq == 0 and epoch !=0 ) or epoch == hyper_params["num_epochs"]+start_epoch-1 :
              if batch_no == 0:
                 target_batch_subsamp = target_batch[0,:,::2,::2].cpu().detach().numpy()
                 predicted_subsamp = predicted[0,:,::2,::2].cpu().detach().numpy()
                 ax_temp.scatter(target_batch_subsamp[:no_depth_levels,:,:],
                                 predicted_subsamp[:no_depth_levels,:,:],
                                 edgecolors=(0, 0, 0), alpha=0.15, color='blue', label='Temp')
                 ax_U.scatter(target_batch_subsamp[no_depth_levels:2*no_depth_levels,:,:],
                              predicted_subsamp[no_depth_levels:2*no_depth_levels,:,:],
                              edgecolors=(0, 0, 0), alpha=0.15, color='red', label='U')
                 ax_V.scatter(target_batch_subsamp[2*no_depth_levels:3*no_depth_levels,:,:], 
                              predicted_subsamp[2*no_depth_levels:3*no_depth_levels,:,:],
                              edgecolors=(0, 0, 0), alpha=0.15, color='orange', label='V')
                 ax_Eta.scatter(target_batch_subsamp[3*no_depth_levels,:,:], 
                                predicted_subsamp[3*no_depth_levels,:,:],
                                edgecolors=(0, 0, 0), alpha=0.15, color='purple', label='Eta')
                 bottom = min(np.amin(predicted_subsamp), np.amin(target_batch_subsamp), bottom)
                 top    = max(np.amax(predicted_subsamp), np.amax(target_batch_subsamp), top)  
   
           gc.collect()
           torch.cuda.empty_cache()
           #for name, size in sorted(((name, sys.getsizeof(value)) for name, value in locals().items()),
           #                         key= lambda x: -x[1])[:10]:
           #    print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))
           #print(torch.cuda.memory_summary(device))
           #print('')

           if first_pass:
              # Plot histogram of data, for bug fixing and checking...
              no_bins = 50
              for no_var in range(4):
                 if no_var < 3:
                    data = input_batch[:,no_var*no_depth_levels:(no_var+1)*no_depth_levels,:,:].cpu().reshape(-1)
                 else:
                    data = input_batch[:,no_var*no_depth_levels,:,:].cpu().reshape(-1)
                 fig2 = plt.figure(figsize=(10, 8))
                 plt.hist(data, bins = no_bins)
                 plt.savefig(plot_dir+'/'+model_name+'_histogram_var'+str(no_var)+'_batch'+str(batch_no)+'.png', bbox_inches = 'tight', pad_inches = 0.1)
                 plt.close()

           del input_batch
           del target_batch
           del predicted
           first_pass = False
           #print('batch complete')
           batch_no = batch_no + 1

       del batch_sample 

       tr_loss_single_epoch = train_loss_epoch_tmp / no_tr_samples
       print('epoch {}, training loss {}'.format( epoch, tr_loss_single_epoch )+'\n')
       output_file.write('epoch {}, training loss {}'.format( epoch, tr_loss_single_epoch )+'\n')
       #print('RF after printing training loss')
       train_loss_epoch.append( tr_loss_single_epoch )
       train_loss_epoch_Temp.append( train_loss_epoch_Temp_tmp / no_tr_samples )
       train_loss_epoch_U.append( train_loss_epoch_U_tmp / no_tr_samples )
       train_loss_epoch_V.append( train_loss_epoch_V_tmp / no_tr_samples )
       train_loss_epoch_Eta.append( train_loss_epoch_Eta_tmp / no_tr_samples )
  
       #print('RF before if statements')
       if TEST: 
          print('IN TEST!!!')
          print('tr_loss_single_epoch; '+str(tr_loss_single_epoch))
          print('train_loss_epoch; '+str(train_loss_epoch))
   
       if ( epoch%plot_freq == 0 and epoch !=0 ) or epoch == hyper_params["num_epochs"]+start_epoch-1 :

           ax_temp.set_xlabel('Truth')
           ax_temp.set_ylabel('Predictions')
           ax_temp.set_title('Training errors, Temperature, epoch '+str(epoch))
           ax_temp.set_xlim(bottom, top)
           ax_temp.set_ylim(bottom, top)
           ax_temp.plot([bottom, top], [bottom, top], 'k--', lw=1, color='blue')
           ax_temp.annotate('Epoch Loss: '+str(tr_loss_single_epoch), (0.15, 0.9), xycoords='figure fraction')
           plt.savefig(plot_dir+'/'+model_name+'_Temp_scatter_epoch'+str(epoch).rjust(3,'0')+'training.png',
                       bbox_inches = 'tight', pad_inches = 0.1)
           plt.close()

           ax_U.set_xlabel('Truth')
           ax_U.set_ylabel('Predictions')
           ax_U.set_title('Training errors, U vel, epoch '+str(epoch))
           ax_U.set_xlim(bottom, top)
           ax_U.set_ylim(bottom, top)
           ax_U.plot([bottom, top], [bottom, top], 'k--', lw=1, color='blue')
           ax_U.annotate('Epoch Loss: '+str(tr_loss_single_epoch), (0.15, 0.9), xycoords='figure fraction')
           plt.savefig(plot_dir+'/'+model_name+'_U_scatter_epoch'+str(epoch).rjust(3,'0')+'training.png',
                       bbox_inches = 'tight', pad_inches = 0.1)
           plt.close()

           ax_V.set_xlabel('Truth')
           ax_V.set_ylabel('Predictions')
           ax_V.set_title('Training errors, V vel, epoch '+str(epoch))
           ax_V.set_xlim(bottom, top)
           ax_V.set_ylim(bottom, top)
           ax_V.plot([bottom, top], [bottom, top], 'k--', lw=1, color='blue')
           ax_V.annotate('Epoch Loss: '+str(tr_loss_single_epoch), (0.15, 0.9), xycoords='figure fraction')
           plt.savefig(plot_dir+'/'+model_name+'_V_scatter_epoch'+str(epoch).rjust(3,'0')+'training.png',
                       bbox_inches = 'tight', pad_inches = 0.1)
           plt.close()

           ax_Eta.set_xlabel('Truth')
           ax_Eta.set_ylabel('Predictions')
           ax_Eta.set_title('Training errors, Eta, epoch '+str(epoch))
           ax_Eta.set_xlim(bottom, top)
           ax_Eta.set_ylim(bottom, top)
           ax_Eta.plot([bottom, top], [bottom, top], 'k--', lw=1, color='blue')
           ax_Eta.annotate('Epoch Loss: '+str(tr_loss_single_epoch), (0.15, 0.9), xycoords='figure fraction')
           plt.savefig(plot_dir+'/'+model_name+'_Eta_scatter_epoch'+str(epoch).rjust(3,'0')+'training.png',
                       bbox_inches = 'tight', pad_inches = 0.1)
           plt.close()
   
       # Validation
   
       val_loss_epoch_tmp = 0.0
       batch_no = 0
       
       if ( epoch%plot_freq == 0 and epoch !=0 ) or epoch == hyper_params["num_epochs"]+start_epoch-1 :
           fig_Temp = plt.figure(figsize=(9,9))
           ax_Temp  = fig_Temp.add_subplot(111)
           fig_U    = plt.figure(figsize=(9,9))
           ax_U     = fig_U.add_subplot(111)
           fig_V    = plt.figure(figsize=(9,9))
           ax_V     = fig_V.add_subplot(111)
           fig_Eta  = plt.figure(figsize=(9,9))
           ax_Eta   = fig_Eta.add_subplot(111)
           bottom = 0.
           top = 0.
   
       #print('RF in val , just about to start batch loop')
       for batch_sample in val_loader:
       #    print('')
       #    print('------------------- Validation batch number : '+str(batch_no)+'-------------------')
   
           input_batch  = batch_sample['input']
           target_batch = batch_sample['output']
       
           # Converting inputs and labels to Variable and send to GPU if available
           if torch.cuda.is_available():
               input_batch = Variable(input_batch.cuda().float())
               target_batch = Variable(target_batch.cuda().float())
           else:
               input_batch = Variable(input_batch.float())
               target_batch = Variable(target_batch.float())
           #print('converted to variable and sent to GPU if using')
           
           h.train(False)
   
           # get prediction from the model, given the inputs
           predicted = h(input_batch)
           #print('prediction made')
       
           # get loss for the predicted output
           val_loss = hyper_params["criterion"](predicted, target_batch)
   
           val_loss_epoch_tmp += val_loss.item()*input_batch.shape[0]
           #print('loss calculated and added')
   
           if TEST: 
              print('val_loss_epoch_tmp; '+str(val_loss_epoch_tmp))
  
           if ( epoch%plot_freq == 0 and epoch !=0 ) or epoch == hyper_params["num_epochs"]+start_epoch-1 :
              if batch_no == 0:
                 target_batch_subsamp = target_batch[0,:,::2,::2].cpu().detach().numpy()
                 predicted_subsamp = predicted[0,:,::2,::2].cpu().detach().numpy()
                 ax_Temp.scatter(target_batch_subsamp[:no_depth_levels,:,:],
                                 predicted_subsamp[:no_depth_levels,:,:],
                                 edgecolors=(0, 0, 0), alpha=0.15, color='blue', label='Temp')
                 ax_U.scatter(target_batch_subsamp[no_depth_levels:2*no_depth_levels,:,:],
                              predicted_subsamp[no_depth_levels:2*no_depth_levels,:,:],
                              edgecolors=(0, 0, 0), alpha=0.15, color='red', label='U')
                 ax_V.scatter(target_batch_subsamp[2*no_depth_levels:3*no_depth_levels,:,:], 
                              predicted_subsamp[2*no_depth_levels:3*no_depth_levels,:,:],
                              edgecolors=(0, 0, 0), alpha=0.15, color='orange', label='V')
                 ax_Eta.scatter(target_batch_subsamp[3*no_depth_levels,:,:], 
                                predicted_subsamp[3*no_depth_levels,:,:],
                                edgecolors=(0, 0, 0), alpha=0.15, color='purple', label='Eta')
                 bottom = min(np.amin(predicted_subsamp), np.amin(target_batch_subsamp), bottom)
                 top    = max(np.amax(predicted_subsamp), np.amax(target_batch_subsamp), top)  
   
           batch_no = batch_no + 1
           del input_batch
           del target_batch
           del predicted
           del val_loss
           gc.collect()
           torch.cuda.empty_cache()
           #print('batch complete')
           #for name, size in sorted(((name, sys.getsizeof(value)) for name, value in locals().items()),
           #                         key= lambda x: -x[1])[:10]:
           #    print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))
           #print(torch.cuda.memory_summary(device))
           #print('')
           #print('variables deleted')
   
       del batch_sample 

       val_loss_single_epoch = val_loss_epoch_tmp / no_val_samples
       print('epoch {}, validation loss {}'.format(epoch, val_loss_single_epoch)+'\n')
       output_file.write('epoch {}, validation loss {}'.format(epoch, val_loss_single_epoch)+'\n')
       val_loss_epoch.append(val_loss_single_epoch)
       #print('loss appended')
   
       if TEST: 
          print('val_loss_single_epoch; '+str(val_loss_single_epoch))
          print('val_loss_epoch; '+str(val_loss_epoch))
   
       if ( epoch%plot_freq == 0 and epoch !=0 ) or epoch == hyper_params["num_epochs"]+start_epoch-1 :

           ax_Temp.set_xlabel('Truth')
           ax_Temp.set_ylabel('Predictions')
           ax_Temp.set_title('Training errors, epoch '+str(epoch))
           ax_Temp.set_xlim(bottom, top)
           ax_Temp.set_ylim(bottom, top)
           ax_Temp.plot([bottom, top], [bottom, top], 'k--', lw=1, color='black')
           ax_Temp.annotate('Epoch Loss: '+str(tr_loss_single_epoch), (0.15, 0.9), xycoords='figure fraction')
           plt.savefig(plot_dir+'/'+model_name+'_Temp_scatter_epoch'+str(epoch).rjust(3,'0')+'val.png',
                       bbox_inches = 'tight', pad_inches = 0.1)
           plt.close()

           ax_U.set_xlabel('Truth')
           ax_U.set_ylabel('Predictions')
           ax_U.set_title('Training errors, epoch '+str(epoch))
           ax_U.set_xlim(bottom, top)
           ax_U.set_ylim(bottom, top)
           ax_U.plot([bottom, top], [bottom, top], 'k--', lw=1, color='black')
           ax_U.annotate('Epoch Loss: '+str(tr_loss_single_epoch), (0.15, 0.9), xycoords='figure fraction')
           plt.savefig(plot_dir+'/'+model_name+'_U_scatter_epoch'+str(epoch).rjust(3,'0')+'val.png',
                       bbox_inches = 'tight', pad_inches = 0.1)
           plt.close()

           print('RF a')
           ax_V.set_xlabel('Truth')
           print('RF b')
           ax_V.set_ylabel('Predictions')
           print('RF c')
           ax_V.set_title('Training errors, epoch '+str(epoch))
           print('RF d')
           ax_V.set_xlim(bottom, top)
           print('RF e')
           ax_V.set_ylim(bottom, top)
           print('RF f')
           ax_V.plot([bottom, top], [bottom, top], 'k--', lw=1, color='black')
           print('RF g')
           ax_V.annotate('Epoch Loss: '+str(tr_loss_single_epoch), (0.15, 0.9), xycoords='figure fraction')
           print('RF h')
           plt.savefig(plot_dir+'/'+model_name+'_V_scatter_epoch'+str(epoch).rjust(3,'0')+'val.png',
                       bbox_inches = 'tight', pad_inches = 0.1)
           print('RF i')
           plt.close()
           print('RF j')

           ax_Eta.set_xlabel('Truth')
           ax_Eta.set_ylabel('Predictions')
           ax_Eta.set_title('Training errors, epoch '+str(epoch))
           ax_Eta.set_xlim(bottom, top)
           ax_Eta.set_ylim(bottom, top)
           ax_Eta.plot([bottom, top], [bottom, top], 'k--', lw=1, color='black')
           ax_Eta.annotate('Epoch Loss: '+str(tr_loss_single_epoch), (0.15, 0.9), xycoords='figure fraction')
           plt.savefig(plot_dir+'/'+model_name+'_Eta_scatter_epoch'+str(epoch).rjust(3,'0')+'val.png',
                       bbox_inches = 'tight', pad_inches = 0.1)
           plt.close()

       if ( epoch%plot_freq == 0 and epoch !=0 ) or epoch == (hyper_params["num_epochs"]+start_epoch-1):
           ### SAVE IT ###
           print('save model \n')
           output_file.write('save model \n')
           pkl_filename = '../../../Channel_nn_Outputs/'+model_name+'/MODELS/'+model_name+'_epoch'+str(epoch)+'_SavedModel.pt'
           #torch.save(h.state_dict(), pkl_filename)
           torch.save({
                       'epoch': epoch,
                       'model_state_dict': h.state_dict(),
                       'optimizer_state_dict': optimizer.state_dict(),
                       'loss': loss,
                       'train_loss_epoch': train_loss_epoch,
                       'train_loss_epoch_Temp': train_loss_epoch_Temp,
                       'train_loss_epoch_U': train_loss_epoch_U,
                       'train_loss_epoch_V': train_loss_epoch_V,
                       'train_loss_epoch_Eta': train_loss_epoch_Eta,
                       'val_loss_epoch': val_loss_epoch
                       }, pkl_filename)
  
       toc = time.time()
       print('Finished epoch {} at {:0.4f} seconds'.format(epoch, toc - tic)+'\n')
       output_file.write('Finished epoch {} at {:0.4f} seconds'.format(epoch, toc - tic)+'\n')
   
   torch.cuda.empty_cache()

   # Plot training and validation loss 
   fig = plt.figure()
   ax1 = fig.add_subplot(111)
   ax1.plot(range(0, start_epoch+hyper_params["num_epochs"]), train_loss_epoch, color='black', label='Total Training Loss')
   ax1.plot(range(0          , start_epoch+hyper_params["num_epochs"]), train_loss_epoch_Temp, color='blue', label='Temperature Training Loss')
   ax1.plot(range(0          , start_epoch+hyper_params["num_epochs"]), train_loss_epoch_U, color='red', label='U Training Loss')
   ax1.plot(range(0          , start_epoch+hyper_params["num_epochs"]), train_loss_epoch_V, color='orange', label='V Training Loss')
   ax1.plot(range(0          , start_epoch+hyper_params["num_epochs"]), train_loss_epoch_Eta, color='purple', label='Eta Training Loss')
   ax1.plot(range(0          , start_epoch+hyper_params["num_epochs"]), val_loss_epoch, color='grey', label='Validation Loss')
   ax1.set_xlabel('Epochs')
   ax1.set_ylabel('Loss')
   ax1.set_yscale('log')
   #ax1.set_ylim(0.02, 0.3)
   ax1.legend()
   plt.savefig(plot_dir+'/'+model_name+'_TrainingValLossPerEpoch.png',
               bbox_inches = 'tight', pad_inches = 0.1)
   plt.close()
   
   info_filename = '../../../Channel_nn_Outputs/'+model_name+'/MODELS/'+model_name+'_info.txt'
   info_file = open(info_filename, 'w')
   np.set_printoptions(threshold=np.inf)
   info_file.write('hyper_params:    '+str(hyper_params)+'\n')
   info_file.write('\n')
   #info_file.write('Weights of the network:\n')
   #info_file.write(str(h[0].weight.data.cpu().numpy()))
  
def LoadModel(model_name, h, optimizer, saved_epoch, tr_inf, existing_losses):
   pkl_filename = '../../../Channel_nn_Outputs/'+model_name+'/MODELS/'+model_name+'_epoch'+str(saved_epoch)+'_SavedModel.pt'
   checkpoint = torch.load(pkl_filename)
   h.load_state_dict(checkpoint['model_state_dict'])
   optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
   epoch = checkpoint['epoch']
   loss = checkpoint['loss']
   existing_losses['train_loss_epoch'] = checkpoint['train_loss_epoch']
   existing_losses['train_loss_epoch_Temp'] = checkpoint['train_loss_epoch_Temp']
   existing_losses['train_loss_epoch_U']    = checkpoint['train_loss_epoch_U']
   existing_losses['train_loss_epoch_V']    = checkpoint['train_loss_epoch_V']
   existing_losses['train_loss_epoch_Eta']  = checkpoint['train_loss_epoch_Eta']
   existing_losses['val_loss_epoch'] = checkpoint['val_loss_epoch']

   if tr_inf == 'tr':
      h.train()
   elif tr_inf == 'inf':
      h.eval()

def OutputSinglePrediction(model_name, MeanStd_prefix, Train_Dataset, h, no_epochs):
   #-----------------------------------------------------------------------------------------
   # Output the predictions for a single input into a netcdf file to check it looks sensible
   #-----------------------------------------------------------------------------------------
   data_mean, data_std, data_range, no_channels = ReadMeanStd(MeanStd_prefix)
   
   #Make predictions for a single sample
   sample = Train_Dataset.__getitem__(5)
   input_sample  = sample['input'][:,:,:].unsqueeze(0)
   target_sample = sample['output'][:,:,:].unsqueeze(0)
   if torch.cuda.is_available():
       input_sample = Variable(input_sample.cuda().float())
       target_sample = Variable(target_sample.cuda().float())
       h = h.cuda()
   else:
       input_sample = Variable(input_sample.float())
       target_sample = Variable(target_sample.float())
   h.train(False)
   predicted = h(input_sample)
   predicted = predicted.data.cpu().numpy()
   target_sample = target_sample.data.cpu().numpy()
   
   # Denormalise
   target_sample = rr.RF_DeNormalise(target_sample, data_mean, data_std, data_range)
   predicted     = rr.RF_DeNormalise(predicted    , data_mean, data_std, data_range)
   
   # Set up netcdf files
   nc_filename = '../../../Channel_nn_Outputs/'+model_name+'/STATS/'+model_name+'_'+str(no_epochs)+'epochs_DummyOutput.nc'
   nc_file = nc4.Dataset(nc_filename,'w', format='NETCDF4') #'w' stands for write
   # Create Dimensions
   no_depth_levels = 38  # Hard coded...perhaps should change...?
   nc_file.createDimension('Z', no_depth_levels)
   nc_file.createDimension('Y', target_sample.shape[2])
   nc_file.createDimension('X', target_sample.shape[3])
   # Create variables
   nc_Z = nc_file.createVariable('Z', 'i4', 'Z')
   nc_Y = nc_file.createVariable('Y', 'i4', 'Y')  
   nc_X = nc_file.createVariable('X', 'i4', 'X')
   nc_TrueTemp = nc_file.createVariable('True_Temp', 'f4', ('Z', 'Y', 'X'))
   nc_PredTemp = nc_file.createVariable('Pred_Temp', 'f4', ('Z', 'Y', 'X'))
   nc_TempErrors = nc_file.createVariable('Temp_Errors', 'f4', ('Z', 'Y', 'X'))
   #nc_TrueSal  = nc_file.createVariable('True_Sal', 'f4', ('Z', 'Y', 'X'))
   #nc_PredSal  = nc_file.createVariable('Pred_Sal', 'f4', ('Z', 'Y', 'X'))
   #nc_SalErrors  = nc_file.createVariable('Sal_Errors', 'f4', ('Z', 'Y', 'X'))
   nc_TrueU = nc_file.createVariable('True_U', 'f4', ('Z', 'Y', 'X'))
   nc_PredU = nc_file.createVariable('Pred_U', 'f4', ('Z', 'Y', 'X'))
   nc_UErrors = nc_file.createVariable('U_Errors', 'f4', ('Z', 'Y', 'X'))
   nc_TrueV = nc_file.createVariable('True_V', 'f4', ('Z', 'Y', 'X'))
   nc_PredV = nc_file.createVariable('Pred_V', 'f4', ('Z', 'Y', 'X'))
   nc_VErrors = nc_file.createVariable('V_Errors', 'f4', ('Z', 'Y', 'X'))
   nc_TrueEta = nc_file.createVariable('True_Eta', 'f4', ('Y', 'X'))
   nc_PredEta = nc_file.createVariable('Pred_Eta', 'f4', ('Y', 'X'))
   nc_EtaErrors = nc_file.createVariable('Eta_Errors', 'f4', ('Y', 'X'))
   # Fill variables
   nc_Z[:] = np.arange(no_depth_levels)
   nc_Y[:] = np.arange(target_sample.shape[2])
   nc_X[:] = np.arange(target_sample.shape[3])
   nc_TrueTemp[:,:,:]   = target_sample[0,0:no_depth_levels,:,:] 
   nc_PredTemp[:,:,:]   = predicted[0,0:no_depth_levels,:,:]
   nc_TempErrors[:,:,:] = predicted[0,0:no_depth_levels,:,:] - target_sample[0,0:no_depth_levels,:,:]
   #nc_TrueSal[:,:,:]    = target_sample[0,1*no_depth_levels:2*no_depth_levels,:,:]
   #nc_PredSal[:,:,:]    = predicted[0,1*no_depth_levels:2*no_depth_levels,:,:]
   #nc_SalErrors[:,:,:]  = predicted[0,1*no_depth_levels:2*no_depth_levels,:,:] - target_sample[0,1*no_depth_levels:2*no_depth_levels,:,:]
   nc_TrueU[:,:,:]      = target_sample[0,1*no_depth_levels:2*no_depth_levels,:,:]
   nc_PredU[:,:,:]      = predicted[0,1*no_depth_levels:2*no_depth_levels,:,:]
   nc_UErrors[:,:,:]    = predicted[0,1*no_depth_levels:2*no_depth_levels,:,:] - target_sample[0,1*no_depth_levels:2*no_depth_levels,:,:]
   nc_TrueV[:,:,:]      = target_sample[0,2*no_depth_levels:3*no_depth_levels,:,:]
   nc_PredV[:,:,:]      = predicted[0,2*no_depth_levels:3*no_depth_levels,:,:]
   nc_VErrors[:,:,:]    = predicted[0,2*no_depth_levels:3*no_depth_levels,:,:] - target_sample[0,2*no_depth_levels:3*no_depth_levels,:,:]
   nc_TrueEta[:,:]    = target_sample[0,3*no_depth_levels,:,:]
   nc_PredEta[:,:]    = predicted[0,3*no_depth_levels,:,:]
   nc_EtaErrors[:,:]  = predicted[0,3*no_depth_levels,:,:] - target_sample[0,3*no_depth_levels,:,:]
   
    
def OutputStats(model_name, MeanStd_prefix, mitgcm_filename, data_loader, h, no_epochs):
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

   data_mean, data_std, data_range, no_channels = ReadMeanStd(MeanStd_prefix)

   # Read in grid data from MITgcm file
   mitgcm_ds = xr.open_dataset(mitgcm_filename)
   da_X = mitgcm_ds['X']
   da_Y = mitgcm_ds['Yp1']
   da_Z = mitgcm_ds['diag_levels'] 

   FirstPass = True

   batch_no = 0

   h.train(False)

   for batch_sample in data_loader:
       print('batch_no ; '+str(batch_no))

       input_batch  = batch_sample['input']
       target_batch = batch_sample['output']
   
       # Converting inputs and labels to Variable and send to GPU if available
       if torch.cuda.is_available():
           input_batch = Variable(input_batch.cuda().float())
           target_batch = Variable(target_batch.cuda().float())
       else:
           input_batch = Variable(input_batch.float())
           target_batch = Variable(target_batch.float())
       
       # get prediction from the model, given the inputs
       predicted = h(input_batch)
  
       if FirstPass:
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

   print('all batches done')
   print(targets.shape)
   print(predictions.shape)

   # Denormalise
   targets = rr.RF_DeNormalise(targets, data_mean, data_std, data_range)
   predictions = rr.RF_DeNormalise(predictions, data_mean, data_std, data_range)
   
   # Set up netcdf files
   nc_filename = '../../../Channel_nn_Outputs/'+model_name+'/STATS/'+model_name+'_'+str(no_epochs)+'epochs_StatsOutput.nc'
   nc_file = nc4.Dataset(nc_filename,'w', format='NETCDF4') #'w' stands for write
   # Create Dimensions
   no_depth_levels = 38  # Hard coded...perhaps should change...?
   no_samples      = min(50, targets.shape[0]) 
   y_dim_used = 96   # point at which land starts - i.e. y dim of used-grid
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

   nc_TrueTemp[:,:,:,:] = targets[:no_samples,0:no_depth_levels,:,:] 
   nc_TrueU[:,:,:,:]    = targets[:no_samples,1*no_depth_levels:2*no_depth_levels,:,:]
   nc_TrueV[:,:,:,:]    = targets[:no_samples,2*no_depth_levels:3*no_depth_levels,:,:]
   nc_TrueEta[:,:,:]    = targets[:no_samples,3*no_depth_levels,:,:]

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
      print('predictions contains a NaN')
      print('nans at '+str(np.argwhere(np.isnan(predictions))) )
   if np.isnan(targets).any():
      print('targets contains a NaN')
      print('nans at '+str(np.argwhere(np.isnan(targets))) )
   for x in range(targets.shape[3]):
     for y in range(targets.shape[2]):
        for z in range(no_depth_levels):
           nc_TempCC[z,y,x] = np.corrcoef( predictions[:,                  z,y,x], targets[:,                  z,y,x], rowvar=False )[0,1]
           nc_U_CC[z,y,x]   = np.corrcoef( predictions[:,1*no_depth_levels+z,y,x], targets[:,1*no_depth_levels+z,y,x], rowvar=False )[0,1]
           nc_V_CC[z,y,x]   = np.corrcoef( predictions[:,2*no_depth_levels+z,y,x], targets[:,2*no_depth_levels+z,y,x], rowvar=False )[0,1]
        nc_EtaCC[y,x]  = np.corrcoef( predictions[:,3*no_depth_levels,y,x], targets[:,3*no_depth_levels,y,x], rowvar=False )[0,1]
   
def IterativelyPredict(model_name, MeanStd_prefix, mitgcm_filename, Iterate_Dataset, h, start, for_len, no_epochs):
   #-------------------------------------------------------------

   data_mean, data_std, data_range, no_channels = ReadMeanStd(MeanStd_prefix)

   # Read in grid data from MITgcm file
   mitgcm_ds = xr.open_dataset(mitgcm_filename)
   da_X = mitgcm_ds['X']
   da_Y = mitgcm_ds['Yp1']
   da_Z = mitgcm_ds['diag_levels'] 

   h.train(False)
   if torch.cuda.is_available():
       h = h.cuda()

   # Read in first sample from MITgcm dataset and save as first entry in both arrays
   input_sample = Iterate_Dataset.__getitem__(start)['input'][:,:,:].unsqueeze(0).numpy()
   iterated_predictions = input_sample
   MITgcm_data = input_sample

   # Make iterative forecast, saving predictions as we go, and pull out MITgcm data
   for time in range(for_len):
      print(time)

      if torch.cuda.is_available():
         predicted = h( Variable(torch.from_numpy(input_sample).cuda().float()) ).cpu().detach().numpy()
      else:
         predicted = h( Variable(torch.from_numpy(input_sample).float()) ).detach().numpy()

      iterated_predictions = np.concatenate( ( iterated_predictions, predicted ), axis=0 )
      MITgcm_data = np.concatenate( ( MITgcm_data, 
                                         Iterate_Dataset.__getitem__(start+time+1)['input'][:,:,:].unsqueeze(0).numpy() ), axis=0 )

      input_sample = predicted   # Set outputs as new inputs for next prediction
      del predicted
      gc.collect()
      torch.cuda.empty_cache()
     
   print('iterated_predictions.shape') 
   print(iterated_predictions.shape) 
   print('MITgcm_data.shape') 
   print(MITgcm_data.shape) 
   # Denormalise 
   iterated_predictions = rr.RF_DeNormalise(iterated_predictions, data_mean, data_std, data_range)
   MITgcm_data          = rr.RF_DeNormalise(MITgcm_data,          data_mean, data_std, data_range)

   # Set up netcdf files
   nc_filename = '../../../Channel_nn_Outputs/'+model_name+'/ITERATED_FORECAST/'+model_name+'_'+str(no_epochs)+'epochs_Forecast.nc'
   nc_file = nc4.Dataset(nc_filename,'w', format='NETCDF4') #'w' stands for write
   # Create Dimensions
   no_depth_levels = 38  # Hard coded...perhaps should change...?
   y_dim_used = 96   # point at which land starts - i.e. y dim of used-grid
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
  
   nc_TrueTemp[:,:,:,:] = MITgcm_data[:,0:no_depth_levels,:,:] 
   nc_TrueU[:,:,:,:]    = MITgcm_data[:,1*no_depth_levels:2*no_depth_levels,:,:]
   nc_TrueV[:,:,:,:]    = MITgcm_data[:,2*no_depth_levels:3*no_depth_levels,:,:]
   nc_TrueEta[:,:,:]    = MITgcm_data[:,3*no_depth_levels,:,:]

   nc_PredTemp[:,:,:,:] = iterated_predictions[:,0:no_depth_levels,:,:]
   nc_PredU[:,:,:,:]    = iterated_predictions[:,1*no_depth_levels:2*no_depth_levels,:,:]
   nc_PredV[:,:,:,:]    = iterated_predictions[:,2*no_depth_levels:3*no_depth_levels,:,:]
   nc_PredEta[:,:,:]    = iterated_predictions[:,3*no_depth_levels,:,:]

   nc_TempErrors[:,:,:,:] = ( iterated_predictions[:,0:no_depth_levels,:,:]
                              - MITgcm_data[:,0:no_depth_levels,:,:] )
   nc_UErrors[:,:,:,:]    = ( iterated_predictions[:,1*no_depth_levels:2*no_depth_levels,:,:]
                              - MITgcm_data[:,1*no_depth_levels:2*no_depth_levels,:,:] )
   nc_VErrors[:,:,:,:]    = ( iterated_predictions[:,2*no_depth_levels:3*no_depth_levels,:,:] 
                              - MITgcm_data[:,2*no_depth_levels:3*no_depth_levels,:,:] )
   nc_EtaErrors[:,:,:]    = ( iterated_predictions[:,3*no_depth_levels,:,:] 
                              - MITgcm_data[:,3*no_depth_levels,:,:] )
