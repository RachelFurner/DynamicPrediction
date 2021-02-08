#!/usr/bin/env python
# coding: utf-8

# Code to train a network to learn the dynamic evolution of an MITgcm channel configuration
# Designed to take the entire field for all variables, apply a NN (with conv layers), and output
# the entire field for all variables one day later (i.e. the next iteration of the MITgcm netcdf 
# file here, although note this is 2 steps of the underlying model which runs a 12 hourly timestep.
# The data is subsampled in time to give quasi-independence

#from comet_ml import Experiment
print('starting script')

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

def CalcMeanStd(model_name, MITGCM_filename, train_end_ratio, subsample_rate, dataset_end_index, batch_size, output_file):

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
   mean_std_file = '../../../INPUT_OUTPUT_ARRAYS/Channel_Whole_Grid_'+model_name+'_MeanStd.npz'
   np.savez( mean_std_file, inputs_mean, inputs_std, inputs_range, no_channels)
   
   output_file.write('no_channels ;'+str(no_channels)+'\n')
   print('no_channels ;'+str(no_channels)+'\n')
  
   del batch_sample
   del input_batch

   return no_channels


def ReadMeanStd(model_name, output_file=None):
   mean_std_file = '../../../INPUT_OUTPUT_ARRAYS/Channel_Whole_Grid_'+model_name+'_MeanStd.npz'
   mean_std_data = np.load(mean_std_file)
   inputs_mean  = mean_std_data['arr_0']
   inputs_std   = mean_std_data['arr_1']
   inputs_range = mean_std_data['arr_2']
   no_channels  = mean_std_data['arr_3']

   if output_file:
      output_file.write('no_channels ;'+str(no_channels)+'\n')
   print('no_channels ;'+str(no_channels)+'\n')

   return(inputs_mean, inputs_std, inputs_range, no_channels)


def TrainModel(model_name, output_file, tic, TEST, no_tr_samples, no_val_samples, plot_freq, save_freq, train_loader, val_loader, h, optimizer, hyper_params, reproducible, seed_value, start_epoch=0):

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

   train_loss_epoch = []
   train_loss_epoch_Temp = []
   train_loss_epoch_U = []
   train_loss_epoch_V = []
   train_loss_epoch_Eta = []
   val_loss_epoch = []
   val_loss_epoch_Temp = []
   val_loss_epoch_U = []
   val_loss_epoch_V = []
   val_loss_epoch_Eta = []
   first_pass = True
   torch.cuda.empty_cache()
   
   plot_dir = '../../../Channel_nn_Outputs/PLOTS/'+model_name
   if not os.path.isdir(plot_dir):
      os.system("mkdir %s" % (plot_dir))
   
   print('')
   print('###### TRAINING MODEL ######')
   print('')
   for epoch in range(start_epoch,hyper_params["num_epochs"]+start_epoch):
   
       print('epoch ; '+str(epoch))
       i=1
   
       # Training
   
       train_loss_epoch_tmp = 0.0
       train_loss_epoch_Temp_tmp = 0.
       train_loss_epoch_U_tmp = 0.
       train_loss_epoch_V_tmp = 0.
       train_loss_epoch_Eta_tmp = 0.
   
       if epoch%plot_freq == 0 or epoch == (hyper_params["num_epochs"]+start_epoch-1):
           fig = plt.figure(figsize=(9,9))
           ax1 = fig.add_subplot(111)
           bottom = 0.
           top = 0.
   
       for batch_sample in train_loader:
           print('')
           print('------------------- Training batch number : '+str(i)+'-------------------')
   
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
   
           if epoch%plot_freq == 0 or epoch == (hyper_params["num_epochs"]+start_epoch-1):
               target_batch = target_batch.cpu().detach().numpy()
               predicted = predicted.cpu().detach().numpy()
               ax1.scatter(target_batch[:,:no_depth_levels,:,:], predicted[:,:no_depth_levels,:,:],
                           edgecolors=(0, 0, 0), alpha=0.15, color='blue', label='Temp')
               ax1.scatter(target_batch[:,no_depth_levels:2*no_depth_levels,:,:], predicted[:,no_depth_levels:2*no_depth_levels,:,:],
                           edgecolors=(0, 0, 0), alpha=0.15, color='red', label='U')
               ax1.scatter(target_batch[:,2*no_depth_levels:3*no_depth_levels,:,:], predicted[:,2*no_depth_levels:3*no_depth_levels,:,:],
                           edgecolors=(0, 0, 0), alpha=0.15, color='orange', label='V')
               ax1.scatter(target_batch[:,3*no_depth_levels,:,:], predicted[:,3*no_depth_levels,:,:],
                           edgecolors=(0, 0, 0), alpha=0.15, color='purple', label='Eta')
               ax1.legend()
               bottom = min(np.amin(predicted), np.amin(target_batch), bottom)
               top    = max(np.amax(predicted), np.amax(target_batch), top)  
               print('did plotting')
   
           gc.collect()
           torch.cuda.empty_cache()
           for name, size in sorted(((name, sys.getsizeof(value)) for name, value in locals().items()),
                                    key= lambda x: -x[1])[:10]:
               print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))
           #print(torch.cuda.memory_summary(device))
           print('')

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
                 #plt.yscale('log')
                 #plt.annotate('skew = '+str(np.round(skew(data),5)), (0.1,0.9), xycoords='figure fraction')
                 plt.savefig(plot_dir+'/'+model_name+'_histogram_var'+str(no_var)+'_batch'+str(i)+'.png', bbox_inches = 'tight', pad_inches = 0.1)
                 plt.close()

           del input_batch
           del target_batch
           first_pass = False
           print('batch complete')
           i=i+1
 
       tr_loss_single_epoch = train_loss_epoch_tmp / no_tr_samples
       print('epoch {}, training loss {}'.format( epoch, tr_loss_single_epoch )+'\n')
       output_file.write('epoch {}, training loss {}'.format( epoch, tr_loss_single_epoch )+'\n')
       train_loss_epoch.append( tr_loss_single_epoch )
       train_loss_epoch_Temp.append( train_loss_epoch_Temp_tmp / no_tr_samples )
       train_loss_epoch_U.append( train_loss_epoch_U_tmp / no_tr_samples )
       train_loss_epoch_V.append( train_loss_epoch_V_tmp / no_tr_samples )
       train_loss_epoch_Eta.append( train_loss_epoch_Eta_tmp / no_tr_samples )
   
       if TEST: 
          print('tr_loss_single_epoch; '+str(tr_loss_single_epoch))
          print('train_loss_epoch; '+str(train_loss_epoch))
   
       if epoch%plot_freq == 0 or epoch == (hyper_params["num_epochs"]+start_epoch-1):
           ax1.set_xlabel('Truth')
           ax1.set_ylabel('Predictions')
           ax1.set_title('Training errors, epoch '+str(epoch))
           ax1.set_xlim(bottom, top)
           ax1.set_ylim(bottom, top)
           ax1.plot([bottom, top], [bottom, top], 'k--', lw=1, color='blue')
           ax1.annotate('Epoch Loss: '+str(tr_loss_single_epoch), (0.15, 0.9), xycoords='figure fraction')
           plt.savefig(plot_dir+'/'+model_name+'_scatter_epoch'+str(epoch).rjust(3,'0')+'training.png',
                       bbox_inches = 'tight', pad_inches = 0.1)
           plt.close()
   
       # Validation
   
       val_loss_epoch_tmp = 0.0
       i=1
       
       if epoch%plot_freq == 0 or epoch == (hyper_params["num_epochs"]+start_epoch-1):
           fig = plt.figure(figsize=(9,9))
           ax1 = fig.add_subplot(111)
           bottom = 0.
           top = 0.
   
       for batch_sample in val_loader:
           print('')
           print('------------------- Validation batch number : '+str(i)+'-------------------')
   
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
       
           # get loss for the predicted output
           val_loss = hyper_params["criterion"](predicted, target_batch)
   
           val_loss_epoch_tmp += val_loss.item()*input_batch.shape[0]
   
           if TEST: 
              print('val_loss_epoch_tmp; '+str(val_loss_epoch_tmp))
  
           if ( epoch%plot_freq == 0 or epoch == (hyper_params["num_epochs"]+start_epoch-1) ):
               target_batch = target_batch.cpu().detach().numpy()
               predicted = predicted.cpu().detach().numpy()
               ax1.scatter(target_batch[:,:no_depth_levels,:,:], predicted[:,:no_depth_levels,:,:],
                           edgecolors=(0, 0, 0), alpha=0.15, color='blue', label='Temp')
               ax1.scatter(target_batch[:,no_depth_levels:2*no_depth_levels,:,:], predicted[:,no_depth_levels:2*no_depth_levels,:,:],
                           edgecolors=(0, 0, 0), alpha=0.15, color='red', label='U')
               ax1.scatter(target_batch[:,2*no_depth_levels:3*no_depth_levels,:,:], predicted[:,2*no_depth_levels:3*no_depth_levels,:,:],
                           edgecolors=(0, 0, 0), alpha=0.15, color='orange', label='V')
               ax1.scatter(target_batch[:,3*no_depth_levels,:,:], predicted[:,3*no_depth_levels,:,:],
                           edgecolors=(0, 0, 0), alpha=0.15, color='purple', label='Eta')
               ax1.legend()
               bottom = min(np.amin(predicted), np.amin(target_batch), bottom)
               top    = max(np.amax(predicted), np.amax(target_batch), top)    
           i=i+1
           del input_batch
           del target_batch
           gc.collect()
           torch.cuda.empty_cache()
           print('batch complete')
           for name, size in sorted(((name, sys.getsizeof(value)) for name, value in locals().items()),
                                    key= lambda x: -x[1])[:10]:
               print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))
           #print(torch.cuda.memory_summary(device))
           print('')
   
       val_loss_single_epoch = val_loss_epoch_tmp / no_val_samples
       print('epoch {}, validation loss {}'.format(epoch, val_loss_single_epoch)+'\n')
       output_file.write('epoch {}, validation loss {}'.format(epoch, val_loss_single_epoch)+'\n')
       val_loss_epoch.append(val_loss_single_epoch)
   
       if TEST: 
          print('val_loss_single_epoch; '+str(val_loss_single_epoch))
          print('val_loss_epoch; '+str(val_loss_epoch))
   
       if epoch%plot_freq == 0 or epoch == (hyper_params["num_epochs"]+start_epoch-1):
           ax1.set_xlabel('Truth')
           ax1.set_ylabel('Predictions')
           ax1.set_title('Validation errors, epoch '+str(epoch))
           ax1.set_xlim(bottom, top)
           ax1.set_ylim(bottom, top)
           ax1.plot([bottom, top], [bottom, top], 'k--', lw=1, color='blue')
           ax1.annotate('Epoch MSE: '+str(np.round(val_loss_epoch_tmp / no_val_samples,5)),
                         (0.15, 0.9), xycoords='figure fraction')
           ax1.annotate('Epoch Loss: '+str(val_loss_single_epoch), (0.15, 0.87), xycoords='figure fraction')
           plt.savefig(plot_dir+'/'+model_name+'_scatter_epoch'+str(epoch).rjust(3,'0')+'validation.png',
                       bbox_inches = 'tight', pad_inches = 0.1)
           plt.close()

       if (epoch%save_freq == 0 and epoch != start_epoch) or epoch == (hyper_params["num_epochs"]+start_epoch-1):
           print('save model \n')
           output_file.write('save model \n')
           pkl_filename = '../../../Channel_nn_Outputs/MODELS/'+model_name+'_epoch'+str(epoch)+'_SavedModel.pt'
           #torch.save(h.state_dict(), pkl_filename)
           torch.save({
                       'epoch': epoch,
                       'model_state_dict': h.state_dict(),
                       'optimizer_state_dict': optimizer.state_dict(),
                       'loss': loss,
                       }, pkl_filename)
   
       toc = time.time()
       print('Finished epoch {} at {:0.4f} seconds'.format(epoch, toc - tic)+'\n')
       output_file.write('Finished epoch {} at {:0.4f} seconds'.format(epoch, toc - tic)+'\n')
   
   torch.cuda.empty_cache()

   # Plot training and validation loss 
   fig = plt.figure()
   ax1 = fig.add_subplot(111)
   ax1.plot(range(start_epoch, start_epoch+hyper_params["num_epochs"]), train_loss_epoch, color='black', label='Total Training Loss')
   ax1.plot(range(start_epoch, start_epoch+hyper_params["num_epochs"]), train_loss_epoch_Temp, color='blue', label='Temperature Training Loss')
   ax1.plot(range(start_epoch, start_epoch+hyper_params["num_epochs"]), train_loss_epoch_U, color='red', label='U Training Loss')
   ax1.plot(range(start_epoch, start_epoch+hyper_params["num_epochs"]), train_loss_epoch_V, color='orange', label='V Training Loss')
   ax1.plot(range(start_epoch, start_epoch+hyper_params["num_epochs"]), train_loss_epoch_Eta, color='purple', label='Eta Training Loss')
   ax1.plot(range(start_epoch, start_epoch+hyper_params["num_epochs"]), val_loss_epoch, color='grey', label='Validation Loss')
   ax1.set_xlabel('Epochs')
   ax1.set_ylabel('Loss')
   ax1.set_yscale('log')
   ax1.set_ylim(0.02, 0.3)
   ax1.legend()
   plt.savefig(plot_dir+'/'+model_name+'_TrainingValLossPerEpoch'+'_epoch'+str(start_epoch)+'-'+str(epoch)+'.png',
               bbox_inches = 'tight', pad_inches = 0.1)
   plt.close()
   
   info_filename = '../../../Channel_nn_Outputs/MODELS/'+model_name+'_info.txt'
   info_file = open(info_filename, 'w')
   np.set_printoptions(threshold=np.inf)
   info_file.write('hyper_params:    '+str(hyper_params)+'\n')
   info_file.write('\n')
   #info_file.write('Weights of the network:\n')
   #info_file.write(str(h[0].weight.data.cpu().numpy()))
  
def LoadModel(model_name, h, optimizer, saved_epoch, tr_inf):
   pkl_filename = '../../../Channel_nn_Outputs/MODELS/'+model_name+'_epoch'+str(saved_epoch)+'_SavedModel.pt'
   checkpoint = torch.load(pkl_filename)
   h.load_state_dict(checkpoint['model_state_dict'])
   optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
   epoch = checkpoint['epoch']
   loss = checkpoint['loss']

   if tr_inf == 'tr':
      h.train()
   elif tr_inf == 'inf':
      h.eval()

def OutputSinglePrediction(model_name, Train_Dataset, h, no_epochs):
   #-----------------------------------------------------------------------------------------
   # Output the predictions for a single input into a netcdf file to check it looks sensible
   #-----------------------------------------------------------------------------------------
   data_mean, data_std, data_range, no_channels = ReadMeanStd(model_name)
   
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
   nc_filename = '../../../Channel_nn_Outputs/STATS/'+model_name+'_'+str(no_epochs)+'epochs_DummyOutput.nc'
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
   
    
def OutputStats(model_name, data_loader, h, no_epochs):
   #-------------------------------------------------------------
   # Output the rms and correlation coefficients over a dataset
   #-------------------------------------------------------------
   #  1. Make predictions over entire dataset (training, val or test)
   #  2. Store predictions and targets in a large array (size of samples in dataset) 
   #     (if this is not possible, cause of memory, need to add RMS errors as 
   #     we go, and the variables needed to calculate the correlation coeff)
   #  3. Calculate the RMS and correlation coefficient of the prediction and target arrays
   #  4. Store the RMS and correlation coeffs in netcdf file

   data_mean, data_std, data_range, no_channels = ReadMeanStd(model_name)

   FirstPass = True

   for batch_sample in data_loader:

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
  
       if FirstPass:
          targets     = target_batch.cpu().detach().numpy()
          predictions = predicted.cpu().detach().numpy() 
       else:
          targets     = np.concatenate( ( targets, target_batch.cpu().detach().numpy() ), axis=0)
          predictions = np.concatenate( ( predictions, predicted.cpu().detach().numpy() ), axis=0)
 
       del input_batch
       del target_batch
       gc.collect()
       torch.cuda.empty_cache()

       FirstPass = False

   print(targets.shape)
   print(predictions.shape)

   # Denormalise
   targets = rr.RF_DeNormalise(targets, data_mean, data_std, data_range)
   predictions = rr.RF_DeNormalise(predictions, data_mean, data_std, data_range)
   
   # Set up netcdf files
   nc_filename = '../../../Channel_nn_Outputs/STATS/'+model_name+'_'+str(no_epochs)+'epochs_StatsOutput.nc'
   nc_file = nc4.Dataset(nc_filename,'w', format='NETCDF4') #'w' stands for write
   # Create Dimensions
   no_depth_levels = 38  # Hard coded...perhaps should change...?
   nc_file.createDimension('T', targets.shape[0])
   nc_file.createDimension('Z', no_depth_levels)
   nc_file.createDimension('Y', targets.shape[2])
   nc_file.createDimension('X', targets.shape[3])
   # Create variables
   nc_T = nc_file.createVariable('T', 'i4', 'T')
   nc_Z = nc_file.createVariable('Z', 'i4', 'Z')
   nc_Y = nc_file.createVariable('Y', 'i4', 'Y')  
   nc_X = nc_file.createVariable('X', 'i4', 'X')

   nc_TrueTemp   = nc_file.createVariable( 'True_Temp'  , 'f4', ('T', 'Z', 'Y', 'X') )
   #nc_TrueSal    = nc_file.createVariable( 'True_Sal'   , 'f4', ('T', 'Z', 'Y', 'X') )
   nc_TrueU      = nc_file.createVariable( 'True_U'     , 'f4', ('T', 'Z', 'Y', 'X') )
   nc_TrueV      = nc_file.createVariable( 'True_V'     , 'f4', ('T', 'Z', 'Y', 'X') )
   nc_TrueEta    = nc_file.createVariable( 'True_Eta'   , 'f4', ('T', 'Y', 'X')      )

   nc_PredTemp   = nc_file.createVariable( 'Pred_Temp'  , 'f4', ('T', 'Z', 'Y', 'X') )
   #nc_PredSal    = nc_file.createVariable( 'Pred_Sal'   , 'f4', ('T', 'Z', 'Y', 'X') )
   nc_PredU      = nc_file.createVariable( 'Pred_U'     , 'f4', ('T', 'Z', 'Y', 'X') )
   nc_PredV      = nc_file.createVariable( 'Pred_V'     , 'f4', ('T', 'Z', 'Y', 'X') ) 
   nc_PredEta    = nc_file.createVariable( 'Pred_Eta'   , 'f4', ('T', 'Y', 'X')      )

   nc_TempErrors = nc_file.createVariable( 'Temp_Errors', 'f4', ('T', 'Z', 'Y', 'X') )
   #nc_SalErrors  = nc_file.createVariable( 'Sal_Errors' , 'f4', ('T', 'Z', 'Y', 'X') )
   nc_UErrors    = nc_file.createVariable( 'U_Errors'   , 'f4', ('T', 'Z', 'Y', 'X') )
   nc_VErrors    = nc_file.createVariable( 'V_Errors'   , 'f4', ('T', 'Z', 'Y', 'X') )
   nc_EtaErrors  = nc_file.createVariable( 'Eta_Errors' , 'f4', ('T', 'Y', 'X')      )

   nc_TempRMS    = nc_file.createVariable( 'TempRMS'    , 'f4', ('Z', 'Y', 'X')      )
   #nc_SalRMS     = nc_file.createVariable( 'SalRMS'     , 'f4', ('Z', 'Y', 'X')      )
   nc_U_RMS      = nc_file.createVariable( 'U_RMS'      , 'f4', ('Z', 'Y', 'X')      )
   nc_V_RMS      = nc_file.createVariable( 'V_RMS'      , 'f4', ('Z', 'Y', 'X')      )
   nc_EtaRMS     = nc_file.createVariable( 'EtaRMS'     , 'f4', ('Y', 'X')           )

   nc_TempCC     = nc_file.createVariable( 'TempCC'     , 'f4', ('Z', 'Y', 'X')      )
   #nc_SalCC      = nc_file.createVariable( 'SalCC'      , 'f4', ('Z', 'Y', 'X')      )
   nc_U_CC       = nc_file.createVariable( 'U_CC'       , 'f4', ('Z', 'Y', 'X')      )
   nc_V_CC       = nc_file.createVariable( 'V_CC'       , 'f4', ('Z', 'Y', 'X')      )
   nc_EtaCC      = nc_file.createVariable( 'EtaCC'      , 'f4', ('Y', 'X')           )

   # Fill variables
   nc_T[:] = np.arange(targets.shape[0])
   nc_Z[:] = np.arange(no_depth_levels)
   nc_Y[:] = np.arange(targets.shape[2])
   nc_X[:] = np.arange(targets.shape[3])
   
   nc_TrueTemp[:,:,:,:] = targets[:,0:no_depth_levels,:,:] 
   #nc_TrueSal[:,:,:,:]  = targets[:,1*no_depth_levels:2*no_depth_levels,:,:]
   nc_TrueU[:,:,:,:]    = targets[:,1*no_depth_levels:2*no_depth_levels,:,:]
   nc_TrueV[:,:,:,:]    = targets[:,2*no_depth_levels:3*no_depth_levels,:,:]
   nc_TrueEta[:,:,:]    = targets[:,3*no_depth_levels,:,:]

   nc_PredTemp[:,:,:,:] = predictions[:,0:no_depth_levels,:,:]
   #nc_PredSal[:,:,:,:]  = predictions[:,1*no_depth_levels:2*no_depth_levels,:,:]
   nc_PredU[:,:,:,:]    = predictions[:,1*no_depth_levels:2*no_depth_levels,:,:]
   nc_PredV[:,:,:,:]    = predictions[:,2*no_depth_levels:3*no_depth_levels,:,:]
   nc_PredEta[:,:,:]    = predictions[:,3*no_depth_levels,:,:]

   nc_TempErrors[:,:,:,:] = predictions[:,0:no_depth_levels,:,:] - targets[:,0:no_depth_levels,:,:]
   #nc_SalErrors[:,:,:,:]  = predictions[:,1*no_depth_levels:2*no_depth_levels,:,:] - targets[:,1*no_depth_levels:2*no_depth_levels,:,:]
   nc_UErrors[:,:,:,:]    = predictions[:,1*no_depth_levels:2*no_depth_levels,:,:] - targets[:,1*no_depth_levels:2*no_depth_levels,:,:]
   nc_VErrors[:,:,:,:]    = predictions[:,2*no_depth_levels:3*no_depth_levels,:,:] - targets[:,2*no_depth_levels:3*no_depth_levels,:,:]
   nc_EtaErrors[:,:,:]    = predictions[:,3*no_depth_levels,:,:] - targets[:,3*no_depth_levels,:,:]

   nc_TempRMS[:,:,:] = np.sqrt( np.mean( np.square(predictions[:,0:no_depth_levels,:,:] - targets[:,0:no_depth_levels,:,:]), axis=0) )
   #nc_SalRMS[:,:,:]  = np.sqrt( np.mean( np.square(predictions[:,1*no_depth_levels:2*no_depth_levels,:,:] - targets[:,1*no_depth_levels:2*no_depth_levels,:,:]), axis=0) )
   nc_U_RMS[:,:,:]   = np.sqrt( np.mean( np.square(predictions[:,1*no_depth_levels:2*no_depth_levels,:,:] - targets[:,1*no_depth_levels:2*no_depth_levels,:,:]), axis=0) )
   nc_V_RMS[:,:,:]   = np.sqrt( np.mean( np.square(predictions[:,2*no_depth_levels:3*no_depth_levels,:,:] - targets[:,2*no_depth_levels:3*no_depth_levels,:,:]), axis=0) )
   nc_EtaRMS[:,:]    = np.sqrt( np.mean( np.square(predictions[:,3*no_depth_levels,:,:] - targets[:,3*no_depth_levels,:,:]), axis=0) )

   # Use a loop for corellation coefficients, as get memory errors otherwise...
   #nc_TempCC[:,:,:] = np.corrcoef( predictions[:,0:no_depth_levels,:,:].reshape((predictions.shape[0],-1)), 
   #                           targets[:,0:no_depth_levels,:,:].reshape((predictions.shape[0],-1)), 
   #                           rowvar=False ).reshape((predictions.shape[0],no_depth_levels,predictions.shape[2],predictions.shape[3]))
   #nc_Sal_CC[:,:,:] = np.corrcoef( predictions[:,1*no_depth_levels:2*no_depth_levels,:,:].reshape((predictions.shape[0],-1)), 
   #                           targets[:,1*no_depth_levels:2*no_depth_levels,:,:].reshape((predictions.shape[0],-1)), 
   #                           rowvar=False ).reshape((predictions.shape[0],no_depth_levels,predictions.shape[2],predictions.shape[3]))
   #nc_U_CC[:,:,:]   = np.corrcoef( predictions[:,2*no_depth_levels:3*no_depth_levels,:,:].reshape((predictions.shape[0],-1)), 
   #                           targets[:,2*no_depth_levels:3*no_depth_levels,:,:].reshape((predictions.shape[0],-1)), 
   #                           rowvar=False ).reshape((predictions.shape[0],no_depth_levels,predictions.shape[2],predictions.shape[3]))
   #nc_V_CC[:,:,:]   = np.corrcoef( predictions[:,3*no_depth_levels:4*no_depth_levels,:,:].reshape((predictions.shape[0],-1)), 
   #                           targets[:,3*no_depth_levels:4*no_depth_levels,:,:].reshape((predictions.shape[0],-1)), 
   #                           rowvar=False ).reshape((predictions.shape[0],no_depth_levels,predictions.shape[2],predictions.shape[3]))
   #nc_EtaCC[:,:,:]  = np.corrcoef( predictions[:,4*no_depth_levels,:,:].reshape((predictions.shape[0],-1)), 
   #                           targets[:,4*no_depth_levels,:,:].reshape((predictions.shape[0],-1)), 
   #                           rowvar=False ).reshape((predictions.shape[0],1,predictions.shape[2],predictions.shape[3]))
   #for x in range(targets.shape[3]):
   #  for y in range(targets.shape[2]):
   #     for z in range(no_depth_levels):
   if np.isnan(predictions).any():
      print('predictions contains a NaN')
      print('nans at '+str(np.argwhere(np.isnan(predictions))) )
   if np.isnan(targets).any():
      print('targets contains a NaN')
      print('nans at '+str(np.argwhere(np.isnan(targets))) )
   #for x in range(3):
   #  for y in range(3):
   #     for z in range(2):
   #        nc_TempCC[z,y,x] = np.corrcoef( predictions[:,                  z,y,x], targets[:,                  z,y,x], rowvar=False )[0,1]
   #        print(nc_TempCC[z,y,x])
   #        nc_Sal_CC[z,y,x] = np.corrcoef( predictions[:,1*no_depth_levels+z,y,x], targets[:,1*no_depth_levels+z,y,x], rowvar=False )[0,1]
   #        nc_U_CC[z,y,x]   = np.corrcoef( predictions[:,2*no_depth_levels+z,y,x], targets[:,2*no_depth_levels+z,y,x], rowvar=False )[0,1]
   #        nc_V_CC[z,y,x]   = np.corrcoef( predictions[:,3*no_depth_levels+z,y,x], targets[:,3*no_depth_levels+z,y,x], rowvar=False )[0,1]
   #     nc_EtaCC[y,x]  = np.corrcoef( predictions[:,4*no_depth_levels,y,x], targets[:,4*no_depth_levels,y,x], rowvar=False )[0,1]
   
