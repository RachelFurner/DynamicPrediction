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

   for batch_sample in mean_std_loader:
       
       print('batch ' +str(batch_no))
       # shape (batch_size, channels, y, x)
       input_batch  = batch_sample['input']
       target_batch = batch_sample['output']

       # Calc mean and std of this batch
       batch_inputs_mean  = np.mean(input_batch.numpy(), axis = (0,2,3))
       batch_inputs_var   = np.var(input_batch.numpy(), axis = (0,2,3))
       batch_targets_mean = np.mean(target_batch.numpy(), axis = (0,2,3))
       batch_targets_var  = np.var(target_batch.numpy(), axis = (0,2,3))
 
       #combine with previous mean and std
       if FirstPass:
          inputs_mean  = batch_inputs_mean  
          inputs_var   = batch_inputs_var
          targets_mean = batch_targets_mean  
          targets_var  = batch_targets_var
       else:
          new_inputs_mean = 1./(count+input_batch.shape[0]) * ( count*inputs_mean + input_batch.shape[0]*batch_inputs_mean )
          inputs_var = ( 1./(count+input_batch.shape[0]) *
                                 ( count*inputs_var + input_batch.shape[0]*batch_inputs_var
                                 + count*np.square(inputs_mean-new_inputs_mean)
                                 + input_batch.shape[0]*np.square(batch_inputs_mean-new_inputs_mean) ) )
          new_targets_mean = 1./(count+target_batch.shape[0]) * ( count*targets_mean + target_batch.shape[0]*batch_targets_mean )
          targets_var = ( 1./(count+target_batch.shape[0]) *
                                 ( count*targets_var + target_batch.shape[0]*batch_targets_var
                                 + count*np.square(targets_mean-new_targets_mean)
                                 + target_batch.shape[0]*np.square(batch_targets_mean-new_targets_mean) ) )

          inputs_mean = new_inputs_mean
          targets_mean = new_targets_mean

       batch_no = batch_no+1
       count = count + input_batch.shape[0]
       FirstPass = False

   print('count ;'+str(count))

   # combine across all depth levels in each field (samples sizes are the same across all depths, so don't need to account for this)
   for i in range(3):  # Temp, U, V. No need to do for Eta as already 2-d/1 channel. Sal not included in model as const.
      inputs_mean_tmp = 1./no_depth_levels * np.sum(inputs_mean[i*no_depth_levels:(i+1)*no_depth_levels])
      inputs_var[i*no_depth_levels:(i+1)*no_depth_levels] = 1./no_depth_levels * \
                                                  ( np.sum(inputs_var[i*no_depth_levels:(i+1)*no_depth_levels]) + 
                                                    np.sum(inputs_mean[i*no_depth_levels:(i+1)*no_depth_levels]-inputs_mean_tmp) )
      inputs_mean[i*no_depth_levels:(i+1)*no_depth_levels] = inputs_mean_tmp 

      targets_mean_tmp = 1./no_depth_levels * np.sum(targets_mean[i*no_depth_levels:(i+1)*no_depth_levels])
      targets_var[i*no_depth_levels:(i+1)*no_depth_levels] = 1./no_depth_levels * \
                                                  ( np.sum(targets_var[i*no_depth_levels:(i+1)*no_depth_levels]) + 
                                                    np.sum(targets_mean[i*no_depth_levels:(i+1)*no_depth_levels]-targets_mean_tmp) )
      targets_mean[i*no_depth_levels:(i+1)*no_depth_levels] = targets_mean_tmp 

   inputs_std  = np.sqrt(inputs_var)
   targets_std = np.sqrt(targets_var)

   no_input_channels = input_batch.shape[1]
   no_target_channels = target_batch.shape[1]
   
   ## Save to file, so can be used to un-normalise when using model to predict
   mean_std_file = '../../../INPUT_OUTPUT_ARRAYS/Channel_Whole_Grid_'+model_name+'_MeanStd.npz'
   np.savez( mean_std_file, inputs_mean, inputs_std, targets_mean, targets_std, no_input_channels, no_target_channels)
   
   output_file.write('no_input_channels ;'+str(no_input_channels)+'\n')
   output_file.write('no_target_channels ;'+str(no_target_channels)+'\n')
   print('no_input_channels ;'+str(no_input_channels)+'\n')
   print('no_target_channels ;'+str(no_target_channels)+'\n')
  
   print(inputs_mean, targets_mean, inputs_std, targets_std)
 
   del batch_sample
   del input_batch
   del target_batch

   return no_input_channels, no_target_channels 


def ReadMeanStd(model_name, output_file=None):
   mean_std_file = '../../../INPUT_OUTPUT_ARRAYS/Channel_Whole_Grid_'+model_name+'_MeanStd.npz'
   mean_std_data = np.load(mean_std_file)
   inputs_mean  = mean_std_data['arr_0']
   inputs_std   = mean_std_data['arr_1']
   targets_mean = mean_std_data['arr_2']
   targets_std  = mean_std_data['arr_3']
   no_input_channels  = mean_std_data['arr_4']
   no_target_channels = mean_std_data['arr_5']

   if output_file:
      output_file.write('no_input_channels ;'+str(no_input_channels)+'\n')
      output_file.write('no_target_channels ;'+str(no_target_channels)+'\n')
   print('no_input_channels ;'+str(no_input_channels)+'\n')
   print('no_target_channels ;'+str(no_target_channels)+'\n')

   return(inputs_mean, inputs_std, targets_mean, targets_std, no_input_channels, no_target_channels)


def CreateModel(no_input_channels, no_target_channels, lr):
   # inputs are (no_samples, 153channels, 108y, 240x).

   h = nn.Sequential(
      # downscale
      # ****Note scher uses a kernel size of 6, but no idea how he then manages to keep his x and y dimensions
      # unchanged over the conv layers....***
      nn.Conv2d(in_channels=no_input_channels, out_channels=64, kernel_size=(7,7), padding=(3,3)),
      nn.ReLU(True),
      nn.MaxPool2d((2,2)),
      nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(7,7),padding=(3,3)),
      nn.ReLU(True),
      nn.MaxPool2d((2,2)),

      nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=(7,7), padding=(3,3)),
      nn.ReLU(True),

      # upscale
      nn.Upsample(scale_factor=(2,2)),
      nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(7,7), padding=(3,3)),
      nn.ReLU(True),
      nn.Upsample(scale_factor=(2,2)),
      nn.Conv2d(in_channels=64, out_channels=no_target_channels, kernel_size=(7,7), padding=(3,3)),
      nn.ReLU(True)
       )
   h = h.cuda()

   optimizer = torch.optim.Adam( h.parameters(), lr=lr )

   return h, optimizer

def TrainModel(model_name, output_file, tic, TEST, no_tr_samples, no_val_samples, plot_freq, train_loader, val_loader, h, optimizer, hyper_params):
   train_loss_epoch = []
   val_loss_epoch = []
   first_pass = True
   torch.cuda.empty_cache()
   
   plot_dir = '../../../Channel_nn_Outputs/PLOTS/'+model_name
   if not os.path.isdir(plot_dir):
      os.system("mkdir %s" % (plot_dir))
   
   print('')
   print('###### TRAINING MODEL ######')
   print('')
   for epoch in range(hyper_params["num_epochs"]):
   
       print('epoch ; '+str(epoch))
       i=1
   
       # Training
   
       train_loss_epoch_temp = 0.0
   
       if epoch%plot_freq == 0:
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
           pre_opt_loss = hyper_params["criterion"](predicted, target_batch)
           #print('calculated loss')
           
           # get gradients w.r.t to parameters
           pre_opt_loss.backward()
           #print('calculated loss gradients')
        
           # update parameters
           optimizer.step()
           #print('did optimization')
   
           if TEST and first_pass:  # Recalculate loss post optimizer step to ensure its decreased
              h.train(False)
              post_opt_predicted = h(input_batch)
              post_opt_loss = hyper_params["criterion"](post_opt_predicted, target_batch)
              print('For test run check if loss decreases over a single training step on a batch of data')
              print('loss pre optimization: ' + str(pre_opt_loss.item()) )
              print('loss post optimization: ' + str(post_opt_loss.item()) )
              output_file.write('For test run check if loss decreases over a single training step on a batch of data\n')
              output_file.write('loss pre optimization: ' + str(pre_opt_loss.item()) + '\n')
              output_file.write('loss post optimization: ' + str(post_opt_loss.item()) + '\n')
           
           train_loss_epoch_temp += pre_opt_loss.item()*input_batch.shape[0]
           #print('Added loss to epoch temp')
   
           if TEST: 
              print('train_loss_epoch_temp; '+str(train_loss_epoch_temp))
   
           if epoch%plot_freq == 0:
               target_batch = target_batch.cpu().detach().numpy()
               predicted = predicted.cpu().detach().numpy()
               ax1.scatter(target_batch, predicted, edgecolors=(0, 0, 0), alpha=0.15)
               bottom = min(np.amin(predicted), np.amin(target_batch), bottom)
               top    = max(np.amax(predicted), np.amax(target_batch), top)  
               print('did plotting')
   
           del input_batch
           del target_batch
           gc.collect()
           torch.cuda.empty_cache()
           first_pass = False
           i=i+1
           print('batch complete')
           for name, size in sorted(((name, sys.getsizeof(value)) for name, value in locals().items()),
                                    key= lambda x: -x[1])[:10]:
               print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))
           #print(torch.cuda.memory_summary(device))
           print('')
   
       tr_loss_single_epoch = train_loss_epoch_temp / no_tr_samples
       print('epoch {}, training loss {}'.format( epoch, tr_loss_single_epoch )+'\n')
       output_file.write('epoch {}, training loss {}'.format( epoch, tr_loss_single_epoch )+'\n')
       train_loss_epoch.append( tr_loss_single_epoch )
   
       if TEST: 
          print('tr_loss_single_epoch; '+str(tr_loss_single_epoch))
          print('train_loss_epoch; '+str(train_loss_epoch))
   
       if epoch%plot_freq == 0:
           ax1.set_xlabel('Truth')
           ax1.set_ylabel('Predictions')
           ax1.set_title('Training errors, epoch '+str(epoch))
           ax1.set_xlim(bottom, top)
           ax1.set_ylim(bottom, top)
           ax1.plot([bottom, top], [bottom, top], 'k--', lw=1, color='blue')
           ax1.annotate('Epoch Loss: '+str(tr_loss_single_epoch), (0.15, 0.9), xycoords='figure fraction')
           plt.savefig(plot_dir+'/'+model_name+'_scatter_epoch'+str(epoch).rjust(3,'0')+'training.png', bbox_inches = 'tight', pad_inches = 0.1)
           plt.close()
   
       # Validation
   
       val_loss_epoch_temp = 0.0
       i=1
       
       if epoch%plot_freq == 0:
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
           loss = hyper_params["criterion"](predicted, target_batch)
   
           val_loss_epoch_temp += loss.item()*input_batch.shape[0]
   
           if TEST: 
              print('val_loss_epoch_temp; '+str(val_loss_epoch_temp))
   
           if epoch%plot_freq == 0:
               target_batch = target_batch.cpu().detach().numpy()
               predicted = predicted.cpu().detach().numpy()
               ax1.scatter(target_batch, predicted, edgecolors=(0, 0, 0), alpha=0.15)
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
   
       val_loss_single_epoch = val_loss_epoch_temp / no_val_samples
       print('epoch {}, validation loss {}'.format(epoch, val_loss_single_epoch)+'\n')
       output_file.write('epoch {}, validation loss {}'.format(epoch, val_loss_single_epoch)+'\n')
       val_loss_epoch.append(val_loss_single_epoch)
   
       if TEST: 
          print('val_loss_single_epoch; '+str(val_loss_single_epoch))
          print('val_loss_epoch; '+str(val_loss_epoch))
   
       if epoch%plot_freq == 0:
           ax1.set_xlabel('Truth')
           ax1.set_ylabel('Predictions')
           ax1.set_title('Validation errors, epoch '+str(epoch))
           ax1.set_xlim(bottom, top)
           ax1.set_ylim(bottom, top)
           ax1.plot([bottom, top], [bottom, top], 'k--', lw=1, color='blue')
           ax1.annotate('Epoch MSE: '+str(np.round(val_loss_epoch_temp / no_val_samples,5)),
                         (0.15, 0.9), xycoords='figure fraction')
           ax1.annotate('Epoch Loss: '+str(val_loss_single_epoch), (0.15, 0.87), xycoords='figure fraction')
           plt.savefig(plot_dir+'/'+model_name+'_scatter_epoch'+str(epoch).rjust(3,'0')+'validation.png', bbox_inches = 'tight', pad_inches = 0.1)
           plt.close()
   
       toc = time.time()
       print('Finished epoch {} at {:0.4f} seconds'.format(epoch, toc - tic)+'\n')
       output_file.write('Finished epoch {} at {:0.4f} seconds'.format(epoch, toc - tic)+'\n')
   
   torch.cuda.empty_cache()

   print('save model \n')
   output_file.write('save model \n')
   pkl_filename = '../../../Channel_nn_Outputs/MODELS/'+model_name+'_SavedModel.pt'
   torch.save(h.state_dict(), pkl_filename)
   
   # Plot training and validation loss 
   fig = plt.figure()
   ax1 = fig.add_subplot(111)
   ax1.plot(train_loss_epoch)
   ax1.plot(val_loss_epoch)
   ax1.set_xlabel('Epochs')
   ax1.set_ylabel('Loss')
   ax1.set_yscale('log')
   ax1.legend(['Training Loss', 'Validation Loss'])
   plt.savefig(plot_dir+'/'+model_name+'_TrainingValLossPerEpoch.png', bbox_inches = 'tight', pad_inches = 0.1)
   plt.close()
   
   info_filename = '../../../Channel_nn_Outputs/MODELS/'+model_name+'_info.txt'
   info_file = open(info_filename, 'w')
   np.set_printoptions(threshold=np.inf)
   info_file.write('hyper_params:    '+str(hyper_params)+'\n')
   info_file.write('\n')
   info_file.write('Weights of the network:\n')
   info_file.write(str(h[0].weight.data.cpu().numpy()))
  
def LoadModel(model_name, h):
   pkl_filename = '../../../Channel_nn_Outputs/MODELS/'+model_name+'_SavedModel.pt'
   h.load_state_dict(torch.load(pkl_filename))
   h.eval()

def OutputSinglePrediction(model_name, Train_Dataset, h):
   #-----------------------------------------------------------------------------------------
   # Output the predictions for a single input into a netcdf file to check it looks sensible
   #-----------------------------------------------------------------------------------------
   inputs_mean, inputs_std, targets_mean, targets_std, no_input_channels, no_target_channels = ReadMeanStd(model_name)
   
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
   target_sample = rr.RF_DeNormalise(target_sample, targets_mean, targets_std)
   predicted     = rr.RF_DeNormalise(predicted    , targets_mean, targets_std)
   
   # Set up netcdf files
   nc_filename = '../../../Channel_nn_Outputs/STATS/'+model_name+'_DummyOutput.nc'
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
   nc_TrueU[:,:,:]      = target_sample[0,1*no_depth_levels:2*no_depth_levels,:,:]
   nc_PredU[:,:,:]      = predicted[0,1*no_depth_levels:2*no_depth_levels,:,:]
   nc_UErrors[:,:,:]    = predicted[0,1*no_depth_levels:2*no_depth_levels,:,:] - target_sample[0,1*no_depth_levels:2*no_depth_levels,:,:]
   nc_TrueV[:,:,:]      = target_sample[0,2*no_depth_levels:3*no_depth_levels,:,:]
   nc_PredV[:,:,:]      = predicted[0,2*no_depth_levels:3*no_depth_levels,:,:]
   nc_VErrors[:,:,:]    = predicted[0,2*no_depth_levels:3*no_depth_levels,:,:] - target_sample[0,2*no_depth_levels:3*no_depth_levels,:,:]
   nc_TrueEta[:,:]    = target_sample[0,3*no_depth_levels,:,:]
   nc_PredEta[:,:]    = predicted[0,3*no_depth_levels,:,:]
   nc_EtaErrors[:,:]  = predicted[0,3*no_depth_levels,:,:] - target_sample[0,3*no_depth_levels,:,:]
   
    
def OutputStats(model_name, data_loader, h):
   #-------------------------------------------------------------
   # Output the rms and correclation coefficients over a dataset
   #-------------------------------------------------------------
   #  1. Make predictions over entire dataset (training, val or test)
   #  2. Store predictions and targets in a large array (size of samples in dataset) 
   #     (if this is not possible, cause of memory, need to add RMS errors as 
   #     we go, and the variables needed to calcualte the correlation coeef)
   #  3. Calculate the RMS and correlation coefficient of the prediction and target arrays
   #  4. Store the RMS and correlation coeffs in netcdf file

   inputs_mean, inputs_std, targets_mean, targets_std, no_input_channels, no_target_channels = ReadMeanStd(model_name)

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
   targets = rr.RF_DeNormalise(targets, targets_mean, targets_std)
   predictions = rr.RF_DeNormalise(predictions, targets_mean, targets_std)
   
   # Set up netcdf files
   nc_filename = '../../../Channel_nn_Outputs/STATS/'+model_name+'_StatsOutput.nc'
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
   nc_TrueTemp = nc_file.createVariable('True_Temp', 'f4', ('T', 'Z', 'Y', 'X'))
   nc_TrueU    = nc_file.createVariable('True_U', 'f4', ('T', 'Z', 'Y', 'X'))
   nc_TrueV    = nc_file.createVariable('True_V', 'f4', ('T', 'Z', 'Y', 'X'))
   nc_TrueEta  = nc_file.createVariable('True_Eta', 'f4', ('T', 'Y', 'X'))
   nc_PredTemp = nc_file.createVariable('Pred_Temp', 'f4', ('T', 'Z', 'Y', 'X'))
   nc_PredU    = nc_file.createVariable('Pred_U'   , 'f4', ('T', 'Z', 'Y', 'X'))
   nc_PredV    = nc_file.createVariable('Pred_V'   , 'f4', ('T', 'Z', 'Y', 'X'))
   nc_PredEta  = nc_file.createVariable('Pred_Eta' , 'f4', ('T', 'Y', 'X'))
   nc_TempErrors = nc_file.createVariable('Temp_Errors', 'f4', ('T', 'Z', 'Y', 'X'))
   nc_UErrors    = nc_file.createVariable('U_Errors'   , 'f4', ('T', 'Z', 'Y', 'X'))
   nc_VErrors    = nc_file.createVariable('V_Errors'   , 'f4', ('T', 'Z', 'Y', 'X'))
   nc_EtaErrors  = nc_file.createVariable('Eta_Errors' , 'f4', ('T', 'Y', 'X'))
   nc_TempRMS = nc_file.createVariable('TempRMS', 'f4', ('Z', 'Y', 'X'))
   nc_U_RMS   = nc_file.createVariable('U_RMS'  , 'f4', ('Z', 'Y', 'X'))
   nc_V_RMS   = nc_file.createVariable('V_RMS'  , 'f4', ('Z', 'Y', 'X'))
   nc_EtaRMS  = nc_file.createVariable('EtaRMS' , 'f4', ('Y', 'X'))
   nc_TempCC = nc_file.createVariable('TempCC', 'f4', ('Z', 'Y', 'X'))
   nc_U_CC   = nc_file.createVariable('U_CC'  , 'f4', ('Z', 'Y', 'X'))
   nc_V_CC   = nc_file.createVariable('V_CC'  , 'f4', ('Z', 'Y', 'X'))
   nc_EtaCC  = nc_file.createVariable('EtaCC' , 'f4', ('Y', 'X'))
   # Fill variables
   # Fill variables
   nc_T[:] = np.arange(targets.shape[0])
   nc_Z[:] = np.arange(no_depth_levels)
   nc_Y[:] = np.arange(targets.shape[2])
   nc_X[:] = np.arange(targets.shape[3])
   nc_TrueTemp[:,:,:,:] = targets[:,0:no_depth_levels,:,:] 
   nc_TrueU[:,:,:,:]    = targets[:,1*no_depth_levels:2*no_depth_levels,:,:]
   nc_TrueV[:,:,:,:]    = targets[:,2*no_depth_levels:3*no_depth_levels,:,:]
   nc_TrueEta[:,:,:]    = targets[:,3*no_depth_levels,:,:]
   nc_PredTemp[:,:,:,:] = predictions[:,0:no_depth_levels,:,:]
   nc_PredU[:,:,:,:]    = predictions[:,1*no_depth_levels:2*no_depth_levels,:,:]
   nc_PredV[:,:,:,:]    = predictions[:,2*no_depth_levels:3*no_depth_levels,:,:]
   nc_PredEta[:,:,:]    = predictions[:,3*no_depth_levels,:,:]
   nc_TempErrors[:,:,:,:] = predictions[:,0:no_depth_levels,:,:] - targets[:,0:no_depth_levels,:,:]
   nc_UErrors[:,:,:,:]    = predictions[:,1*no_depth_levels:2*no_depth_levels,:,:] - targets[:,1*no_depth_levels:2*no_depth_levels,:,:]
   nc_VErrors[:,:,:,:]    = predictions[:,2*no_depth_levels:3*no_depth_levels,:,:] - targets[:,2*no_depth_levels:3*no_depth_levels,:,:]
   nc_EtaErrors[:,:,:]    = predictions[:,3*no_depth_levels,:,:] - targets[:,3*no_depth_levels,:,:]
   nc_TempRMS[:,:,:] = np.sqrt( np.mean( np.square(predictions[:,0:no_depth_levels,:,:] - targets[:,0:no_depth_levels,:,:]), axis=0) )
   nc_U_RMS[:,:,:]   = np.sqrt( np.mean( np.square(predictions[:,1*no_depth_levels:2*no_depth_levels,:,:] - targets[:,1*no_depth_levels:2*no_depth_levels,:,:]), axis=0) )
   nc_V_RMS[:,:,:]   = np.sqrt( np.mean( np.square(predictions[:,2*no_depth_levels:3*no_depth_levels,:,:] - targets[:,2*no_depth_levels:3*no_depth_levels,:,:]), axis=0) )
   nc_EtaRMS[:,:]    = np.sqrt( np.mean( np.square(predictions[:,3*no_depth_levels,:,:] - targets[:,3*no_depth_levels,:,:]), axis=0) )
   # Use a loop for corellation coefficients, as get memory errors otherwise...
   #nc_TempCC[:,:,:] = np.corrcoef( predictions[:,0:no_depth_levels,:,:].reshape((predictions.shape[0],-1)), 
   #                           targets[:,0:no_depth_levels,:,:].reshape((predictions.shape[0],-1)), 
   #                           rowvar=False ).reshape((predictions.shape[0],no_depth_levels,predictions.shape[2],predictions.shape[3]))
   #nc_U_CC[:,:,:]   = np.corrcoef( predictions[:,1*no_depth_levels:2*no_depth_levels,:,:].reshape((predictions.shape[0],-1)), 
   #                           targets[:,1*no_depth_levels:2*no_depth_levels,:,:].reshape((predictions.shape[0],-1)), 
   #                           rowvar=False ).reshape((predictions.shape[0],no_depth_levels,predictions.shape[2],predictions.shape[3]))
   #nc_V_CC[:,:,:]   = np.corrcoef( predictions[:,2*no_depth_levels:3*no_depth_levels,:,:].reshape((predictions.shape[0],-1)), 
   #                           targets[:,2*no_depth_levels:3*no_depth_levels,:,:].reshape((predictions.shape[0],-1)), 
   #                           rowvar=False ).reshape((predictions.shape[0],no_depth_levels,predictions.shape[2],predictions.shape[3]))
   #nc_EtaCC[:,:,:]  = np.corrcoef( predictions[:,3*no_depth_levels,:,:].reshape((predictions.shape[0],-1)), 
   #                           targets[:,3*no_depth_levels,:,:].reshape((predictions.shape[0],-1)), 
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
   #        nc_U_CC[z,y,x]   = np.corrcoef( predictions[:,1*no_depth_levels+z,y,x], targets[:,1*no_depth_levels+z,y,x], rowvar=False )[0,1]
   #        nc_V_CC[z,y,x]   = np.corrcoef( predictions[:,2*no_depth_levels+z,y,x], targets[:,2*no_depth_levels+z,y,x], rowvar=False )[0,1]
   #     nc_EtaCC[y,x]  = np.corrcoef( predictions[:,3*no_depth_levels,y,x], targets[:,3*no_depth_levels,y,x], rowvar=False )[0,1]
   
