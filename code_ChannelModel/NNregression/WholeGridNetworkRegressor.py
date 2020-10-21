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


import time as time

print('packages imported')
print('matplotlib backend is: \n')
print(matplotlib.get_backend()+'\n')
print("os.environ['DISPLAY']: \n")
print(os.environ['DISPLAY']+'\n')


torch.cuda.empty_cache()
tic = time.time()
#-------------------- -----------------
# Manually set variables for this run
#--------------------------------------
TEST = True

hyper_params = {
    "batch_size": 32,
    "num_epochs":  1,
    "learning_rate": 0.001,
    "criterion": torch.nn.MSELoss(),
}

model_type = 'nn'
time_step = '24hrs'

model_name = 'ScherStyleCNNNetwork_lr'+str(hyper_params['learning_rate'])+'_batchsize'+str(hyper_params['batch_size'])

###subsample_rate = 5      # number of time steps to skip over when creating training and test data
subsample_rate = 500    # number of time steps to skip over when creating training and test data
train_end_ratio = 0.7   # Take training samples from 0 to this far through the dataset
val_end_ratio = 0.9     # Take validation samples from train_end_ratio to this far through the dataset

DIR =  '/data/hpcdata/users/racfur/MITgcm/verification/MundayChannelConfig10km_SmallDomain/runs/100yrs/'
#DIR = '/nfs/st01/hpc-cmih-cbs31/raf59/MITgcm_Channel_Data/'
MITGCM_filename = DIR+'daily_ave_50yrs.nc'
dataset_end_index = 50*360  # Look at 50 yrs of data

plot_freq = 10    # Plot scatter plot every n epochs
seed_value = 12321


#-------------------------------------------
# Overwrite certain variables if testing code and set up test bits
if TEST:
   hyper_params['num_epochs'] = 1
   subsample_rate = 1500 # Keep datasets small when in testing mode

if TEST:
   output_filename = '../../../Channel_'+model_type+'_Outputs/MODELS/'+model_name+'_TestOutput.txt'
else:
   output_filename = '../../../Channel_'+model_type+'_Outputs/MODELS/'+model_name+'_Output.txt'
output_file = open(output_filename, 'w', buffering=1)

plot_dir = '../../../Channel_'+model_type+'_Outputs/PLOTS/'+model_name
if not os.path.isdir(plot_dir):
   os.system("mkdir %s" % (plot_dir))

ds       = xr.open_dataset(MITGCM_filename)
tr_start = 0
tr_end   = int(train_end_ratio * dataset_end_index)
val_end  = int(val_end_ratio * dataset_end_index)
x_dim    = ( ds.isel( T=slice(0) ) ).sizes['X']
y_dim    = ( ds.isel( T=slice(0) ) ).sizes['Y']

no_tr_samples = int(tr_end/subsample_rate)+1
no_val_samples = int((val_end - tr_end)/subsample_rate)+1
print('no_input_samples ;'+str(no_tr_samples)+'\n')
print('no_target_samples ;'+str(no_val_samples)+'\n')
output_file.write('no_input_samples ;'+str(no_tr_samples)+'\n')
output_file.write('no_target_samples ;'+str(no_val_samples)+'\n')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using device:'+device+'\n')
output_file.write('Using device:'+device+'\n')

toc = time.time()
print('Finished setting variables at {:0.4f} seconds'.format(toc - tic)+'\n')
output_file.write('Finished setting variables at {:0.4f} seconds'.format(toc - tic)+'\n')

#------------------------------------------------------------------------------------------
# Set up Dataset and dataloader to run through one epoch of data to obtain estimate of mean
# and s.d. for normalisation, and to get no. of features.
#------------------------------------------------------------------------------------------

mean_std_Dataset = rr.MITGCM_Wholefield_Dataset( MITGCM_filename, 0.0, train_end_ratio, subsample_rate, dataset_end_index, transform=None)
mean_std_loader = data.DataLoader(mean_std_Dataset,batch_size=hyper_params["batch_size"], shuffle=False, num_workers=0)

inputs_train_mean = []
inputs_train_std = []
targets_train_mean = []
targets_train_std = []

for batch_sample in mean_std_loader:
    # shape (batch_size, channels, y, z)
    input_batch  = batch_sample['input']
    target_batch = batch_sample['output']
    
    inputs_batch_mean  = np.nanmean(input_batch.numpy(), axis = (0,2,3))
    targets_batch_mean = np.nanmean(target_batch.numpy(), axis = (0,2,3))

    inputs_batch_std  = np.nanstd(input_batch.numpy() , axis = (0,2,3))
    targets_batch_std = np.nanstd(target_batch.numpy(), axis = (0,2,3))
    
    inputs_train_mean.append(inputs_batch_mean)
    inputs_train_std.append(inputs_batch_std)

    targets_train_mean.append(targets_batch_mean)
    targets_train_std.append(targets_batch_std)

inputs_train_mean = np.array(inputs_train_mean).mean(axis=0)
inputs_train_std = np.array(inputs_train_std).mean(axis=0)
targets_train_mean = np.array(targets_train_mean).mean(axis=0)
targets_train_std = np.array(targets_train_std).mean(axis=0)

## Save mean and std to file, so can be used to un-normalise when using model to predict
mean_std_file = '../../../INPUT_OUTPUT_ARRAYS/Channel_Whole_Grid_MeanStd.npz'
np.savez( mean_std_file, inputs_train_mean, inputs_train_std, targets_train_mean, targets_train_std )

no_input_channels = input_batch.shape[1]
no_target_channels = target_batch.shape[1]

print('no_input_channels ;'+str(no_input_channels)+'\n')
print('no_target_channels ;'+str(no_target_channels)+'\n')
output_file.write('no_input_channels ;'+str(no_input_channels)+'\n')
output_file.write('no_target_channels ;'+str(no_target_channels)+'\n')

toc = time.time()
print('Finished calculating mean and std at {:0.4f} seconds'.format(toc - tic)+'\n')
output_file.write('Finished calculating mean and std at {:0.4f} seconds'.format(toc - tic)+'\n')

#-------------------------------------------------------------------------------------
# Create training and valdiation Datsets and dataloaders this time with normalisation
#-------------------------------------------------------------------------------------
Train_Dataset = rr.MITGCM_Wholefield_Dataset( MITGCM_filename, 0.0, train_end_ratio, subsample_rate, dataset_end_index, 
                                              transform = transforms.Compose( [ rr.RF_Normalise(inputs_train_mean, inputs_train_std, targets_train_mean, targets_train_std)] ) )

Val_Dataset   = rr.MITGCM_Wholefield_Dataset( MITGCM_filename, train_end_ratio, val_end_ratio, subsample_rate, dataset_end_index, 
                                              transform = transforms.Compose( [ rr.RF_Normalise(inputs_train_mean, inputs_train_std, targets_train_mean, targets_train_std)] ) )

train_loader = torch.utils.data.DataLoader(Train_Dataset, batch_size=hyper_params["batch_size"])
val_loader   = torch.utils.data.DataLoader(Val_Dataset,  batch_size=hyper_params["batch_size"])

toc = time.time()
print('Finished reading in training and validation data at {:0.4f} seconds'.format(toc - tic)+'\n')
output_file.write('Finished reading in training and validation data at {:0.4f} seconds'.format(toc - tic)+'\n')

#--------------
# Set up model
#--------------
# inputs are (no_samples, 169channels, 78y, 11x).
# Not doing pooling, as dataset small to begin with, and no computational need, or scientific justification....
h = nn.Sequential(
            # downscale
            nn.Conv2d(in_channels=no_input_channels, out_channels=256, kernel_size=(3,3), padding=(1,1)),
            nn.ReLU(True),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3,3),padding=(1,1)),
            nn.ReLU(True),

            # upscale
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=(3,3), padding=(1,1)),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=256, out_channels=no_target_channels, kernel_size=(3,3), padding=(1,1)),
            nn.ReLU(True)
             )
h = h.cuda()

optimizer = torch.optim.Adam( h.parameters(), lr=hyper_params["learning_rate"] )

toc = time.time()
print('Finished Setting up model at {:0.4f} seconds'.format(toc - tic)+'\n')
output_file.write('Finished Setting up model at {:0.4f} seconds'.format(toc - tic)+'\n')

#------------------
# Train the model:
#------------------

train_loss_batch = []
train_loss_epoch = []
val_loss_epoch = []
first_pass = True

for epoch in range(hyper_params["num_epochs"]):

    # Training

    train_loss_epoch_temp = 0.0

    if epoch%plot_freq == 0:
        fig = plt.figure(figsize=(9,9))
        ax1 = fig.add_subplot(111)
        bottom = 0.
        top = 0.

    for batch_sample in train_loader:
        input_batch  = batch_sample['input']
        target_batch = batch_sample['output']
        # Converting inputs and labels to Variable and send to GPU if available
        if torch.cuda.is_available():
            input_batch = Variable(input_batch.cuda().float())
            target_batch = Variable(target_batch.cuda().float())
        else:
            inputs = Variable(inputs.float())
            targets = Variable(targets.float())
        
        h.train(True)

        # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
        optimizer.zero_grad()
    
        # get prediction from the model, given the inputs
        predicted = h(input_batch)

        if TEST and first_pass:
           print('For test run check if output is correct shape, the following two shapes should match!\n')
           print('target_batch.shape for training data: '+str(target_batch.shape)+'\n')
           print('predicted.shape for training data: '+str(predicted.shape)+'\n')
           output_file.write('For test run check if output is correct shape, the following two shapes should match!\n')
           output_file.write('target_batch.shape for training data: '+str(target_batch.shape)+'\n')
           output_file.write('predicted.shape for training data: '+str(predicted.shape)+'\n')
           
        # get loss for the predicted output
        pre_opt_loss = hyper_params["criterion"](predicted, target_batch)
        
        # get gradients w.r.t to parameters
        pre_opt_loss.backward()
     
        # update parameters
        optimizer.step()

        if TEST:  # Recalculate loss post optimizer step to ensure its decreased
           h.train(False)
           post_opt_predicted = h(input_batch)
           post_opt_loss = hyper_params["criterion"](post_opt_predicted, target_batch)
           print('For test run check if loss decreases over a single training step on a batch of data\n')
           print('loss pre optimization: ' + str(pre_opt_loss.item()) + '\n')
           print('loss post optimization: ' + str(post_opt_loss.item()) + '\n')
           output_file.write('For test run check if loss decreases over a single training step on a batch of data\n')
           output_file.write('loss pre optimization: ' + str(pre_opt_loss.item()) + '\n')
           output_file.write('loss post optimization: ' + str(post_opt_loss.item()) + '\n')
        
        train_loss_batch.append( pre_opt_loss.item() )
        train_loss_epoch_temp += pre_opt_loss.item()*input_batch.shape[0] 

        if epoch%plot_freq == 0:
            target_batch = target_batch.cpu().detach().numpy()
            predicted = predicted.cpu().detach().numpy()
            ax1.scatter(target_batch, predicted, edgecolors=(0, 0, 0), alpha=0.15)
            bottom = min(np.amin(predicted), np.amin(target_batch), bottom)
            top    = max(np.amax(predicted), np.amax(target_batch), top)  

    tr_loss_single_epoch = train_loss_epoch_temp / no_tr_samples
    print('epoch {}, training loss {}'.format( epoch, tr_loss_single_epoch )+'\n')
    output_file.write('epoch {}, training loss {}'.format( epoch, tr_loss_single_epoch )+'\n')
    train_loss_epoch.append( tr_loss_single_epoch )

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
    
    if epoch%plot_freq == 0:
        fig = plt.figure(figsize=(9,9))
        ax1 = fig.add_subplot(111)
        bottom = 0.
        top = 0.

    for batch_sample in val_loader:
        input_batch  = batch_sample['input']
        target_batch = batch_sample['output']
    
        # Converting inputs and labels to Variable and send to GPU if available
        if torch.cuda.is_available():
            input_batch = Variable(input_batch.cuda().float())
            target_batch = Variable(target_batch.cuda().float())
        else:
            input_batch = Variable(input_batch.float())
            target_batch = Variable(target_batch.float())
        
        h.train(False)

        # get prediction from the model, given the inputs
        predicted = h(input_batch)
    
        # get loss for the predicted output
        loss = hyper_params["criterion"](predicted, target_batch)

        val_loss_epoch_temp += loss.item()*input_batch.shape[0]

        if epoch%plot_freq == 0:
            target_batch = target_batch.cpu().detach().numpy()
            predicted = predicted.cpu().detach().numpy()
            ax1.scatter(target_batch, predicted, edgecolors=(0, 0, 0), alpha=0.15)
            bottom = min(np.amin(predicted), np.amin(target_batch), bottom)
            top    = max(np.amax(predicted), np.amax(target_batch), top)    

    val_loss_single_epoch = val_loss_epoch_temp / no_val_samples
    print('epoch {}, validation loss {}'.format(epoch, val_loss_single_epoch)+'\n')
    output_file.write('epoch {}, validation loss {}'.format(epoch, val_loss_single_epoch)+'\n')
    val_loss_epoch.append(val_loss_single_epoch)

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

    #first_pass = False

del input_batch, target_batch, predicted 
torch.cuda.empty_cache()

toc = time.time()
print('Finished training model at {:0.4f} seconds'.format(toc - tic)+'\n')
output_file.write('Finished training model at {:0.4f} seconds'.format(toc - tic)+'\n')

#if TEST: # Output the predictions for a single input into a netcdf so can check for data ranges etc and ensure it looks sensible
#   print('Train_Dataset :')
#   print(Train_Dataset)
#   test_loader = torch.utils.data.DataLoader(Train_Dataset, batch_size=1)
#   for batch_sample in test_loader:
#       input_batch  = batch_sample['input'][0,:,:,:]
#       target_batch = batch_sample['output'][0,:,:,:]
#    
#       # Converting inputs and labels to Variable and send to GPU if available
#       if torch.cuda.is_available():
#           input_batch = Variable(input_batch.cuda().float())
#           target_batch = Variable(target_batch.cuda().float())
#       else:
#           input_batch = Variable(input_batch.float())
#           target_batch = Variable(target_batch.float())
#
#       input_batch.reshape[1,input_batch.shape[0],input_batch.shape[1],input_batch.shape[2]]
#       target_batch.reshape[1,target_batch.shape[0],target_batch.shape[1],target_batch.shape[2]]
#       print('input_batch.shape: ')
#       print(input_batch.shape)
#       print('target_batch.shape: ')
#       print(target_batch.shape)
#       
#       h.train(False)
#
#       # get prediction from the model, given the inputs
#       predicted = h(input_batch)
#   print(predicted.shape)
#
#   toc = time.time()
#   print('Finished creating TEST output ncfile at {:0.4f} seconds'.format(toc - tic)+'\n')
#   output_file.write('Finished creating TEST output ncfile at {:0.4f} seconds'.format(toc - tic)+'\n')
 
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
info_file.write('no_tr_samples:   '+str(len(Train_Dataset))+'\n')
info_file.write('no_val_samples:  '+str(len(Val_Dataset))+'\n')
info_file.write('no_input_channels:   '+str(no_input_channels)+'\n')
info_file.write('no_target_channels:  '+str(no_target_channels)+'\n')
info_file.write('\n')
info_file.write('\n')
info_file.write('Weights of the network:\n')
info_file.write(str(h[0].weight.data.cpu().numpy()))

print('pickle model \n')
output_file.write('pickle model \n')
pkl_filename = '../../../Channel_nn_Outputs/MODELS/'+model_name+'_pickle.pkl'
with open(pkl_filename, 'wb') as pckl_file:
    pickle.dump(h, pckl_file)

toc = time.time()
print('Finished script at {:0.4f} seconds'.format(toc - tic)+'\n')
output_file.write('Finished script at {:0.4f} seconds'.format(toc - tic)+'\n')

output_file.close()
