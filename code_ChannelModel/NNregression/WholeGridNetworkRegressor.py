#!/usr/bin/env python
# coding: utf-8

from comet_ml import Experiment

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

torch.cuda.empty_cache()
tic = time.time()
#-------------------- -----------------
# Manually set variables for this run
#--------------------------------------
hyper_params = {
    "batch_size": 512,
    "num_epochs":  2,
    "learning_rate": 0.0001,
    "criterion": torch.nn.MSELoss(),
}

model_type = 'nn'
time_step = '24hrs'

model_name = 'ScherStyleCNNNetwork_lr'+str(hyper_params['learning_rate'])+'_batchsize'+str(hyper_params['batch_size'])

###subsample_rate = 5      # number of time steps to skip over when creating training and test data
subsample_rate = 20     # number of time steps to skip over when creating training and test data
train_end_ratio = 0.7   # Take training samples from 0 to this far through the dataset
val_end_ratio = 0.9     # Take validation samples from train_end_ratio to this far through the dataset

DIR = '/data/hpcdata/users/racfur/MITgcm/verification/MundayChannelConfig10km_SmallDomain/runs/100yrs/'
MITGCM_filename = DIR+'daily_ave_50yrs.nc'
dataset_end_index = 50*360  # Look at 50 yrs of data

plot_freq = 1    # Plot scatter plot every n epochs
seed_value = 12321

#-------------------------------------------
output_filename = '../../../Channel_'+model_type+'_Outputs/MODELS/'+model_name+'_output.txt'
output_file = open(output_filename, 'w', buffering=1)

plot_dir = '../../../Channel_'+model_type+'_Outputs/PLOTS/'+model_name
if not os.path.isdir(plot_dir):
   os.system("mkdir %s" % (plot_dir))

ds       = xr.open_dataset(MITGCM_filename)
tr_start = 0
tr_end   = int(train_end_ratio * dataset_end_index)
val_end   = int(val_end_ratio * dataset_end_index)
x_dim         = ( ds.isel( T=slice(0) ) ).sizes['X']
y_dim         = ( ds.isel( T=slice(0) ) ).sizes['Y']

no_tr_samples = tr_end/subsample_rate
no_val_samples = (val_end - tr_end)/subsample_rate
output_file.write('no_input_samples ;'+str(no_tr_samples)+'\n')
output_file.write('no_output_samples ;'+str(no_val_samples)+'\n')

output_file.write('matplotlib backend is: \n')
output_file.write(matplotlib.get_backend()+'\n')
output_file.write("os.environ['DISPLAY']: \n")
output_file.write(os.environ['DISPLAY']+'\n')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
output_file.write('Using device:'+device+'\n')

toc = time.time()
output_file.write('Finished setting variables at {:0.4f} seconds'.format(toc - tic)+'\n')

#------------------------------------------------------------------------------------------
# Set up Dataset and dataloader to run through one epoch of data to obtain estimate of mean
# and s.d. for normalisation, and to get no. of features.
#------------------------------------------------------------------------------------------

mean_std_Dataset = rr.MITGCM_Wholefield_Dataset( MITGCM_filename, 0.0, train_end_ratio, subsample_rate, dataset_end_index, transform=None)
mean_std_loader = data.DataLoader(mean_std_Dataset,batch_size=hyper_params["batch_size"], shuffle=False, num_workers=0)

inputs_train_mean = []
inputs_train_std = []
outputs_train_mean = []
outputs_train_std = []

for batch_sample in mean_std_loader:
    # shape (batch_size, channels, y, z)
    input_batch  = batch_sample['input']
    output_batch = batch_sample['output']
    
    inputs_batch_mean  = np.nanmean(input_batch.numpy(), axis = (0,2,3))
    outputs_batch_mean = np.nanmean(output_batch.numpy(), axis = (0,2,3))

    inputs_batch_std  = np.nanstd(input_batch.numpy() , axis = (0,2,3))
    outputs_batch_std = np.nanstd(output_batch.numpy(), axis = (0,2,3))
    
    inputs_train_mean.append(inputs_batch_mean)
    inputs_train_std.append(inputs_batch_std)

    outputs_train_mean.append(outputs_batch_mean)
    outputs_train_std.append(outputs_batch_std)

inputs_train_mean = np.array(inputs_train_mean).mean(axis=0)
inputs_train_std = np.array(inputs_train_std).mean(axis=0)
outputs_train_mean = np.array(outputs_train_mean).mean(axis=0)
outputs_train_std = np.array(outputs_train_std).mean(axis=0)

## Save mean and std to file, so can be used to un-normalise when using model to predict
mean_std_file = '../../../INPUT_OUTPUT_ARRAYS/Channel_Whole_Grid_MeanStd.npz'
np.savez( mean_std_file, inputs_train_mean, inputs_train_std, outputs_train_mean, outputs_train_std )

no_input_channels = input_batch.shape[1]
no_output_channels = output_batch.shape[1]

output_file.write('no_input_channels ;'+str(no_input_channels)+'\n')
output_file.write('no_output_channels ;'+str(no_output_channels)+'\n')

#-------------------------------------------------------------------------------------
# Create training and valdiation Datsets and dataloaders this time with normalisation
#-------------------------------------------------------------------------------------
Train_Dataset = rr.MITGCM_Wholefield_Dataset( MITGCM_filename, 0.0, train_end_ratio, subsample_rate, dataset_end_index, 
                                              transform = transforms.Compose( [ rr.RF_Normalise(inputs_train_mean, inputs_train_std, outputs_train_mean, outputs_train_std)] ) )

Val_Dataset   = rr.MITGCM_Wholefield_Dataset( MITGCM_filename, train_end_ratio, val_end_ratio, subsample_rate, dataset_end_index, 
                                              transform = transforms.Compose( [ rr.RF_Normalise(inputs_train_mean, inputs_train_std, outputs_train_mean, outputs_train_std)] ) )

train_loader = torch.utils.data.DataLoader(Train_Dataset, batch_size=hyper_params["batch_size"])
val_loader   = torch.utils.data.DataLoader(Val_Dataset,  batch_size=hyper_params["batch_size"])

toc = time.time()
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
            nn.ConvTranspose2d(in_channels=256, out_channels=no_output_channels, kernel_size=(3,3), padding=(1,1)),
            nn.ReLU(True)
             )
h = h.cuda()

optimizer = torch.optim.Adam( h.parameters(), lr=hyper_params["learning_rate"] )

toc = time.time()
output_file.write('Finished Setting up model at {:0.4f} seconds'.format(toc - tic)+'\n')

#------------------
# Train the model:
#------------------

train_loss_batch = []
train_loss_epoch = []
val_loss_epoch = []

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
        output_batch = batch_sample['output']
        # Converting inputs and labels to Variable and send to GPU if available
        if torch.cuda.is_available():
            input_batch = Variable(input_batch.cuda().float())
            output_batch = Variable(output_batch.cuda().float())
        else:
            inputs = Variable(inputs.float())
            outputs = Variable(outputs.float())
        
        h.train(True)

        # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
        optimizer.zero_grad()
    
        # get prediction from the model, given the inputs
        predicted = h(input_batch)
 
        # get loss for the predicted output
        loss = hyper_params["criterion"](predicted, output_batch)
        # get gradients w.r.t to parameters
        loss.backward()
    
        # update parameters
        optimizer.step()
        
        train_loss_batch.append( loss.item() )
        train_loss_epoch_temp += loss.item()*input_batch.shape[0] 

        if epoch%plot_freq == 0:
            output_batch = output_batch.cpu().detach().numpy()
            predicted = predicted.cpu().detach().numpy()
            ax1.scatter(output_batch, predicted, edgecolors=(0, 0, 0), alpha=0.15)
            bottom = min(np.amin(predicted), np.amin(output_batch), bottom)
            top    = max(np.amax(predicted), np.amax(output_batch), top)  

    tr_loss_single_epoch = train_loss_epoch_temp / no_tr_samples
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
        output_batch = batch_sample['output']
    
        # Converting inputs and labels to Variable and send to GPU if available
        if torch.cuda.is_available():
            input_batch = Variable(input_batch.cuda().float())
            output_batch = Variable(output_batch.cuda().float())
        else:
            input_batch = Variable(input_batch.float())
            output_batch = Variable(output_batch.float())
        
        h.train(False)

        # get prediction from the model, given the inputs
        predicted = h(input_batch)
    
        # get loss for the predicted output
        loss = hyper_params["criterion"](predicted, output_batch)

        val_loss_epoch_temp += loss.item()*input_batch.shape[0]

        if epoch%plot_freq == 0:
            output_batch = output_batch.cpu().detach().numpy()
            predicted = predicted.cpu().detach().numpy()
            ax1.scatter(output_batch, predicted, edgecolors=(0, 0, 0), alpha=0.15)
            bottom = min(np.amin(predicted), np.amin(output_batch), bottom)
            top    = max(np.amax(predicted), np.amax(output_batch), top)    

    val_loss_single_epoch = val_loss_epoch_temp / no_val_samples
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
    output_file.write('Finished epoch {} at {:0.4f} seconds'.format(epoch, toc - tic)+'\n')

del input_batch, output_batch, predicted 
torch.cuda.empty_cache()

toc = time.time()
output_file.write('Finished training model at {:0.4f} seconds'.format(toc - tic)+'\n')
 
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
info_file.write('no_output_channels:  '+str(no_output_channels)+'\n')
info_file.write('\n')
info_file.write('\n')
info_file.write('Weights of the network:\n')
info_file.write(str(h[0].weight.data.cpu().numpy()))

output_file.write('pickle model \n')
pkl_filename = '../../../Channel_nn_Outputs/MODELS/'+model_name+'_pickle.pkl'
with open(pkl_filename, 'wb') as pckl_file:
    pickle.dump(h, pckl_file)

toc = time.time()
output_file.write('Finished script at {:0.4f} seconds'.format(toc - tic)+'\n')

output_file.close()
