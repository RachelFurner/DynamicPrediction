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

#add this rotuine to memory profile...having memory issues!
def sizeof_fmt(num, suffix='B'):
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)

print('packages imported')
print('matplotlib backend is: ')
print(matplotlib.get_backend())
print("os.environ['DISPLAY']: ")
print(os.environ['DISPLAY'])
print('torch.__version__ :')
print(torch.__version__)
#

torch.cuda.empty_cache()
tic = time.time()
#-------------------- -----------------
# Manually set variables for this run
#--------------------------------------
TEST = False

hyper_params = {
    "batch_size":  32,
    "num_epochs":  40,
    "learning_rate": 0.0001,  # might need further tuning, but note loss increases on first pass with 0.001
    "criterion": torch.nn.MSELoss(),
}

model_type = 'nn'
time_step = '24hrs'

model_name = 'ScherStyleCNNNetwork_lr'+str(hyper_params['learning_rate'])+'_batchsize'+str(hyper_params['batch_size'])

subsample_rate = 25     # number of time steps to skip over when creating training and test data
train_end_ratio = 0.75  # Take training samples from 0 to this far through the dataset
val_end_ratio = 0.9     # Take validation samples from train_end_ratio to this far through the dataset

#DIR =  '/data/hpcdata/users/racfur/MITgcm/verification/MundayChannelConfig10km_SmallDomain/runs/100yrs/'
DIR = '/nfs/st01/hpc-cmih-cbs31/raf59/MITgcm_Channel_Data/'
MITGCM_filename = DIR+'daily_ave_50yrs.nc'
dataset_end_index = 50*360  # Look at 50 yrs of data

plot_freq = 100    # Plot scatter plot every n epochs

reproducible = True
seed_value = 12321

#-----------------------------------------------------------------------------
# Set additional variables etc, these lines should not change between runs...
#-----------------------------------------------------------------------------

# Overwrite certain variables if testing code and set up test bits
if TEST:
   hyper_params['batch_size'] = 16
   hyper_params['num_epochs'] = 1
   subsample_rate = 500 # Keep datasets small when in testing mode

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
print('no_training_samples ;'+str(no_tr_samples)+'\n')
print('no_validation_samples ;'+str(no_val_samples)+'\n')
output_file.write('no_training_samples ;'+str(no_tr_samples)+'\n')
output_file.write('no_validation_samples ;'+str(no_val_samples)+'\n')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using device:'+device+'\n')
output_file.write('Using device:'+device+'\n')

print('Batch Size: '+str(hyper_params['batch_size'])+'\n')
output_file.write('Batch Size: '+str(hyper_params['batch_size'])+'\n')

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
mean_std_file = '../../../INPUT_OUTPUT_ARRAYS/Channel_Whole_Grid_'+model_name+'_MeanStd.npz'
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

del batch_sample
del input_batch
del target_batch

#-------------------------------------------------------------------------------------
# Create training and valdiation Datsets and dataloaders this time with normalisation
#-------------------------------------------------------------------------------------
# Note normalisation is carries out channel by channel, over the inputs and targets, using mean and std from training data
Train_Dataset = rr.MITGCM_Wholefield_Dataset( MITGCM_filename, 0.0, train_end_ratio, subsample_rate, dataset_end_index, 
                                              transform = transforms.Compose( [ rr.RF_Normalise(inputs_train_mean, inputs_train_std, targets_train_mean, targets_train_std)] ) )

Val_Dataset   = rr.MITGCM_Wholefield_Dataset( MITGCM_filename, train_end_ratio, val_end_ratio, subsample_rate, dataset_end_index, 
                                              transform = transforms.Compose( [ rr.RF_Normalise(inputs_train_mean, inputs_train_std, targets_train_mean, targets_train_std)] ) )

train_loader = torch.utils.data.DataLoader(Train_Dataset, batch_size=hyper_params["batch_size"], shuffle=True )
val_loader   = torch.utils.data.DataLoader(Val_Dataset,  batch_size=hyper_params["batch_size"], shuffle=True )

toc = time.time()
print('Finished reading in training and validation data at {:0.4f} seconds'.format(toc - tic)+'\n')
output_file.write('Finished reading in training and validation data at {:0.4f} seconds'.format(toc - tic)+'\n')

#--------------
# Set up model
#--------------
# inputs are (no_samples, 153channels, 108y, 240x).
# Not doing pooling, as dataset small to begin with, and no computational need, or scientific justification....
h = nn.Sequential(
            # downscale
            # ****Note scher uses a kernel size of 6, but no idea how he then manages to keep his x and y dimensions
            # unchanged over the conv layers....***
            nn.Conv2d(in_channels=no_input_channels, out_channels=150, kernel_size=(7,7), padding=(3,3)),
            nn.ReLU(True),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(in_channels=150, out_channels=150, kernel_size=(7,7),padding=(3,3)),
            nn.ReLU(True),
            nn.MaxPool2d((2,2)),

            nn.ConvTranspose2d(in_channels=150, out_channels=150, kernel_size=(7,7), padding=(3,3)),
            nn.ReLU(True),

            # upscale
            nn.Upsample(scale_factor=(2,2)),
            nn.Conv2d(in_channels=150, out_channels=150, kernel_size=(7,7), padding=(3,3)),
            nn.ReLU(True),
            nn.Upsample(scale_factor=(2,2)),
            nn.Conv2d(in_channels=150, out_channels=no_target_channels, kernel_size=(7,7), padding=(3,3)),
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
train_loss_epoch = []
val_loss_epoch = []
first_pass = True
torch.cuda.empty_cache()

print('')
for name, size in sorted(((name, sys.getsizeof(value)) for name, value in locals().items()),
                         key= lambda x: -x[1])[:10]:
    print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))
#print(torch.cuda.memory_summary(device))
print('')

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

toc = time.time()
print('Finished training model at {:0.4f} seconds'.format(toc - tic)+'\n')
output_file.write('Finished training model at {:0.4f} seconds'.format(toc - tic)+'\n')

#-----------------------------------------------------------------------------------------
# Output the predictions for a single input into a netcdf file to check it looks sensible
#-----------------------------------------------------------------------------------------

#Make predictions for a single sample
sample = Train_Dataset.__getitem__(5)
input_sample  = sample['input'][:,:,:].unsqueeze(0)
target_sample = sample['output'][:,:,:].unsqueeze(0)
if torch.cuda.is_available():
    input_sample = Variable(input_sample.cuda().float())
    target_sample = Variable(target_sample.cuda().float())
else:
    input_sample = Variable(input_sample.float())
    target_sample = Variable(target_sample.float())
h.train(False)
predicted = h(input_sample)
predicted = predicted.data.cpu().numpy()
target_sample = target_sample.data.cpu().numpy()

# Denormalise
target_sample = rr.RF_DeNormalise(target_sample, targets_train_mean, targets_train_std)
predicted     = rr.RF_DeNormalise(predicted    , targets_train_mean, targets_train_std)

# Set up netcdf files
nc_filename = '../../../Channel_'+model_type+'_Outputs/MODELS/'+model_name+'_DummyOutput.nc'
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
nc_TrueTemp   = nc_file.createVariable('True_Temp'  , 'f4', ('Z', 'Y', 'X'))
nc_PredTemp   = nc_file.createVariable('Pred_Temp'  , 'f4', ('Z', 'Y', 'X'))
nc_TempErrors = nc_file.createVariable('Temp_Errors', 'f4', ('Z', 'Y', 'X'))
nc_TrueSal    = nc_file.createVariable('True_Sal'   , 'f4', ('Z', 'Y', 'X'))
nc_PredSal    = nc_file.createVariable('Pred_Sal'   , 'f4', ('Z', 'Y', 'X'))
nc_SalErrors  = nc_file.createVariable('Sal_Errors' , 'f4', ('Z', 'Y', 'X'))
nc_TrueU      = nc_file.createVariable('True_U'     , 'f4', ('Z', 'Y', 'X'))
nc_PredU      = nc_file.createVariable('Pred_U'     , 'f4', ('Z', 'Y', 'X'))
nc_UErrors    = nc_file.createVariable('U_Errors'   , 'f4', ('Z', 'Y', 'X'))
nc_TrueV      = nc_file.createVariable('True_V'     , 'f4', ('Z', 'Y', 'X'))
nc_PredV      = nc_file.createVariable('Pred_V'     , 'f4', ('Z', 'Y', 'X'))
nc_VErrors    = nc_file.createVariable('V_Errors'   , 'f4', ('Z', 'Y', 'X'))
nc_TrueEta    = nc_file.createVariable('True_Eta'   , 'f4', ('Y', 'X'))
nc_PredEta    = nc_file.createVariable('Pred_Eta'   , 'f4', ('Y', 'X'))
nc_EtaErrors  = nc_file.createVariable('Eta_Errors' , 'f4', ('Y', 'X'))
# Fill variables
nc_Z[:] = np.arange(no_depth_levels)
nc_Y[:] = np.arange(target_sample.shape[2])
nc_X[:] = np.arange(target_sample.shape[3])
nc_TrueTemp[:,:,:]   = target_sample[0,0:no_depth_levels,:,:] 
nc_PredTemp[:,:,:]   = predicted[0,0:no_depth_levels,:,:]
nc_TempErrors[:,:,:] = predicted[0,0:no_depth_levels,:,:] - target_sample[0,0:no_depth_levels,:,:]
nc_TrueSal[:,:,:]    = target_sample[0,no_depth_levels:2*no_depth_levels,:,:] 
nc_PredSal[:,:,:]    = predicted[0,no_depth_levels:2*no_depth_levels,:,:]
nc_SalErrors[:,:,:]  = predicted[0,no_depth_levels:2*no_depth_levels,:,:] - target_sample[0,no_depth_levels:2*no_depth_levels,:,:]
nc_TrueU[:,:,:]      = target_sample[0,2*no_depth_levels:3*no_depth_levels,:,:]
nc_PredU[:,:,:]      = predicted[0,2*no_depth_levels:3*no_depth_levels,:,:]
nc_UErrors[:,:,:]    = predicted[0,2*no_depth_levels:3*no_depth_levels,:,:] - target_sample[0,2*no_depth_levels:3*no_depth_levels,:,:]
nc_TrueV[:,:,:]      = target_sample[0,3*no_depth_levels:4*no_depth_levels,:,:]
nc_PredV[:,:,:]      = predicted[0,3*no_depth_levels:4*no_depth_levels,:,:]
nc_VErrors[:,:,:]    = predicted[0,3*no_depth_levels:4*no_depth_levels,:,:] - target_sample[0,3*no_depth_levels:4*no_depth_levels,:,:]
nc_TrueEta[:,:]      = target_sample[0,4*no_depth_levels,:,:]
nc_PredEta[:,:]      = predicted[0,4*no_depth_levels,:,:]
nc_EtaErrors[:,:]    = predicted[0,4*no_depth_levels,:,:] - target_sample[0,4*no_depth_levels,:,:]

toc = time.time()
print('Finished creating dummy output ncfile at {:0.4f} seconds'.format(toc - tic)+'\n')
output_file.write('Finished creating dummy output ncfile at {:0.4f} seconds'.format(toc - tic)+'\n')
 
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

print('save model \n')
output_file.write('save model \n')
pkl_filename = '../../../Channel_nn_Outputs/MODELS/'+model_name+'_SavedModel.pt'
torch.save(h.state_dict(), pkl_filename)

toc = time.time()
print('Finished script at {:0.4f} seconds'.format(toc - tic)+'\n')
output_file.write('Finished script at {:0.4f} seconds'.format(toc - tic)+'\n')

output_file.close()
