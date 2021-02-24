#!/usr/bin/env pyton
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

sys.path.append('.')
from WholeGridNetworkRegressorModules import *
from Models import CreateModel

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
#-------------------------------------
# Manually set variables for this run
#--------------------------------------
TEST = False
CalculatingMeanStd = False
TrainingModel      = True 
LoadSavedModel = False 
saved_epochs = 0

name_prefix = ''

#model_style = 'ScherStyle'
model_style = 'UNet'

hyper_params = {
    "batch_size": 32,
    "num_epochs": 10, # amount of new epochs to train, so added to any already saved.
    "learning_rate": 0.0001,  # might need further tuning, but note loss increases on first pass with 0.001
    "criterion": torch.nn.MSELoss(),
    }

#subsample_rate = 5      # number of time steps to skip over when creating training and test data
subsample_rate = 100    # number of time steps to skip over when creating training and test data
train_end_ratio = 0.75  # Take training samples from 0 to this far through the dataset
val_end_ratio = 0.9     # Take validation samples from train_end_ratio to this far through the dataset
#subsample_rate = 10000   # number of time steps to skip over when creating training and test data
#train_end_ratio = 0.5   # Take training samples from 0 to this far through the dataset
#val_end_ratio = 0.99    # Take validation samples from train_end_ratio to this far through the dataset

plot_freq = 250    # Plot scatter plot every n epochs
save_freq = 250
#-----------------------------------------------------------------------------
# Set additional variables etc, these lines should not change between runs...
#-----------------------------------------------------------------------------
reproducible = True
seed_value = 12321

#DIR =  '/data/hpcdata/users/racfur/MITgcm/verification/MundayChannelConfig10km_SmallDomain/runs/100yrs/'
DIR = '/nfs/st01/hpc-cmih-cbs31/raf59/MITgcm_Channel_Data/'
MITGCM_filename = DIR+'daily_ave_50yrs.nc'
dataset_end_index = 50*360  # Look at 50 yrs of data

model_name = name_prefix+model_style+'_lr'+str(hyper_params['learning_rate'])

# Overwrite certain variables if testing code and set up test bits
if TEST:
   hyper_params['batch_size'] = 16
   hyper_params['num_epochs'] = 1
   subsample_rate = 1000 # Keep datasets small when in testing mode
   model_name = model_name + '_TEST'

output_filename = '../../../Channel_nn_Outputs/MODELS/'+model_name+'_Output.txt'
output_file = open(output_filename, 'w', buffering=1)

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

TimeCheck(tic, output_file, 'setting variables')

#-----------------------------------
# Calulate, or read in mean and std
#-----------------------------------
if CalculatingMeanStd:
   no_channels = CalcMeanStd(model_name, MITGCM_filename, train_end_ratio, subsample_rate, dataset_end_index, hyper_params["batch_size"], output_file)

data_mean, data_std, data_range, no_channels = ReadMeanStd(model_name, output_file = output_file)

TimeCheck(tic, output_file, 'getting mean & std')

#-------------------------------------------------------------------------------------
# Create training and valdiation Datsets and dataloaders with normalisation
#-------------------------------------------------------------------------------------
# Note normalisation is carries out channel by channel, over the inputs and targets, using mean and std from training data
Train_Dataset = rr.MITGCM_Wholefield_Dataset( MITGCM_filename, 0.0, train_end_ratio, subsample_rate, dataset_end_index, 
                                              transform = transforms.Compose( [ rr.RF_Normalise(data_mean, data_std, data_range)] ) )
Val_Dataset   = rr.MITGCM_Wholefield_Dataset( MITGCM_filename, train_end_ratio, val_end_ratio, subsample_rate, dataset_end_index, 
                                              transform = transforms.Compose( [ rr.RF_Normalise(data_mean, data_std, data_range)] ) )
#Train_Dataset = rr.MITGCM_Wholefield_Dataset( MITGCM_filename, 0.0, train_end_ratio, subsample_rate, dataset_end_index )
#Val_Dataset   = rr.MITGCM_Wholefield_Dataset( MITGCM_filename, train_end_ratio, val_end_ratio, subsample_rate, dataset_end_index )
train_loader = torch.utils.data.DataLoader(Train_Dataset, batch_size=hyper_params["batch_size"], shuffle=True )
val_loader   = torch.utils.data.DataLoader(Val_Dataset,  batch_size=hyper_params["batch_size"], shuffle=True )

TimeCheck(tic, output_file, 'reading training & validation data')

#--------------
# Set up model
#--------------
h, optimizer = CreateModel(model_style, no_channels, no_channels, hyper_params["learning_rate"], reproducible, seed_value)

TimeCheck(tic, output_file, 'setting up model')

#------------------
# Train or load the model:
#------------------
print('')
for name, size in sorted(((name, sys.getsizeof(value)) for name, value in locals().items()),
                         key= lambda x: -x[1])[:10]:
    print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))
print('')

if LoadSavedModel:
   if TrainingModel:
      LoadModel(model_name, h, optimizer, saved_epochs, 'tr')
      TrainModel(model_name, output_file, tic, TEST, no_tr_samples, no_val_samples, plot_freq, save_freq,
                 train_loader, val_loader, h, optimizer, hyper_params, reproducible, seed_value, start_epoch=saved_epochs+1)
   else:
      LoadModel(model_name, h, optimizer, saved_epochs, 'inf')
elif TrainingModel:  # Training mode BUT NOT loading model!
   TrainModel(model_name, output_file, tic, TEST, no_tr_samples, no_val_samples, plot_freq, save_freq,
              train_loader, val_loader, h, optimizer, hyper_params, reproducible, seed_value)

TimeCheck(tic, output_file, 'training model')

#------------------
# Assess the model 
#------------------
#OutputSinglePrediction(model_name, Train_Dataset, h, saved_epochs+hyper_params['num_epochs'])

OutputStats(model_name, train_loader, h, saved_epochs+hyper_params['num_epochs'])

TimeCheck(tic, output_file, 'assessing model')

TimeCheck(tic, output_file, 'script')

output_file.close()
