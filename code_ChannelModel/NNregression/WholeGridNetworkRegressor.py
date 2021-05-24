#!/usr/bin/env pyton
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
import torch.autograd.profiler as profiler

import netCDF4 as nc4

import time as time
import gc

import argparse
import logging

import multiprocessing as mp
#import torch.multiprocessing
#torch.multiprocessing.set_sharing_strategy('file_system')

import GPUtil

def parse_args():
    a = argparse.ArgumentParser()

    a.add_argument("-n", "--name", default="", action='store')
    a.add_argument("-te", "--test", default=False, type=bool, action='store')
    a.add_argument("-cm", "--calculatemeanstd", default=False, type=bool, action='store')
    a.add_argument("-tr", "--trainmodel", default=True, type=bool, action='store')
    a.add_argument("-lo", "--loadmodel", default=False, type=bool, action='store')
    a.add_argument("-se", "--savedepochs", default=0, type=int, action='store')
    a.add_argument("-a", "--assess", default=True, type=bool, action='store')
    a.add_argument("-i", "--iterate", default=True, type=bool, action='store')

    a.add_argument("-d", "--dim", default='2d', type=str, action='store')
    a.add_argument("-la", "--land", default='ExcLand', type=str, action='store')
    a.add_argument("-p", "--padding", default='None', type=str, action='store')
    a.add_argument("-m", "--modelstyle", default='UNet2dtransp', type=str, action='store')

    a.add_argument("-w", "--numworkers", default=4, type=int, action='store')
    a.add_argument("-b", "--batchsize", default=32, type=int, action='store')
    a.add_argument("-e", "--epochs", default=200, type=int, action='store')
    a.add_argument("-lr", "--learningrate", default=0.0001, type=float, action='store')

    a.add_argument("-s", "--seed", default=30475, type=int, action='store')

    return a.parse_args()

if __name__ == "__main__":
    
    mp.set_start_method('spawn')

    args = parse_args()
  
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s') 

    logging.info('packages imported')
    logging.info('matplotlib backend: '+str(matplotlib.get_backend()) )
    logging.info("os.environ['DISPLAY']: "+str(os.environ['DISPLAY']))
    logging.info('torch.__version__ : '+str(torch.__version__))
    
    tic = time.time()
    #-------------------------------------
    # Manually set variables for this run
    #--------------------------------------

    logging.info('args')
    logging.info(args)
    
    hyper_params = {
        "batch_size": args.batchsize,
        "num_epochs": args.epochs, # amount of new epochs to train, so added to any already saved.
        "learning_rate": args.learningrate,  # might need further tuning, but note loss increases on first pass with 0.001
        "criterion": torch.nn.MSELoss(),
        }
    
    subsample_rate = 5      # number of time steps to skip over when creating training and test data
    train_end_ratio = 0.75  # Take training samples from 0 to this far through the dataset
    val_end_ratio = 0.9     # Take validation samples from train_end_ratio to this far through the dataset
    
    plot_freq = 25     # Plot scatter plot, and save the model every n epochs (save in case of a crash etc)
    save_freq = 10      # Plot scatter plot, and save the model every n epochs (save in case of a crash etc)
    
    for_len = 30*6   # How long to iteratively predict for
    start = 5        #
    
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    if args.test: 
       model_name = args.name+args.land+'_'+args.padding+'_'+args.modelstyle+'_lr'+str(hyper_params['learning_rate'])+'_TEST'
    else:
       model_name = args.name+args.land+'_'+args.padding+'_'+args.modelstyle+'_lr'+str(hyper_params['learning_rate'])
    MeanStd_prefix = args.land+'_'+args.dim
    
    # Overwrite certain variables if testing code and set up test bits
    if args.test:
       hyper_params['batch_size'] = 32
       hyper_params['num_epochs'] = 5
       subsample_rate = 500 # Keep datasets small when in testing mode
    
    model_dir = '../../../Channel_nn_Outputs/'+model_name
    if not os.path.isdir(model_dir):
       os.system("mkdir %s" % (model_dir))
       os.system("mkdir %s" % (model_dir+'/MODELS'))
       os.system("mkdir %s" % (model_dir+'/PLOTS'))
       os.system("mkdir %s" % (model_dir+'/STATS'))
       os.system("mkdir %s" % (model_dir+'/ITERATED_FORECAST'))
       os.system("mkdir %s" % (model_dir+'/ITERATED_FORECAST/PLOTS'))
    
    if args.trainmodel:
       if args.loadmodel: 
          start_epoch = args.savedepochs+1
          total_epochs = args.savedepochs+hyper_params['num_epochs']
       else:
          start_epoch = 0
          total_epochs = hyper_params['num_epochs']-1
    else:
       start_epoch = args.savedepochs
       total_epochs = args.savedepochs
    
    if args.land == 'IncLand':
       y_dim_used = 104
    elif args.land == 'ExcLand':
       y_dim_used = 96
    else:
       raise RuntimeError('ERROR, Whats going on with args.land?!')
    
    DIR =  '/data/hpcdata/users/racfur/MITgcm/verification/MundayChannelConfig10km_97x240Domain/runs/100yrs/'
    #DIR = '/nfs/st01/hpc-cmih-cbs31/raf59/MITgcm_Channel_Data/'
    MITGCM_filename = DIR+'daily_ave_50yrs.nc'
    dataset_end_index = 50*360  # Look at 50 yrs of data
    
    ds       = xr.open_dataset(MITGCM_filename)
    tr_start = 0
    tr_end   = int(train_end_ratio * dataset_end_index)
    val_end  = int(val_end_ratio * dataset_end_index)
    x_dim    = ( ds.isel( T=slice(0) ) ).sizes['X']
    y_dim    = ( ds.isel( T=slice(0) ) ).sizes['Y']
    
    no_tr_samples = int(tr_end/subsample_rate)+1
    no_val_samples = int((val_end - tr_end)/subsample_rate)+1
    ds.close()

    logging.info('no_training_samples ; '+str(no_tr_samples)+'\n')
    logging.info('no_validation_samples ; '+str(no_val_samples)+'\n')
    logging.info('Model ; '+model_name+'\n')
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info('Using device: '+device+'\n')
    
    logging.info('Batch Size: '+str(hyper_params['batch_size'])+'\n')
    
    TimeCheck(tic, 'setting variables')
    logging.info( GPUtil.showUtilization() )
    
    #-----------------------------------
    # Calulate, or read in mean and std
    #-----------------------------------
    if args.calculatemeanstd:
       no_in_channels, no_out_channels = CalcMeanStd(MeanStd_prefix, MITGCM_filename, train_end_ratio, subsample_rate, dataset_end_index, hyper_params["batch_size"], args.land, args.dim, tic)
    
    data_mean, data_std, data_range, no_in_channels, no_out_channels = ReadMeanStd(MeanStd_prefix)
    
    TimeCheck(tic, 'getting mean & std')
    logging.info( GPUtil.showUtilization() )
    
    #-------------------------------------------------------------------------------------
    # Create training and valdiation Datsets and dataloaders with normalisation
    #-------------------------------------------------------------------------------------
    # Note normalisation is carries out channel by channel, over the inputs and targets, using mean and std from training data
    if args.dim == '2d':
       Train_Dataset = rr.MITGCM_Dataset_2d( MITGCM_filename, 0.0, train_end_ratio, subsample_rate, dataset_end_index, args.land, tic, 
                                             transform = transforms.Compose( [ rr.RF_Normalise(data_mean, data_std, data_range)] ) )
       Val_Dataset   = rr.MITGCM_Dataset_2d( MITGCM_filename, train_end_ratio, val_end_ratio, subsample_rate, dataset_end_index, args.land, tic, 
                                             transform = transforms.Compose( [ rr.RF_Normalise(data_mean, data_std, data_range)] ) )
    elif args.dim == '3d':
       Train_Dataset = rr.MITGCM_Dataset_3d( MITGCM_filename, 0.0, train_end_ratio, subsample_rate, dataset_end_index, args.land, tic, 
                                                  transform = transforms.Compose( [ rr.RF_Normalise(data_mean, data_std, data_range)] ) )
       Val_Dataset   = rr.MITGCM_Dataset_3d( MITGCM_filename, train_end_ratio, val_end_ratio, subsample_rate, dataset_end_index, args.land, tic, 
                                                  transform = transforms.Compose( [ rr.RF_Normalise(data_mean, data_std, data_range)] ) )
    else:
       raise RuntimeError("ERROR!!! what's happening with dimensions?!")
    
    train_loader = torch.utils.data.DataLoader(Train_Dataset, batch_size=hyper_params["batch_size"], shuffle=True, num_workers=args.numworkers )
    val_loader   = torch.utils.data.DataLoader(Val_Dataset,  batch_size=hyper_params["batch_size"], shuffle=True, num_workers=args.numworkers )
    
    TimeCheck(tic, 'reading training & validation data')
    logging.info( GPUtil.showUtilization() )
    
    #--------------
    # Set up model
    #--------------
    h, optimizer = CreateModel(args.modelstyle, no_in_channels, no_out_channels, hyper_params["learning_rate"], args.seed, args.padding)
    
    TimeCheck(tic, 'setting up model')
    logging.info( GPUtil.showUtilization() )
    
    #------------------
    # Train or load the model:
    #------------------
    # define dictionary to store losses at each epoch
    losses = {'train'     : [],
              'train_Temp': [],
              'train_U'   : [],
              'train_V'   : [],
              'train_Eta' : [],
              'val'       : [] }
    
    if args.loadmodel:
       if args.trainmodel:
          LoadModel(model_name, h, optimizer, args.savedepochs, 'tr', losses)
          losses, plotting_data, hist_data = TrainModel(model_name, args.dim, tic, args.test, no_tr_samples, no_val_samples, 
                                                        plot_freq, save_freq, train_loader, val_loader, h, optimizer, hyper_params,
                                                        args.seed, losses, start_epoch=start_epoch)
          plot_training_output(model_name, start_epoch, total_epochs, plot_freq, losses, plotting_data, hist_data)
       else:
          LoadModel(model_name, h, optimizer, args.savedepochs, 'inf', losses)
    elif args.trainmodel:  # Training mode BUT NOT loading model!
       losses, plotting_data, hist_data = TrainModel(model_name, args.dim, tic, args.test, no_tr_samples, no_val_samples, plot_freq,
                                                     save_freq, train_loader, val_loader, h, optimizer, hyper_params, args.seed,
                                                     losses)
       plot_training_output(model_name, start_epoch, total_epochs, plot_freq, losses, plotting_data, hist_data)
    
    TimeCheck(tic, 'loading/training model')
    logging.info( GPUtil.showUtilization() )
    
    #------------------
    # Assess the model 
    #------------------
    if args.assess:
    
       OutputStats(model_name, MeanStd_prefix, MITGCM_filename, train_loader, h, total_epochs, y_dim_used, args.dim)
    
    TimeCheck(tic, 'assessing model')
    logging.info( GPUtil.showUtilization() )
    
    #---------------------
    # Iteratively predict 
    #---------------------
    if args.dim == '2d':
       Iterate_Dataset = rr.MITGCM_Dataset_2d( MITGCM_filename, 0.0, train_end_ratio, 1, dataset_end_index, args.land, tic, 
                                                  transform = transforms.Compose( [ rr.RF_Normalise(data_mean, data_std, data_range)] ) )
    elif args.dim == '3d':
       Iterate_Dataset = rr.MITGCM_Dataset_3d( MITGCM_filename, 0.0, train_end_ratio, 1, dataset_end_index, args.land, tic, 
                                                  transform = transforms.Compose( [ rr.RF_Normalise(data_mean, data_std, data_range)] ) )
    
    if args.iterate:
       IterativelyPredict(model_name, MeanStd_prefix, MITGCM_filename, Iterate_Dataset, h, start, for_len, total_epochs, y_dim_used, args.land, args.dim) 
    
    TimeCheck(tic, 'iteratively forecasting')
    logging.info( GPUtil.showUtilization() )
    
    TimeCheck(tic, 'script')
    logging.info( GPUtil.showUtilization() )

