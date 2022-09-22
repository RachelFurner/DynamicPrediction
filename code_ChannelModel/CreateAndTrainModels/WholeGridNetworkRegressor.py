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

import netCDF4 as nc4

import time as time
import gc

import argparse
import logging

import multiprocessing as mp
#import torch.multiprocessing
#torch.multiprocessing.set_sharing_strategy('file_system')

def parse_args():
    a = argparse.ArgumentParser()

    a.add_argument("-n", "--name", default="", action='store')
    a.add_argument("-te", "--test", default=False, type=bool, action='store')
    a.add_argument("-cm", "--calculatemeanstd", default=False, type=bool, action='store')
    a.add_argument("-m", "--modelstyle", default='UNet2dtransp', type=str, action='store')
    a.add_argument("-d", "--dim", default='2d', type=str, action='store')
    a.add_argument("-la", "--land", default='Spits', type=str, action='store')
    a.add_argument("-hl", "--histlen", default=1, type=int, action='store')
    a.add_argument("-p", "--padding", default='None', type=str, action='store')
    a.add_argument("-k", "--kernsize", default=3, type=int, action='store')
    a.add_argument("-b", "--batchsize", default=16, type=int, action='store')
    a.add_argument("-bw", "--bdyweight", default=1., type=float, action='store')
    a.add_argument("-lr", "--learningrate", default=0.000003, type=float, action='store')
    a.add_argument("-wd", "--weightdecay", default=0., type=float, action='store')
    a.add_argument("-w", "--numworkers", default=8, type=int, action='store')
    a.add_argument("-s", "--seed", default=30475, type=int, action='store')
    a.add_argument("-lo", "--loadmodel", default=False, type=bool, action='store')
    a.add_argument("-se", "--savedepochs", default=0, type=int, action='store')
    a.add_argument("-be", "--best", default=False, type=bool, action='store')
    a.add_argument("-tr", "--trainmodel", default=False, type=bool, action='store')
    a.add_argument("-e", "--epochs", default=200, type=int, action='store')
    a.add_argument("-ps", "--plotscatter", default=False, type=bool, action='store')
    a.add_argument("-a", "--assess", default=False, type=bool, action='store')
    a.add_argument("-i", "--iterate", default=False, type=bool, action='store')
    a.add_argument("-im", "--iteratemethod", default='simple', type=str, action='store')
    a.add_argument("-lv", "--landvalue", default=0., type=float, action='store')

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

    logging.info('args '+str(args))
    
    subsample_rate = 5      # number of time steps to skip over when creating training and test data
    train_end_ratio = 0.75  # Take training samples from 0 to this far through the dataset
    val_end_ratio = 0.9     # Take validation samples from train_end_ratio to this far through the dataset
    
    plot_freq = 10     # Plot scatter plot, and save the model every n epochs (save in case of a crash etc)
    save_freq = 10      # Plot scatter plot, and save the model every n epochs (save in case of a crash etc)
    
    #for_len = 30*6   # How long to iteratively predict for
    for_len = 5      # How long to iteratively predict for
    start = 5        #
    
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
   
    # Amend some variables if testing code
    if args.test: 
       model_name = args.name+args.land+'_'+args.modelstyle+'_histlen'+str(args.histlen)+'_seed'+str(args.seed)+'_TEST'
       for_len = min(for_len, 50)
       args.epochs = 5
    else:
       model_name = args.name+args.land+'_'+args.modelstyle+'_histlen'+str(args.histlen)+'_seed'+str(args.seed)
    
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
          total_epochs = args.savedepochs+args.epochs
       else:
          start_epoch = 1
          total_epochs = args.epochs
    else:
       start_epoch = args.savedepochs
       total_epochs = args.savedepochs
   
    if args.land == 'IncLand' or args.land == 'Spits':
       y_dim_used = 104
    elif args.land == 'ExcLand':
       y_dim_used = 97
    else:
       raise RuntimeError('ERROR, Whats going on with args.land?!')
    
    if args.land == 'Spits':
       DIR =  '/local/extra/racfur/MundayChannelConfig10km_LandSpits/runs/50yr_Cntrl/'
       if args.test: 
          MITGCM_filename = '/data/hpcdata/users/racfur/MITgcm/verification/MundayChannelConfig10km_LandSpits/runs/50yr_Cntrl_orig/12hrly_small_set.nc'
       else:
          MITGCM_filename = DIR +'12hrly_data.nc'
       grid_filename = '/data/hpcdata/users/racfur/MITgcm/verification/MundayChannelConfig10km_LandSpits/runs/50yr_Cntrl_orig/grid.nc'
    else:
       DIR =  '/local/extra/racfur/MundayChannelConfig10km_nodiff/runs/50yr_Cntrl/'
       if args.test: 
          MITGCM_filename = '/data/hpcdata/users/racfur/MITgcm/verification/MundayChannelConfig10km_nodiff/runs/50yr_Cntrl/12hrly_small_set.nc'
       else:
          MITGCM_filename = DIR+'12hrly_data.nc'
       grid_filename = '/data/hpcdata/users/racfur/MITgcm/verification/MundayChannelConfig10km_nodiff/runs/50yr_Cntrl/grid.nc'
    
    ds = xr.open_dataset(MITGCM_filename)
    
    ds.close()

    logging.info('Model ; '+model_name+'\n')
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info('Using device: '+device+'\n')
    
    #TimeCheck(tic, 'setting variables')
    
    #-----------------------------------
    # Calulate, or read in mean and std
    #-----------------------------------
    if args.calculatemeanstd:
       CalcMeanStd(args.land+'_'+args.dim, MITGCM_filename, train_end_ratio, subsample_rate,
                   args.batchsize, args.land, args.dim, args.bdyweight, args.histlen, grid_filename, tic)

    inputs_mean, inputs_std, inputs_range, targets_mean, targets_std, targets_range = ReadMeanStd(args.land+'_'+args.dim)

    z_dim = ( ds.isel( T=slice(0) ) ).sizes['Zmd000038'] 
    if args.dim == '2d':
       no_in_channels = args.histlen * ( 3*z_dim + 1) + (3*z_dim + 1)  # Eta, plus Temp, U, V through depth, for each past time, plus masks
       no_out_channels = 3*z_dim + 1                    # Eta field, plus Temp, U, V through depth, just once
    elif args.dim == '3d':
       no_in_channels = args.histlen * 4 + 4  # Eta Temp, U, V , for each past time, plus masks
       no_out_channels = 4                # Eta, Temp, U, V just once
   
    logging.debug('no_in_channels ;'+str(no_in_channels)+'\n')
    logging.debug('no_out_channels ;'+str(no_out_channels)+'\n')

    #-------------------------------------------------------------------------------------
    # Create training and valdiation Datsets and dataloaders with normalisation
    #-------------------------------------------------------------------------------------
    if args.landvalue == -999:
       landvalues = inputs_mean
    else:
       landvalues = np.ones(inputs_mean.shape)
       landvalues[:] = args.landvalue
    # Note normalisation is carries out channel by channel, over the inputs and targets, using mean and std from training data
    Train_Dataset = rr.MITGCM_Dataset( MITGCM_filename, 0.0, train_end_ratio, subsample_rate,
                                             args.histlen, args.land, tic, args.bdyweight, landvalues, grid_filename, args.dim,
                                             transform = transforms.Compose( [ rr.RF_Normalise_sample(inputs_mean, inputs_std, inputs_range,
                                                                               targets_mean, targets_std, targets_range,
                                                                               args.histlen, no_out_channels, args.dim)] ) )
    Val_Dataset   = rr.MITGCM_Dataset( MITGCM_filename, train_end_ratio, val_end_ratio, subsample_rate,
                                             args.histlen, args.land, tic, args.bdyweight, landvalues, grid_filename, args.dim,
                                             transform = transforms.Compose( [ rr.RF_Normalise_sample(inputs_mean, inputs_std, inputs_range,
                                                                               targets_mean, targets_std, targets_range,
                                                                               args.histlen, no_out_channels, args.dim)] ) )

    no_tr_samples = len(Train_Dataset)
    no_val_samples = len(Val_Dataset)
    logging.debug('no_training_samples ; '+str(no_tr_samples)+'\n')
    logging.debug('no_validation_samples ; '+str(no_val_samples)+'\n')

    train_loader = torch.utils.data.DataLoader(Train_Dataset, batch_size=args.batchsize, shuffle=True,
                                               num_workers=args.numworkers, pin_memory=True )
    val_loader   = torch.utils.data.DataLoader(Val_Dataset,  batch_size=args.batchsize, shuffle=True, 
                                               num_workers=args.numworkers, pin_memory=True )

    #--------------
    # Set up model
    #--------------
    h, optimizer = CreateModel( args.modelstyle, no_in_channels, no_out_channels, args.learningrate, args.seed, args.padding,
                                ds.isel(T=slice(0)).sizes['X'], y_dim_used, z_dim, args.kernsize, args.weightdecay )
   
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
          losses, h, optimizer, current_best_loss = LoadModel(model_name, h, optimizer, args.savedepochs, 'tr', losses, args.best)
          losses, histogram_inputs, histogram_targets = TrainModel(model_name, args.dim, args.histlen, tic, args.test, no_tr_samples, no_val_samples, 
                                                             save_freq, train_loader, val_loader, h, optimizer,
                                                             args.epochs, args.seed, losses, 
                                                             no_in_channels, no_out_channels, start_epoch=start_epoch, current_best_loss=current_best_loss)
          plot_training_output(model_name, start_epoch, total_epochs, plot_freq, losses )
          #  Save histogram data to file and plot it
          histogram_file = '../../../Channel_nn_Outputs/'+args.land+'_'+args.dim+'_norm_histogram.npz'
          np.savez( histogram_file, histogram_inputs, histogram_targets )
          plot_histograms(args.land+'_'+args.dim, histogram_inputs, histogram_targets, norm='norm')
       else:
          LoadModel(model_name, h, optimizer, args.savedepochs, 'inf', losses, args.best)
    elif args.trainmodel:  # Training mode BUT NOT loading model
       losses, histogram_inputs, histogram_targets = TrainModel(model_name, args.dim, args.histlen, tic, args.test, no_tr_samples, no_val_samples,
                                                     save_freq, train_loader, val_loader, h, optimizer,
                                                     args.epochs, args.seed,
                                                     losses, no_in_channels, no_out_channels)
       plot_training_output(model_name, start_epoch, total_epochs, plot_freq, losses)
       #  Save histogram data to file and plot it
       histogram_file = '../../../Channel_nn_Outputs/'+args.land+'_'+args.dim+'_norm_histogram.npz'
       np.savez( histogram_file, histogram_inputs, histogram_targets )
       plot_histograms(args.land+'_'+args.dim, histogram_inputs, histogram_targets, norm='norm')
   
    #--------------------
    # Plot scatter plots
    #--------------------
    if args.plotscatter:
    
       PlotScatter(model_name, args.dim, train_loader, h, total_epochs, 'training', no_out_channels, args.land+'_'+args.dim)
       PlotScatter(model_name, args.dim, val_loader, h, total_epochs, 'validation', no_out_channels, args.land+'_'+args.dim)
    
    #------------------
    # Assess the model 
    #------------------
    if args.assess:
    
       OutputStats(model_name, args.land+'_'+args.dim, MITGCM_filename, train_loader, h, total_epochs, y_dim_used, args.dim, 
                   args.histlen, no_in_channels, no_out_channels, args.land, 'training')
    
       OutputStats(model_name, args.land+'_'+args.dim, MITGCM_filename, val_loader, h, total_epochs, y_dim_used, args.dim, 
                   args.histlen, no_in_channels, no_out_channels, args.land, 'validation')
    
    #---------------------
    # Iteratively predict 
    #---------------------
    Iterate_Dataset = rr.MITGCM_Dataset( MITGCM_filename, 0., 1., 1, args.histlen, args.land, tic, args.bdyweight, landvalues, grid_filename, args.dim,
                                                  transform = transforms.Compose( [ rr.RF_Normalise_sample(inputs_mean, inputs_std, inputs_range,
                                                                                    targets_mean, targets_std, targets_range,
                                                                                    args.histlen, no_out_channels, args.dim)] ) )
    
    if args.iterate:
       IterativelyPredict(model_name, args.land+'_'+args.dim, MITGCM_filename, Iterate_Dataset, h, start, for_len, total_epochs,
                          y_dim_used, args.land, args.dim, args.histlen, no_in_channels, no_out_channels, landvalues, args.iteratemethod) 
    
