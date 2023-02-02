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

import wandb

def parse_args():
    a = argparse.ArgumentParser()

    a.add_argument("-na", "--name", default="", action='store')
    a.add_argument("-te", "--test", default=False, type=bool, action='store')
    a.add_argument("-cd", "--createdataset", default=False, type=bool, action='store')
    a.add_argument("-ms", "--modelstyle", default='UNet2dtransp', type=str, action='store')
    a.add_argument("-di", "--dim", default='2d', type=str, action='store')
    a.add_argument("-la", "--land", default='Spits', type=str, action='store')
    a.add_argument("-pj", "--predictionjump", default='12hrly', type=str, action='store')
    a.add_argument("-nm", "--normmethod", default='range', type=str, action='store')
    a.add_argument("-hl", "--histlen", default=1, type=int, action='store')
    a.add_argument("-pl", "--predlen", default=1, type=int, action='store')
    a.add_argument("-pa", "--padding", default='None', type=str, action='store')
    a.add_argument("-ks", "--kernsize", default=3, type=int, action='store')
    a.add_argument("-bs", "--batchsize", default=16, type=int, action='store')
    a.add_argument("-bw", "--bdyweight", default=1., type=float, action='store')
    a.add_argument("-lr", "--learningrate", default=0.000003, type=float, action='store')
    a.add_argument("-wd", "--weightdecay", default=0., type=float, action='store')
    a.add_argument("-nw", "--numworkers", default=8, type=int, action='store')
    a.add_argument("-sd", "--seed", default=30475, type=int, action='store')
    a.add_argument("-lv", "--landvalue", default=0., type=float, action='store')
    a.add_argument("-lo", "--loadmodel", default=False, type=bool, action='store')
    a.add_argument("-se", "--savedepochs", default=0, type=int, action='store')
    a.add_argument("-be", "--best", default=False, type=bool, action='store')
    a.add_argument("-tr", "--trainmodel", default=False, type=bool, action='store')
    a.add_argument("-ep", "--epochs", default=200, type=int, action='store')
    a.add_argument("-ps", "--plotscatter", default=False, type=bool, action='store')
    a.add_argument("-as", "--assess", default=False, type=bool, action='store')
    a.add_argument("-it", "--iterate", default=False, type=bool, action='store')
    a.add_argument("-im", "--iteratemethod", default='simple', type=str, action='store')
    a.add_argument("-is", "--iteratesmooth", default=0, type=int, action='store')
    a.add_argument("-pe", "--plotevolution", default=False, type=bool, action='store')
    a.add_argument("-ee", "--evolutionepochs", default='10,50,100,150,200', type=str, action='store')

    return a.parse_args()

if __name__ == "__main__":
    
    mp.set_start_method('spawn')

    args = parse_args()
  
    ev_epoch_ls = [int(item) for item in args.evolutionepochs.split(',')]

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
    
    if args.predictionjump == '12hrly':
       for_len = 120    # How long to iteratively predict for
       for_subsample = 1
    elif args.predictionjump == 'hrly':
       for_len = 1440   # How long to iteratively predict for
       for_subsample = 12
    elif args.predictionjump == '10min':
       for_len = 8640   # How long to iteratively predict for
       for_subsample = 72
    start = 0        # Start from zero to fit with perturbed runs
    
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
   
    model_name = args.name+args.land+args.predictionjump+'_'+args.modelstyle+'_histlen'+str(args.histlen)+'_predlen'+str(args.predlen)+'_seed'+str(args.seed)
    # Amend some variables if testing code
    if args.test: 
       model_name = model_name+'_TEST'
       for_len = min(for_len, 50)
       args.epochs = 5
       args.evolutionepochs = '5'
       ev_epoch_ls = [int(item) for item in args.evolutionepochs.split(',')]
    
    model_dir = '../../../Channel_nn_Outputs/'+model_name
    if not os.path.isdir(model_dir):
       os.system("mkdir %s" % (model_dir))
       os.system("mkdir %s" % (model_dir+'/MODELS'))
       os.system("mkdir %s" % (model_dir+'/PLOTS'))
       os.system("mkdir %s" % (model_dir+'/STATS'))
       os.system("mkdir %s" % (model_dir+'/ITERATED_FORECAST'))
       os.system("mkdir %s" % (model_dir+'/ITERATED_FORECAST/PLOTS'))
       os.system("mkdir %s" % (model_dir+'/TRAIN_EVOLUTION'))
    
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

    if args.modelstyle == 'ConvLSTM' or args.modelstyle == 'UNetConvLSTM':
       channel_dim = 2
    elif args.dim == '2d':
       channel_dim = 1
    elif args.dim == '3d':
       channel_dim = 1
    
    if args.land == 'Spits':
       grid_filename = '/data/hpcdata/users/racfur/MITgcm/verification/MundayChannelConfig10km_LandSpits/runs/50yr_Cntrl/grid.nc'
       if args.predictionjump == '12hrly':
          if args.test: 
             #MITgcm_filename = '/data/hpcflash/users/racfur/50yr_Cntrl/50yr_Cntrl/12hrly_small_set.nc'
             MITgcm_filename = '/data/hpcdata/users/racfur/MITgcm/verification/MundayChannelConfig10km_LandSpits/runs/50yr_Cntrl/12hrly_small_set.nc'
          else:
             MITgcm_filename = '/data/hpcdata/users/racfur/MITgcm/verification/MundayChannelConfig10km_LandSpits/runs/50yr_Cntrl/12hrly_data.nc'
             #MITgcm_filename = '/data/hpcflash/users/racfur/50yr_Cntrl/12hrly_data.nc'
             #MITgcm_filename = '/local/extra/racfur/MundayChannelConfig10km_LandSpits/runs/12hrly_data.nc'
       elif args.predictionjump == 'hrly':
          if args.test: 
             MITgcm_filename = '/data/hpcdata/users/racfur/MITgcm/verification/MundayChannelConfig10km_LandSpits/runs/hrly_output/hrly_small_set.nc'
          else:
             MITgcm_filename = '/local/extra/racfur/MundayChannelConfig10km_LandSpits/runs/hrly_data.nc'
       elif args.predictionjump == '10min':
          if args.test: 
             MITgcm_filename = '/data/hpcdata/users/racfur/MITgcm/verification/MundayChannelConfig10km_LandSpits/runs/10min_output/10min_small_set.nc'
          else:
             MITgcm_filename = '/local/extra/racfur/MundayChannelConfig10km_LandSpits/runs/10min_data.nc' 
          subsample_rate = 3*subsample_rate # give larger subsample for such small timestepping, MITgcm dataset is longer so same amount of training/val samples
    else:
       grid_filename = '/data/hpcdata/users/racfur/MITgcm/verification/MundayChannelConfig10km_nodiff/runs/50yr_Cntrl/grid.nc'
       if args.test: 
          MITgcm_filename = '/data/hpcdata/users/racfur/MITgcm/verification/MundayChannelConfig10km_nodiff/runs/50yr_Cntrl/12hrly_small_set.nc'
       else:
          DIR =  '/local/extra/racfur/MundayChannelConfig10km_nodiff/runs/50yr_Cntrl/'
          MITgcm_filename = DIR+'12hrly_data.nc'
    
    print(MITgcm_filename)
    ds = xr.open_dataset(MITgcm_filename)
    
    ds.close()

    wandb.init(project="ChannelConfig", entity="rf-phd", name=model_name)
    logging.info('Model ; '+model_name+'\n')
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info('Using device: '+device+'\n')
    
    wandb.config = {
       "name": model_name,
       "norm_method": args.normmethod,
       "learning_rate": args.learningrate,
       "epochs": args.epochs,
       "batch_size": args.batchsize
    }
 
    #TimeCheck(tic, 'setting variables')
   
    #----------------------------------------
    # Read in mean and std, and set channels
    #----------------------------------------
    z_dim = ( ds.isel( T=slice(0) ) ).sizes['Zmd000038'] 

    inputs_mean, inputs_std, inputs_range, targets_mean, targets_std, targets_range = ReadMeanStd(args.dim, z_dim, args.predictionjump)

    if args.modelstyle == 'ConvLSTM' or args.modelstyle == 'UNetConvLSTM':
       no_phys_channels = 3*z_dim + 1                              # Eta, plus Temp, U, V through depth
       no_in_channels = no_phys_channels + z_dim                   # Eta, plus Temp, U, V through depth, plus masks
       no_out_channels = no_phys_channels                          # Eta, plus Temp, U, V through depth
    elif args.dim == '2d':
       no_phys_channels = 3*z_dim + 1                              # Eta, plus Temp, U, V through depth
       no_in_channels = args.histlen * no_phys_channels + z_dim    # Eta, plus Temp, U, V through depth, for each past time, plus masks
       no_out_channels = no_phys_channels                          # Eta, plus Temp, U, V through depth (predict 1 step ahead, even if LF calculated over mutliple steps
    elif args.dim == '3d':
       no_phys_channels = 4                                        # Eta, Temp, U, V
       no_in_channels = args.histlen * no_phys_channels + 1        # Eta Temp, U, V , for each past time, plus masks
       no_out_channels = no_phys_channels                          # Eta, Temp, U, V just once
   
    logging.info('no_phys_channels ;'+str(no_phys_channels)+'\n')
    logging.info('no_in_channels ;'+str(no_in_channels)+'\n')
    logging.info('no_out_channels ;'+str(no_out_channels)+'\n')

    #---------------------------------------------------------------------------------------------------------------------------------------
    # Create Dataset and save to disk - only needs doing once, ideally on CPU, then training, validation, test data read in from saved file
    #---------------------------------------------------------------------------------------------------------------------------------------
    if args.landvalue == -999:
       landvalues = inputs_mean
    else:
       landvalues = np.ones(inputs_mean.shape)
       landvalues[:] = args.landvalue
       
    if args.createdataset:

       rr.CreateDataset(MITgcm_filename, subsample_rate, args.histlen, args.predlen, args.land, tic, args.bdyweight, landvalues,
                        grid_filename, args.dim, args.modelstyle, args.predictionjump, z_dim, no_phys_channels, no_in_channels, no_out_channels,
                        args.normmethod, model_name)

    #-------------------------------------------------------------------------------------
    # Create training and validation Datsets and dataloaders with normalisation
    #-------------------------------------------------------------------------------------
    # Note normalisation is carries out channel by channel, over the inputs and targets, using mean and std from training data
    #Train_Dataset = rr.MITgcm_Dataset( MITgcm_filename, 0.0, train_end_ratio, subsample_rate,
    #                                   args.histlen, args.predlen, args.land, tic, args.bdyweight, landvalues, grid_filename, args.dim, args.modelstyle,
    #                                   transform = transforms.Compose( [ rr.RF_Normalise_sample(inputs_mean, inputs_std, inputs_range,
    #                                                                     targets_mean, targets_std, targets_range, args.histlen, args.predlen,
    #                                                                     no_phys_channels, args.dim, args.normmethod, args.modelstyle)] ) )
    #Val_Dataset   = rr.MITgcm_Dataset( MITgcm_filename, train_end_ratio, val_end_ratio, subsample_rate,
    #                                   args.histlen, args.predlen, args.land, tic, args.bdyweight, landvalues, grid_filename, args.dim, args.modelstyle,
    #                                   transform = transforms.Compose( [ rr.RF_Normalise_sample(inputs_mean, inputs_std, inputs_range,
    #                                                                     targets_mean, targets_std, targets_range, args.histlen, args.predlen,
    #                                                                     no_phys_channels, args.dim, args.normmethod, args.modelstyle)] ) )
    #Test_Dataset  = rr.MITgcm_Dataset( MITgcm_filename, val_end_ratio, 1.0, subsample_rate,
    #                                   args.histlen, args.predlen, args.land, tic, args.bdyweight, landvalues, grid_filename, args.dim, args.modelstyle,
    #                                   transform = transforms.Compose( [ rr.RF_Normalise_sample(inputs_mean, inputs_std, inputs_range,
    #                                                                     targets_mean, targets_std, targets_range, args.histlen, args.predlen,
    #                                                                     no_phys_channels, args.dim, args.normmethod, args.modelstyle)] ) )
    Train_Dataset = rr.ProcData_Dataset(model_name, 0., train_end_ratio)
    Val_Dataset   = rr.ProcData_Dataset(model_name, train_end_ratio, val_end_ratio)
    Test_Dataset  = rr.ProcData_Dataset(model_name, val_end_ratio, 1.)

    no_tr_samples = len(Train_Dataset)
    no_val_samples = len(Val_Dataset)
    no_test_samples = len(Test_Dataset)
    logging.info('no_training_samples ; '+str(no_tr_samples)+'\n')
    logging.info('no_validation_samples ; '+str(no_val_samples)+'\n')
    logging.info('no_test_samples ; '+str(no_test_samples)+'\n')

    train_loader = torch.utils.data.DataLoader(Train_Dataset, batch_size=args.batchsize, shuffle=True,
                                               num_workers=args.numworkers, pin_memory=True )
    val_loader   = torch.utils.data.DataLoader(Val_Dataset,  batch_size=args.batchsize, shuffle=True, 
                                               num_workers=args.numworkers, pin_memory=True )
    test_loader  = torch.utils.data.DataLoader(Test_Dataset,  batch_size=args.batchsize, shuffle=True, 
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
          losses = TrainModel(model_name, args.modelstyle, args.dim, args.histlen, args.predlen, tic, args.test, no_tr_samples, no_val_samples, 
                              save_freq, train_loader, val_loader, h, optimizer, args.epochs, args.seed, losses, 
                              channel_dim, no_phys_channels, start_epoch=start_epoch, current_best_loss=current_best_loss)
          plot_training_output(model_name, start_epoch, total_epochs, plot_freq, losses )
       else:
          LoadModel(model_name, h, optimizer, args.savedepochs, 'inf', losses, args.best)
    elif args.trainmodel:  # Training mode BUT NOT loading model
       losses = TrainModel(model_name, args.modelstyle, args.dim, args.histlen, args.predlen, tic, args.test, no_tr_samples, no_val_samples,
                           save_freq, train_loader, val_loader, h, optimizer, args.epochs, args.seed,
                           losses, channel_dim, no_phys_channels)
       plot_training_output(model_name, start_epoch, total_epochs, plot_freq, losses)
   
    #--------------------
    # Plot scatter plots
    #--------------------
    if args.plotscatter:
    
       PlotScatter(model_name, args.dim, train_loader, h, total_epochs, 'training', args.land+'_'+args.dim,
                   args.normmethod, channel_dim, args.predictionjump, no_phys_channels)
       PlotScatter(model_name, args.dim, val_loader, h, total_epochs, 'validation', args.land+'_'+args.dim,
                   args.normmethod, channel_dim, args.predictionjump, no_phys_channels)
       PlotScatter(model_name, args.dim, test_loader, h, total_epochs, 'test', args.land+'_'+args.dim,
                   args.normmethod, channel_dim, args.predictionjump, no_phys_channels)
    
    #------------------
    # Assess the model 
    #------------------
    if args.assess:
    
       stats_train_loader = torch.utils.data.DataLoader(Train_Dataset, batch_size=args.batchsize, shuffle=True,
                                                  num_workers=args.numworkers, pin_memory=True )
       stats_val_loader   = torch.utils.data.DataLoader(Val_Dataset,  batch_size=args.batchsize, shuffle=True, 
                                               num_workers=args.numworkers, pin_memory=True )
       stats_test_loader  = torch.utils.data.DataLoader(Test_Dataset,  batch_size=args.batchsize, shuffle=True, 
                                               num_workers=args.numworkers, pin_memory=True )
    
       OutputStats(model_name, args.modelstyle, args.land+'_'+args.dim, MITgcm_filename, stats_train_loader, h, total_epochs, y_dim_used, args.dim, 
                   args.histlen, args.land, 'training', args.normmethod, channel_dim, args.predictionjump, no_phys_channels)
    
       OutputStats(model_name, args.modelstyle, args.land+'_'+args.dim, MITgcm_filename, stats_val_loader, h, total_epochs, y_dim_used, args.dim, 
                   args.histlen, args.land, 'validation', args.normmethod, channel_dim, args.predictionjump, no_phys_channels)
    
       OutputStats(model_name, args.modelstyle, args.land+'_'+args.dim, MITgcm_filename, stats_test_loader, h, total_epochs, y_dim_used, args.dim, 
                   args.histlen, args.land, 'test', args.normmethod, channel_dim, args.predictionjump, no_phys_channels)
    
    #---------------------
    # Iteratively predict 
    #---------------------
    if args.iterate:

       Iterate_Dataset = rr.MITgcm_Dataset( MITgcm_filename, 0., 1., 1, args.histlen, args.predlen, args.land, tic, args.bdyweight, landvalues,
                                            grid_filename, args.dim,  args.modelstyle,
                                            transform = transforms.Compose( [ rr.RF_Normalise_sample(inputs_mean, inputs_std, inputs_range,
                                                                              targets_mean, targets_std, targets_range,
                                                                              args.histlen, args.predlen, no_phys_channels, args.dim, 
                                                                              args.normmethod, args.modelstyle)] ) )
    
       IterativelyPredict(model_name, args.modelstyle, args.land+'_'+args.dim, MITgcm_filename, Iterate_Dataset, h, start, for_len, total_epochs,
                          y_dim_used, args.land, args.dim, args.histlen, landvalues,
                          args.iteratemethod, args.iteratesmooth, args.normmethod, channel_dim, args.predictionjump, for_subsample) 
    
    #------------------------------------------------------
    # Plot fields from various training steps of the model
    #------------------------------------------------------
    if args.plotevolution:

       #Evolve_Dataset = rr.MITgcm_Dataset( MITgcm_filename, 0.0, train_end_ratio, subsample_rate, args.histlen, args.predlen, 
       #                                    args.land, tic, args.bdyweight, landvalues,
       #                                    grid_filename, args.dim, args.modelstyle,
       #                                    transform = transforms.Compose( [ rr.RF_Normalise_sample(inputs_mean, inputs_std, inputs_range,
       #                                                                      targets_mean, targets_std, targets_range,
       #                                                                      args.histlen, args.predlen, no_phys_channels, args.dim,
       #                                                                      args.normmethod, args.modelstyle)] ) )

       PlotTrainingEvolution(model_name, args.modelstyle, args.land+'_'+args.dim, MITgcm_filename, Train_Dataset, h, optimizer, ev_epoch_ls,
                             args.dim, args.histlen, landvalues, args.normmethod, channel_dim, args.predictionjump, no_phys_channels)
