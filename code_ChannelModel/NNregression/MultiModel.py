#!/usr/bin/env pyton
# coding: utf-8

# Code to create iterative prediction by combining predictions from mutliple models

#from comet_ml import Experiment
import sys
sys.path.append('../Tools')
import ReadRoutines as rr

sys.path.append('.')
from WholeGridNetworkRegressorModulesNew import *
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

import GPUtil

def parse_args():
    a = argparse.ArgumentParser()

    a.add_argument('modelseeds', metavar='N', type=int, nargs='+',
                   help='list of random seeds from the models')
    a.add_argument("-n", "--name", default="", action='store')
    a.add_argument("-d", "--dim", default='2d', type=str, action='store')
    a.add_argument("-la", "--land", default='ExcLand', type=str, action='store')
    a.add_argument("-k", "--kernsize", default=3, type=int, action='store')
    a.add_argument("-p", "--padding", default='None', type=str, action='store')
    a.add_argument("-m", "--modelstyle", default='UNet2dtransp', type=str, action='store')
    a.add_argument("-se", "--savedepochs", default=0, type=int, action='store')
    a.add_argument("-w", "--numworkers", default=8, type=int, action='store')
    a.add_argument("-s", "--seed", default=30475, type=int, action='store') 
    a.add_argument("-b", "--batchsize", default=8, type=int, action='store')

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
    # Set variables for this run
    #--------------------------------------

    logging.info('args')
    logging.info(args)
    
    for_len = 30*6   # How long to iteratively predict for
    start = 5        #    a.add_argument("-se", "--savedepochs", default=0, type=int, action='    a.add_argument("-se", "--savedepochs", default=0, type=int, action='store')store')
    
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
   
    MeanStd_prefix = args.land+'_'+args.dim

    model_dir = '../../../Channel_nn_Outputs/'+args.name
    if not os.path.isdir(model_dir):
       os.system("mkdir %s" % (model_dir))
       os.system("mkdir %s" % (model_dir+'/STATS'))
       os.system("mkdir %s" % (model_dir+'/ITERATED_FORECAST'))
       os.system("mkdir %s" % (model_dir+'/ITERATED_FORECAST/PLOTS'))
 
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
    subsample_rate = 5      # number of time steps to skip over when creating training and test data
    train_end_ratio = 0.75  # Take training samples from 0 to this far through the dataset
    val_end_ratio = 0.9     # Take validation samples from train_end_ratio to this far through the dataset
 
    ds       = xr.open_dataset(MITGCM_filename)
    tr_start = 0
    tr_end   = int(train_end_ratio * dataset_end_index)
    val_end  = int(val_end_ratio * dataset_end_index)
    x_dim    = ( ds.isel( T=slice(0) ) ).sizes['X']
    y_dim    = ( ds.isel( T=slice(0) ) ).sizes['Y']
    z_dim    = ( ds.isel( T=slice(0) ) ).sizes['Zld000038']
    ds.close()

    logging.info('Model ; '+args.name+'\n')
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info('Using device: '+device+'\n')
    
    TimeCheck(tic, 'setting variables')
    logging.info( GPUtil.showUtilization() )
    
    #-----------------------------------
    # Read in mean and std
    #-----------------------------------
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

    train_loader = torch.utils.data.DataLoader(Train_Dataset, batch_size=args.batchsize, shuffle=True,
                                               num_workers=args.numworkers, pin_memory=True )
    val_loader   = torch.utils.data.DataLoader(Val_Dataset,  batch_size=args.batchsize, shuffle=True,
                                               num_workers=args.numworkers, pin_memory=True )

    TimeCheck(tic, 'reading training & validation data')
    logging.info( GPUtil.showUtilization() )

    #---------------
    # Set up models
    #---------------
    h=[]
    optimizer=[]
    for i in range(len(args.modelseeds)):
       h_temp, optimizer_temp = CreateModel(args.modelstyle, no_in_channels, no_out_channels, 0.0001,
                                args.seed, args.padding, x_dim, y_dim_used, z_dim, args.kernsize)
       h.append(h_temp)
       optimizer.append(optimizer_temp)
   
    TimeCheck(tic, 'setting up model')
    logging.info( GPUtil.showUtilization() )
    
    #-----------------
    # Load the models
    #-----------------
    # define dictionary to store losses at each epoch
    losses=[]
    for i in range(len(args.modelseeds)):
       losses.append({'train'     : [],
                      'train_Temp': [],
                      'train_U'   : [],
                      'train_V'   : [],
                      'train_Eta' : [],
                      'val'       : [] })

       model_name = args.land+'_ksize'+str(args.kernsize)+'_'+args.modelstyle+'_lr0.0001_seed'+str(args.modelseeds[i]) 
 
       LoadModel(model_name, h[i], optimizer[i], args.savedepochs, 'inf', losses[i])
    
    TimeCheck(tic, 'loading/training model')
    logging.info( GPUtil.showUtilization() )
    
    #-------------------------------
    # Assess the multi-model set up 
    #-------------------------------

    OutputStats(args.name, MeanStd_prefix, MITGCM_filename, train_loader, h, args.savedepochs, y_dim_used, args.dim)

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
    
    IterativelyPredict(args.name, MeanStd_prefix, MITGCM_filename, Iterate_Dataset, h, start, for_len, args.savedepochs, y_dim_used, args.land, args.dim) 
    
    TimeCheck(tic, 'iteratively forecasting')
    logging.info( GPUtil.showUtilization() )
    
    TimeCheck(tic, 'script')
    logging.info( GPUtil.showUtilization() )

