#!/usr/bin/env pyton
# coding: utf-8

# Code to create iterative prediction by combining predictions from mutliple models

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

import GPUtil

def parse_args():
    a = argparse.ArgumentParser()

    a.add_argument('modelseeds', metavar='N', type=int, nargs='+',
                   help='list of random seeds from the models')
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
    # Set variables for this run
    #--------------------------------------

    logging.info('args')
    logging.info(args)
    
    for_len = 120    # How long to iteratively predict for
    start = 5        #    a.add_argument("-se", "--savedepochs", default=0, type=int, action='    a.add_argument("-se", "--savedepochs", default=0, type=int, action='store')store')
    
    rand_seed = 30475
    os.environ['PYTHONHASHSEED'] = str(rand_seed)
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)
    torch.cuda.manual_seed_all(rand_seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
   
    MeanStd_prefix = args.land+'_'+args.dim

    base_name = args.name+args.land+'_'+args.modelstyle+'_histlen'+str(args.histlen)
    model_name = 'MultiModel_'+base_name
    model_dir = '../../../Channel_nn_Outputs/'+model_name
    if not os.path.isdir(model_dir):
       os.system("mkdir %s" % (model_dir))
       os.system("mkdir %s" % (model_dir+'/STATS'))
       os.system("mkdir %s" % (model_dir+'/ITERATED_FORECAST'))
       os.system("mkdir %s" % (model_dir+'/ITERATED_FORECAST/PLOTS'))
 
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

    subsample_rate = 5      # number of time steps to skip over when creating training and test data
    train_end_ratio = 0.75  # Take training samples from 0 to this far through the dataset
    val_end_ratio = 0.9     # Take validation samples from train_end_ratio to this far through the dataset
 
    z_dim    = ( ds.isel( T=slice(0) ) ).sizes['Zmd000038']
    ds.close()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info('Using device: '+device+'\n')
    
    #-----------------------------------
    # Read in mean and std
    #-----------------------------------
    inputs_mean, inputs_std, inputs_range, targets_mean, targets_std, targets_range = ReadMeanStd(args.dim, z_dim)

    if args.dim == '2d':
       no_in_channels = args.histlen * ( 3*z_dim + 1) + (3*z_dim + 1)  # Eta, plus Temp, U, V through depth, for each past time, plus masks
       no_out_channels = 3*z_dim + 1                    # Eta field, plus Temp, U, V through depth, just once
    elif args.dim == '3d':
       no_in_channels = args.histlen * 4 + 4  # Eta Temp, U, V , for each past time, plus masks
       no_out_channels = 4                # Eta, Temp, U, V just once
 
    #--------------------------------------------------------
    # Create training and validation datsets and dataloaders
    #--------------------------------------------------------
    if args.landvalue == -999:
       landvalues = inputs_mean
    else:
       landvalues = np.ones(inputs_mean.shape)
       landvalues[:] = args.landvalue

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

    train_loader = torch.utils.data.DataLoader(Train_Dataset, batch_size=args.batchsize, shuffle=True,
                                               num_workers=args.numworkers, pin_memory=True )
    val_loader   = torch.utils.data.DataLoader(Val_Dataset,  batch_size=args.batchsize, shuffle=True,
                                               num_workers=args.numworkers, pin_memory=True )

    #---------------
    # Set up models
    #---------------
    h=[]
    optimizer=[]
    for i in range(len(args.modelseeds)):
       h_temp, optimizer_temp = CreateModel( args.modelstyle, no_in_channels, no_out_channels, args.learningrate, rand_seed, args.padding,
                                ds.isel(T=slice(0)).sizes['X'], y_dim_used, z_dim, args.kernsize, args.weightdecay )
       h.append(h_temp)
       optimizer.append(optimizer_temp)
    print(len(h))
   
    #-----------------
    # Load the models
    #-----------------
    for i in range(len(args.modelseeds)):
       losses = {'train'     : [],            
                  'train_Temp': [],            
                  'train_U'   : [],
                  'train_V'   : [],
                  'train_Eta' : [],
                  'val'       : [] } 

       model_i_name = base_name+'_seed'+str(args.modelseeds[i])
       print(model_i_name)
       LoadModel(model_i_name, h[i], optimizer[i], args.savedepochs, 'inf', losses, args.best) 
    
    #-------------------------------
    # Assess the multi-model set up 
    #-------------------------------

    if args.assess:

       print(len(h))
       OutputStats(model_name, args.land+'_'+args.dim, MITGCM_filename, train_loader, h, args.savedepochs, y_dim_used, args.dim,
                      args.histlen, no_in_channels, no_out_channels, args.land, 'training')
   
       OutputStats(model_name, args.land+'_'+args.dim, MITGCM_filename, val_loader, h, args.savedepochs, y_dim_used, args.dim,
                      args.histlen, no_in_channels, no_out_channels, args.land, 'validation')

    #---------------------
    # Iteratively predict 
    #---------------------
    if args.iterate:
       Iterate_Dataset = rr.MITGCM_Dataset( MITGCM_filename, 0., 1., 1, args.histlen, args.land, tic, args.bdyweight, landvalues, grid_filename, args.dim,
                                                     transform = transforms.Compose( [ rr.RF_Normalise_sample(inputs_mean, inputs_std, inputs_range,
                                                                                       targets_mean, targets_std, targets_range,
                                                                                       args.histlen, no_out_channels, args.dim)] ) )
   
       IterativelyPredict(model_name, args.land+'_'+args.dim, MITGCM_filename, Iterate_Dataset, h, start, for_len, args.savedepochs,
                          y_dim_used, args.land, args.dim, args.histlen, no_in_channels, no_out_channels, landvalues, args.iteratemethod)
       
   
