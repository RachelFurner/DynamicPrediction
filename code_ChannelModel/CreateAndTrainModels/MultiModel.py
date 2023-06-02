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

from torch.utils import data
#import torch.nn as nn
#from torch.autograd import Variable
#import torch
from torchvision import transforms#, utils
#import torch.autograd.profiler as profiler

#import netCDF4 as nc4

import argparse
import logging

import multiprocessing as mp

def parse_args():
    a = argparse.ArgumentParser()

    a.add_argument('modelseeds', metavar='N', type=int, nargs='+',
                   help='list of random seeds from the models')
    a.add_argument("-na", "--name", default="", action='store')
    a.add_argument("-te", "--test", default=False, type=bool, action='store')
    a.add_argument("-ms", "--modelstyle", default='UNet2dtransp', type=str, action='store')
    a.add_argument("-di", "--dim", default='2d', type=str, action='store')
    a.add_argument("-la", "--land", default='Spits', type=str, action='store')
    a.add_argument("-pj", "--predictionjump", default='12hrly', type=str, action='store')
    a.add_argument("-nm", "--normmethod", default='range', type=str, action='store')
    a.add_argument("-hl", "--histlen", default=1, type=int, action='store')
    a.add_argument("-pl", "--rolllen", default=1, type=int, action='store')
    a.add_argument("-pa", "--padding", default='None', type=str, action='store')
    a.add_argument("-ks", "--kernsize", default=3, type=int, action='store')
    a.add_argument("-bs", "--batchsize", default=36, type=int, action='store')
    a.add_argument("-bw", "--bdyweight", default=1., type=float, action='store')
    a.add_argument("-lr", "--learningrate", default=0.000003, type=float, action='store')
    a.add_argument("-wd", "--weightdecay", default=0., type=float, action='store')
    a.add_argument("-nw", "--numworkers", default=2, type=int, action='store')
    a.add_argument("-lv", "--landvalue", default=0., type=float, action='store')
    a.add_argument("-se", "--savedepochs", default=0, type=int, action='store')
    a.add_argument("-be", "--best", default=False, type=bool, action='store')
    a.add_argument("-ps", "--plotscatter", default=False, type=bool, action='store')
    a.add_argument("-as", "--assess", default=False, type=bool, action='store')
    a.add_argument("-it", "--iterate", default=False, type=bool, action='store')
    a.add_argument("-im", "--iteratemethod", default='simple', type=str, action='store')
    a.add_argument("-is", "--iteratesmooth", default=0, type=int, action='store')
    a.add_argument("-ss", "--smoothsteps", default=0, type=int, action='store')

    return a.parse_args()

if __name__ == "__main__":
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s') 

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info('Using device: '+device+'\n')

    mp.set_start_method('spawn')

    args = parse_args()

    logging.info('packages imported')
    logging.info("os.environ['DISPLAY']: "+str(os.environ['DISPLAY']))
    logging.info('torch.__version__ : '+str(torch.__version__))
    
    #-------------------------------------
    # Manually set variables for this run
    #-------------------------------------

    logging.info('args '+str(args))
    
    subsample_rate = 5      # number of time steps to skip over when creating training and test data
    train_end_ratio = 0.75  # Take training samples from 0 to this far through the dataset
    val_end_ratio = 0.9     # Take validation samples from train_end_ratio to this far through the dataset
 
    for_len = 180    # How long to iteratively predict for
    for_subsample = 1
    start = 0  
    
    routine_seed=30475
    os.environ['PYTHONHASHSEED'] = str(routine_seed)
    np.random.seed(routine_seed)
    torch.manual_seed(routine_seed)
    torch.cuda.manual_seed(routine_seed)
    torch.cuda.manual_seed_all(routine_seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.matmul.allow_tf32 = True

    base_name = args.name+args.land+args.predictionjump+'_'+args.modelstyle+'_histlen'+str(args.histlen)+'_rolllen'+str(args.rolllen)
    model_name = 'MultiModel_'+base_name
    model_dir = '../../../Channel_nn_Outputs/'+model_name
    if not os.path.isdir(model_dir):
       os.system("mkdir %s" % (model_dir))
       os.system("mkdir %s" % (model_dir+'/TRAINING_PLOTS'))
       os.system("mkdir %s" % (model_dir+'/STATS'))
       os.system("mkdir %s" % (model_dir+'/STATS/PLOTS'))
       os.system("mkdir %s" % (model_dir+'/ITERATED_FORECAST'))
       os.system("mkdir %s" % (model_dir+'/ITERATED_FORECAST/PLOTS'))
 
    if args.land == 'IncLand' or args.land == 'Spits':
       y_dim_used = 104
    elif args.land == 'ExcLand':
       y_dim_used = 98
    else:
       raise RuntimeError('ERROR, Whats going on with args.land?!')

    if args.modelstyle == 'ConvLSTM' or args.modelstyle == 'UNetConvLSTM':
       channel_dim = 2
    elif args.dim == '2d':
       channel_dim = 1
    elif args.dim == '3d':
       channel_dim = 1
 
    if args.land == 'Spits':
       MITgcm_dir = '/data/hpcdata/users/racfur/MITgcm/verification/MundayChannelConfig10km_LandSpits/runs/50yr_Cntrl/'
    else:
       MITgcm_dir = '/data/hpcdata/users/racfur/MITgcm/verification/MundayChannelConfig10km_noSpits/runs/50yr_Cntrl/'
    ProcDataFilename = MITgcm_dir+'Dataset_'+args.land+args.predictionjump+'_'+args.modelstyle+'_histlen'+str(args.histlen)+'_rolllen'+str(args.rolllen)
    if args.test: 
       for_len = min(for_len, 20)
       MITgcm_filename = MITgcm_dir+args.predictionjump+'_small_set.nc'
       ProcDataFilename = ProcDataFilename+'_TEST.nc'
    else:
       MITgcm_filename = MITgcm_dir+args.predictionjump+'_data.nc'
       ProcDataFilename = ProcDataFilename+'.nc'
    MITgcm_stats_filename = MITgcm_dir+args.land+'_stats.nc'
    grid_filename = MITgcm_dir+'grid.nc'
    mean_std_file = MITgcm_dir+args.land+'_'+args.predictionjump+'_MeanStd.npz'
 
    print(MITgcm_filename)
    print(ProcDataFilename)
    ds = xr.open_dataset(MITgcm_filename)
    ds.close()

    logging.info('Model ; '+model_name+'\n')
 
    #---------------------------------------
    # Read in mean and std and set channels
    #---------------------------------------
    z_dim    = ( ds.isel( T=slice(0) ) ).sizes['Zmd000038']

    if args.modelstyle == 'ConvLSTM' or args.modelstyle == 'UNetConvLSTM':
       no_phys_in_channels = 3*z_dim + 3                              # Temp, U, V through depth, plus Eta, gTforc and taux 
       no_model_in_channels = no_phys_in_channels + z_dim             # Phys_in channels plus masks
       no_out_channels = 3*z_dim + 1                                  # Temp, U, V through depth, plus Eta
    elif args.dim == '2d':
       no_phys_in_channels = 3*z_dim + 3                              # Temp, U, V through depth, plus Eta, gTforc and taux
       no_model_in_channels = args.histlen * no_phys_in_channels + z_dim # Phys_in channels for each past time, plus masks
       no_out_channels = 3*z_dim+1                                    # Eta, plus Temp, U, V through depth (predict 1 step ahead, even if LF calculated over mutliple steps
    elif args.dim == '3d':
       no_phys_in_channels = 6                                        # Temp, U, V, Eta, gTforc and taux
       no_model_in_channels = args.histlen * no_phys_in_channels + 1  # Phys_in channels for each past time, plus masks
       no_out_channels = 4                                            # Temp, U, V, Eta just once

    inputs_mean, inputs_std, inputs_range, targets_mean, targets_std, targets_range = \
                                          ReadMeanStd(mean_std_file, args.dim, no_phys_in_channels, no_out_channels, z_dim, routine_seed)

    logging.info('no_phys_in_channels ;'+str(no_phys_in_channels)+'\n')
    logging.info('no_model_in_channels ;'+str(no_model_in_channels)+'\n')
    logging.info('no_out_channels ;'+str(no_out_channels)+'\n')
 
    #-------------------------------------------------------------------------------------
    # Read in training and validation Datsets and dataloaders
    #-------------------------------------------------------------------------------------
    if args.landvalue == -999:
       landvalues = inputs_mean
    else:
       landvalues = np.ones(inputs_mean.shape)
       landvalues[:] = args.landvalue

    Train_Dataset = rr.ReadProcData_Dataset(model_name, ProcDataFilename, 0., train_end_ratio, routine_seed)
    Val_Dataset   = rr.ReadProcData_Dataset(model_name, ProcDataFilename, train_end_ratio, val_end_ratio, routine_seed)
    Test_Dataset  = rr.ReadProcData_Dataset(model_name, ProcDataFilename, val_end_ratio, 1., routine_seed)

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

    #---------------
    # Set up models
    #---------------
    h=[]
    optimizer=[]
    for i in range(len(args.modelseeds)):
       h_temp, optimizer_temp = CreateModel( args.modelstyle, no_model_in_channels, no_out_channels, args.learningrate,
                                             args.modelseeds[i], args.padding, ds.isel(T=slice(0)).sizes['X'], y_dim_used,
                                             z_dim, args.kernsize, args.weightdecay )
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
       LoadModel(model_i_name, h[i], optimizer[i], args.savedepochs, 'inf', losses, args.best, args.modelseeds[i]) 
    
    #-------------------------------
    # Assess the multi-model set up 
    #-------------------------------

    #--------------------
    # Plot scatter plots
    #--------------------
    if args.plotscatter:
    
       PlotScatter(model_name, args.dim, train_loader, h, args.savedepochs, 'training', args.normmethod, channel_dim,
                   mean_std_file, no_out_channels, no_phys_in_channels, z_dim, args.land, routine_seed)
       if not args.test: 
          PlotScatter(model_name, args.dim, val_loader, h, args.savedepochs, 'validation', args.normmethod, channel_dim,
                      mean_std_file, no_out_channels, no_phys_in_channels, z_dim, args.land, routine_seed)
          PlotScatter(model_name, args.dim, test_loader, h, args.savedepochs, 'test', args.normmethod, channel_dim, 
                      mean_std_file, no_out_channels, no_phys_in_channels, z_dim, args.land, routine_seed)

    #---------------------
    # Iteratively predict 
    #---------------------
    if args.iterate:

       Iterate_Dataset = rr.MITgcm_Dataset( MITgcm_filename, 0., 1., 1, args.histlen, args.rolllen, args.land, args.bdyweight, landvalues,
                                            grid_filename, args.dim,  args.modelstyle, no_phys_in_channels, no_out_channels, routine_seed,
                                            transform = transforms.Compose( [ rr.RF_Normalise_sample(inputs_mean, inputs_std, inputs_range,
                                                                              targets_mean, targets_std, targets_range,
                                                                              args.histlen, args.rolllen, no_phys_in_channels, no_out_channels, args.dim, 
                                                                              args.normmethod, args.modelstyle, routine_seed)] ) )
    
       IterativelyPredict(model_name, args.modelstyle, MITgcm_filename, Iterate_Dataset, h, start, for_len, args.savedepochs,
                          y_dim_used, args.land, args.dim, args.histlen, landvalues, args.iteratemethod, args.iteratesmooth, args.smoothsteps,
                          args.normmethod, channel_dim, mean_std_file, for_subsample, no_phys_in_channels, no_out_channels, routine_seed) 
    
    #------------------
    # Assess the model 
    #------------------
    if args.assess:
    
       stats_train_loader = torch.utils.data.DataLoader(Train_Dataset, batch_size=args.batchsize, shuffle=True,
                                                  num_workers=args.numworkers, pin_memory=True )
       if not args.test: 
          stats_val_loader   = torch.utils.data.DataLoader(Val_Dataset,  batch_size=args.batchsize, shuffle=True, 
                                                  num_workers=args.numworkers, pin_memory=True )
          stats_test_loader  = torch.utils.data.DataLoader(Test_Dataset,  batch_size=args.batchsize, shuffle=True, 
                                                  num_workers=args.numworkers, pin_memory=True )
    
       OutputStats(model_name, args.modelstyle, MITgcm_filename, stats_train_loader, h, args.savedepochs, y_dim_used, args.dim, 
                   args.histlen, args.land, 'training', args.normmethod, channel_dim, mean_std_file, no_phys_in_channels, no_out_channels,
                   MITgcm_stats_filename, routine_seed)
    
       if not args.test: 
          OutputStats(model_name, args.modelstyle, MITgcm_filename, stats_val_loader, h, args.savedepochs, y_dim_used, args.dim, 
                      args.histlen, args.land, 'validation', args.normmethod, channel_dim, mean_std_file, no_phys_in_channels, no_out_channels,
                      MITgcm_stats_filename, routine_seed)
    
          OutputStats(model_name, args.modelstyle, MITgcm_filename, stats_test_loader, h, args.savedepochs, y_dim_used, args.dim, 
                      args.histlen, args.land, 'test', args.normmethod, channel_dim, mean_std_file, no_phys_in_channels, no_out_channels,
                      MITgcm_stats_filename, routine_seed) 
