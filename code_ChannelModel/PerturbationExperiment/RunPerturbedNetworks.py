#!/usr/bin/env pyton
# coding: utf-8

# Code to train a network to learn the dynamic evolution of an MITgcm channel configuration
# Designed to take the entire field for all variables, apply a NN (with conv layers), and output
# the entire field for all variables one day later (i.e. the next iteration of the MITgcm netcdf 
# file here, although note this is 2 steps of the underlying model which runs a 12 hourly timestep.
# The data is subsampled in time to give quasi-independence

import torch
from torchvision import transforms, utils
import os
import sys
sys.path.append('../Tools')
import ReadRoutines as rr
from WholeGridNetworkRegressorModules import *
from Models import CreateModel
import numpy as np
import xarray as xr
import netCDF4 as nc4
import gc
import argparse
import logging
import multiprocessing as mp

def parse_args():
    a = argparse.ArgumentParser()

    a.add_argument("-na", "--name", default="", action='store')
    a.add_argument("-ms", "--modelstyle", default='UNet2dtransp', type=str, action='store')
    a.add_argument("-di", "--dim", default='2d', type=str, action='store')
    a.add_argument("-la", "--land", default='IncLand', type=str, action='store')
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
    a.add_argument("-sd", "--seed", default=30475, type=int, action='store')
    a.add_argument("-lv", "--landvalue", default=0., type=float, action='store')
    a.add_argument("-lo", "--loadmodel", default=False, type=bool, action='store')
    a.add_argument("-be", "--best", default=False, type=bool, action='store')
    a.add_argument("-tr", "--trainmodel", default=False, type=bool, action='store')
    a.add_argument("-it", "--iterate", default=False, type=bool, action='store')
    a.add_argument("-im", "--iteratemethod", default='simple', type=str, action='store')
    a.add_argument("-is", "--iteratesmooth", default=0, type=int, action='store')
    a.add_argument("-ss", "--smoothsteps", default=0, type=int, action='store')

    return a.parse_args()

if __name__ == "__main__":
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s') 

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info('Using device: '+device+'\n')
    
    #mp.set_start_method('spawn')

    args = parse_args()

    logging.info('packages imported')
    logging.info('matplotlib backend: '+str(matplotlib.get_backend()) )
    logging.info("os.environ['DISPLAY']: "+str(os.environ['DISPLAY']))
    logging.info('torch.__version__ : '+str(torch.__version__))
    logging.info('args '+str(args))
    
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.matmul.allow_tf32 = True

    for_subsample = 1
    epochs = 200
    y_dim_used = 104
    channel_dim = 1

    mean_std_file = '/data/hpcdata/users/racfur/MITgcm/verification/MundayChannelConfig10km_noSpits/runs/50yr_Cntrl/IncLand_12hrly_MeanStd.npz'
    model_dir = '../../../Channel_nn_Outputs/Perturbed_runs'

    #-------------------------------------
    # Manually set variables for this run
    #-------------------------------------
    perturb=False
    if perturb == True:
       model_name = 'PerturbationExp_Perturbed'
       start = 0        # Start point - zero if pertubed run, correct location in Control if Control run
       MITgcm_dir = '/data/hpcdata/users/racfur/MITgcm/verification/MundayChannelConfig10km_noSpits/runs/6mnthPerturbed2593440/'
       MITgcm_filename = MITgcm_dir+args.predictionjump+'_data.nc'
       grid_filename = MITgcm_dir+'grid.nc'
       for_len = 180    # How long to iteratively predict for
    else:
       model_name = 'PerturbationExp_Cntrl'
       start = 20       # Start point - zero if pertubed run, correct location in Control if Control run
       MITgcm_dir = '/data/hpcdata/users/racfur/MITgcm/verification/MundayChannelConfig10km_noSpits/runs/50yr_Cntrl/'
       MITgcm_filename = MITgcm_dir+'1yr_dataset.nc'
       grid_filename = MITgcm_dir+'grid.nc'
       for_len = 180    # How long to iteratively predict for
    
    #-------------------------------------
    
    print(MITgcm_filename)
    ds = xr.open_dataset(MITgcm_filename)
    ds.close()

    logging.info('Model ; '+model_name+'\n')
    
    #----------------------------------------
    # Read in mean and std, and set channels
    #----------------------------------------
    z_dim = ( ds.isel( T=slice(0) ) ).sizes['Zmd000038'] 
    no_phys_in_channels = 3*z_dim + 3                       # Temp, U, V through depth, plus Eta, gTforc and taux
    no_model_in_channels = args.histlen * no_phys_in_channels + 3*z_dim # Phys_in channels for each past time, plus masks
    no_out_channels = 3*z_dim+1                             # Eta, plus Temp, U, V through depth (predict 1 step ahead, even for rolout loss)
   
    inputs_mean, inputs_std, inputs_range, targets_mean, targets_std, targets_range = \
                                          ReadMeanStd(mean_std_file, args.dim, no_phys_in_channels, no_out_channels, z_dim, args.seed)

    landvalues = np.ones(inputs_mean.shape)
    landvalues[:] = args.landvalue
    logging.info('no_phys_in_channels ;'+str(no_phys_in_channels)+'\n')
    logging.info('no_model_in_channels ;'+str(no_model_in_channels)+'\n')
    logging.info('no_out_channels ;'+str(no_out_channels)+'\n')

       
    #--------------
    # Set up model
    #--------------
    h, optimizer = CreateModel( args.modelstyle, no_model_in_channels, no_out_channels, args.learningrate, args.seed, args.padding,
                                ds.isel(T=slice(0)).sizes['X'], y_dim_used, z_dim, args.kernsize, args.weightdecay )
   
    #-----------------
    # Load the model:
    #-----------------
    # define dictionary to store losses at each epoch
    losses = {'train'     : [],
              'train_Temp': [],
              'train_U'   : [],
              'train_V'   : [],
              'train_Eta' : [],
              'val'       : [] }
 
    LoadModel('IncLand12hrly_UNet2dtransp_histlen1_rolllen1_seed30475',
               h, optimizer, epochs, 'inf', losses, args.best, args.seed)
   
    #---------------------
    # Iteratively predict 
    #---------------------
    Iterate_Dataset = rr.MITgcm_Dataset( MITgcm_filename, 0., 1., 1, args.histlen, args.rolllen, args.land, args.bdyweight, landvalues,
                                         grid_filename, args.dim,  args.modelstyle, no_phys_in_channels, no_out_channels, args.seed,
                                         transform = transforms.Compose( [ rr.RF_Normalise_sample(inputs_mean, inputs_std, inputs_range,
                                                                           targets_mean, targets_std, targets_range, args.histlen,
                                                                           args.rolllen, no_phys_in_channels, no_out_channels, args.dim, 
                                                                           args.normmethod, args.modelstyle, args.seed)] ) )
    
    IterativelyPredict(model_name, args.modelstyle, MITgcm_filename, Iterate_Dataset, h, start, for_len, epochs,
                       y_dim_used, args.land, args.dim, args.histlen, landvalues, args.iteratemethod, args.iteratesmooth, args.smoothsteps,
                       args.normmethod, channel_dim, mean_std_file, for_subsample, no_phys_in_channels, no_out_channels, args.seed, 'singlepred') 
