#!/usr/bin/env python
# coding: utf-8

#----------------------------
print('Import neccessary packages')
#----------------------------
from comet_ml import Experiment

import sys
sys.path.append('/data/hpcdata/users/racfur/DynamicPrediction/code_git/')
from Tools import CreateDataName as cn
from Tools import ReadRoutines as rr
from Tools import AssessModel as am

import numpy as np
import os
import xarray as xr
import matplotlib.pyplot as plt
import pickle

from sklearn.preprocessing import PolynomialFeatures
import sklearn.metrics as metrics

from torch.utils import data
import torch.nn as nn
from torch.autograd import Variable
import torch


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using device:', device)
print()

torch.cuda.empty_cache()

#-------------------- -----------------
# Manually set variables for this run
#--------------------------------------
run_vars={'dimension':3, 'lat':True , 'lon':False, 'dep':True , 'current':True , 'sal':True , 'eta':True , 'poly_degree':2}

DIR = '/data/hpcdata/users/racfur/MITGCM_OUTPUT/20000yr_Windx1.00_mm_diag/'
MITGCM_filename=DIR+'cat_tave_2000yrs_SelectedVars_masked.nc'

hyper_params = {
    "batch_size": 32,
    "num_epochs": 200, 
    "learning_rate": 0.001,
    "criterion": torch.nn.MSELoss(),
    "no_layers": 0,    # no of *hidden* layers
    "no_nodes": 1000
}

data_name = cn.create_dataname(run_vars)

#--------------------------------------------------------------
# Call module to read in the data, or open it from saved array
#--------------------------------------------------------------
norm_inputs_tr, norm_inputs_te, norm_outputs_tr, norm_outputs_te = rr.ReadMITGCMfield(MITGCM_filename, 0.7, data_name)
# no need to save here as saved in the Read Routine
