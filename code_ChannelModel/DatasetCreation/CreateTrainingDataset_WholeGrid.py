#!/usr/bin/env python
# coding: utf-8

#----------------------------
print('Import neccessary packages')
#----------------------------
from comet_ml import Experiment

import sys
sys.path.append('../../code_git/')
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

#-------------------- -----------------
# Manually set variables for this run
#--------------------------------------
run_vars={'dimension':3, 'lat':True , 'lon':True , 'dep':True , 'current':True , 'bolus_vel':True , 'sal':True , 'eta':True , 'density':True , 'poly_degree':2}

time_step = '1mnth'

DIR = '/data/hpcdata/users/racfur/MITGCM_OUTPUT/20000yr_Windx1.00_mm_diag/'
MITGCM_filename=DIR+'cat_tave_2000yrs_SelectedVars_masked_withBolus.nc'

data_prefix = 'NormalisePerChannel_'

data_name = cn.create_dataname(run_vars)
data_name = time_step+'_'+data_prefix+data_name

#--------------------------------------------------------------
# Call module to read in the data, or open it from saved array
#--------------------------------------------------------------
rr.ReadMITGCMfield(MITGCM_filename, 0.7, 0.9, data_name)
# no need to save here as saved in the Read Routine
