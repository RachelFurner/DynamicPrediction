#!/usr/bin/env python
# coding: utf-8

#----------------------------
# Import neccessary packages
#----------------------------
import sys
sys.path.append('../')
from Tools import CreateDataName as cn
from Tools import ReadRoutines as rr
from Tools import AssessModel as am
from Tools import Model_Plotting as rfplt

import numpy as np
import matplotlib.pyplot as plt
import pickle
import xarray as xr

from sklearn import linear_model
import sklearn.metrics as metrics
from sklearn.model_selection import GridSearchCV
import torch as torch

torch.cuda.empty_cache()

plt.rcParams.update({'font.size': 14})

#----------------------------
# Set variables for this run
#----------------------------
run_vars = { 'dimension':3, 'lat':True , 'lon':True , 'dep':True , 'current':True ,
             'bolus_vel':True , 'sal':True , 'eta':True , 'density':True , 'poly_degree':2 }

time_step = '24hrs'
data_prefix = ''

#---------------------------
# calculate other variables 
#---------------------------
data_name = cn.create_dataname(run_vars)
data_name = data_prefix+data_name+'_'+time_step

if time_step == '1mnth':
   DIR = '/data/hpcdata/users/racfur/MITGCM_OUTPUT/20000yr_Windx1.00_mm_diag/'
   MITGCM_filename=DIR+'cat_tave_2000yrs_SelectedVars_masked_withBolus.nc'
   density_file = DIR+'DensityData.npy'
elif time_step == '24hrs':
   DIR = '/data/hpcdata/users/racfur/MITGCM_OUTPUT/100yr_Windx1.00_FrequentOutput/'
   MITGCM_filename=DIR+'cat_tave_50yr_SelectedVars_masked_withBolus.nc'
   density_file = DIR+'DensityData.npy'

#---------------------------------
# Call module to read in the data
#---------------------------------
norm_inputs_tr, norm_inputs_val, norm_inputs_te, norm_outputs_tr, norm_outputs_val, norm_outputs_te = rr.ReadMITGCM(MITGCM_filename, density_file, 0.7, 0.9, data_name, run_vars, time_step=time_step, save_arrays=True, plot_histograms=True)
