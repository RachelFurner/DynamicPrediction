#!/usr/bin/env python
# coding: utf-8

#----------------------------
# Import neccessary packages
#----------------------------
import sys
sys.path.append('/data/hpcdata/users/racfur/DynamicPrediction/code_git/')
from Tools import CreateDataName as cn
from Tools import ReadRoutines as rr
from Tools import AssessModel as am
from Tools import Model_Plotting as rfplt

import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn import linear_model
import sklearn.metrics as metrics
from sklearn.model_selection import GridSearchCV
import torch as torch

torch.cuda.empty_cache()

plt.rcParams.update({'font.size': 14})

#----------------------------
# Set variables for this run
#----------------------------
run_vars={'dimension':3, 'lat':True , 'lon':True , 'dep':True , 'current':True , 'sal':True , 'eta':True , 'poly_degree':2}

DIR = '/data/hpcdata/users/racfur/MITGCM_OUTPUT/20000yr_Windx1.00_mm_diag/'
MITGCM_filename=DIR+'cat_tave_2000yrs_SelectedVars_masked.nc'

#---------------------------
# calculate other variables 
#---------------------------
data_name = cn.create_dataname(run_vars)

#data_name = 'Steps1to10_'+data_name

#--------------------------------------------------------------
# Call module to read in the data, or open it from saved array
#--------------------------------------------------------------
denorm_inputs_tr, denorm_inputs_val, denorm_inputs_te, denorm_outputs_tr, denorm_outputs_val, denorm_outputs_te = rr.ReadMITGCM(MITGCM_filename, 0.7, 0.9, data_name, run_vars)

#-----------------------------
# Plot histograms of the data
#-----------------------------
fig = rfplt.Plot_Histogram(denorm_outputs_tr, 100)  
plt.savefig('/data/hpcdata/users/racfur/DynamicPrediction/INPUT_OUTPUT_ARRAYS/'+data_name+'_TrainOutputsHistogram', bbox_inches = 'tight', pad_inches = 0.1)

fig = rfplt.Plot_Histogram(denorm_outputs_val, 100)  
plt.savefig('/data/hpcdata/users/racfur/DynamicPrediction/INPUT_OUTPUT_ARRAYS/'+data_name+'_ValOutputsHistogram', bbox_inches = 'tight', pad_inches = 0.1)

fig = rfplt.Plot_Histogram(denorm_outputs_te, 100)  
plt.savefig('/data/hpcdata/users/racfur/DynamicPrediction/INPUT_OUTPUT_ARRAYS/'+data_name+'_TestOutputsHistogram', bbox_inches = 'tight', pad_inches = 0.1)

