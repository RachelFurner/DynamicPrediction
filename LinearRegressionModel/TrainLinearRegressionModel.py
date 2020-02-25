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

import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn import linear_model
import sklearn.metrics as metrics
from sklearn.model_selection import GridSearchCV
import torch as torch

torch.cuda.empty_cache()

import struct
print(struct.calcsize("P") * 8)

#----------------------------
# Set variables for this run
#----------------------------
run_vars={'dimension':3, 'lat':False, 'lon':True , 'dep':True , 'current':True , 'sal':True , 'eta':True , 'poly_degree':2}
model_type = 'lr'
exp_prefix = ''

DIR = '/data/hpcdata/users/racfur/MITGCM_OUTPUT/20000yr_Windx1.00_mm_diag/'
MITGCM_filename=DIR+'cat_tave_5000yrs_SelectedVars_masked.nc'

#---------------------------
# calculate other variables 
#---------------------------
data_name = cn.create_dataname(run_vars)
exp_name = exp_prefix+data_name

## Calculate x,y position in each feature list for extracting 'now' value
if run_vars['dimension'] == 2:
   xy_pos = 4
elif run_vars['dimension'] == 3:
   xy_pos = 13
else:
   print('ERROR, dimension neither two or three!!!')

#--------------------------------------------------------------
# Open data from saved array
#--------------------------------------------------------------
inputsoutputs_file = '/data/hpcdata/users/racfur/DynamicPrediction/INPUT_OUTPUT_ARRAYS/SinglePoint_'+data_name+'_InputsOutputs.npz'
norm_inputs_tr, norm_inputs_te, norm_outputs_tr, norm_outputs_te = np.load(inputsoutputs_file).values()

#-------------------------------------------------------------
# Set up a model in scikitlearn to predict deltaT (the trend)
# Run ridge regression tuning alpha through cross val
#-------------------------------------------------------------

alpha_s = [0.00, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10, 30]
parameters = {'alpha': alpha_s}
n_folds=3

lr = linear_model.Ridge(fit_intercept=True)

# set up regressor
lr_cv = GridSearchCV(lr, param_grid=parameters, cv=n_folds, scoring='neg_mean_squared_error', refit=True)

# fit the model
lr.fit(norm_inputs_tr, norm_outputs_tr)
lr.get_params()

# pickle it
pkl_filename = '/data/hpcdata/users/racfur/DynamicPrediction/lr_Outputs/MODELS/pickle_'+model_type+'_'+exp_name+'.pkl'
with open(pkl_filename, 'wb') as pckl_file:
    pickle.dump(lr, pckl_file)

# predict values and calculate the error
lr_predicted_tr = lr.predict(norm_inputs_tr)
lr_tr_mse = metrics.mean_squared_error(norm_outputs_tr, lr_predicted_tr)

lr_predicted_te = lr.predict(norm_inputs_te)
lr_te_mse = metrics.mean_squared_error(norm_outputs_te, lr_predicted_te)

#------------------
# Assess the model
#------------------

am.get_stats(model_type, exp_name, norm_outputs_tr, norm_outputs_te, lr_predicted_tr, lr_predicted_te)
am.plot_results(model_type, data_name, exp_name, norm_outputs_tr, norm_outputs_te, lr_predicted_tr, lr_predicted_te)

