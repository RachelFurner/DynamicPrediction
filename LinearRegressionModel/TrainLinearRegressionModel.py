#!/usr/bin/env python
# coding: utf-8

#----------------------------
# Import neccessary packages
#----------------------------
import sys
sys.path.append('/data/hpcdata/users/racfur/DynamicPrediction/code_git/')
from Tools import CreateExpName as cn
from Tools import ReadRoutines as rr
from Tools import AssessModel as am

import numpy as np
import os
import xarray as xr
import matplotlib.pyplot as plt
import pickle

from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report
import sklearn.metrics as metrics
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures

from torch.utils import data
import torch
from torch.autograd import Variable

torch.cuda.empty_cache()

#----------------------------
# Set variables for this run
#----------------------------
from Tools import CreateExpName as cn
run_vars={'dimension':3, 'lat':True, 'lon':True, 'dep':True, 'current':True, 'sal':True, 'eta':False, 'poly_degree':1}
model_type = 'lr'

DIR = '/data/hpcdata/users/racfur/MITGCM_OUTPUT/20000yr_Windx1.00_mm_diag/'
MITGCM_filename=DIR+'cat_tave_5000yrs_SelectedVars_masked.nc'

read_data=True

#---------------------------
# calculate other variables 
#---------------------------
exp_name = cn.create_expname(model_type, run_vars)

## Calculate x,y position in each feature list for extracting 'now' value
if run_vars['dimension'] == 2:
   xy_pos = 4
elif run_vars['dimension'] == 3:
   xy_pos = 13
else:
   print('ERROR, dimension neither two or three!!!')

#--------------------------------------------------------------
# Call module to read in the data, or open it from saved array
#--------------------------------------------------------------
if read_data:
   norm_inputs_tr, norm_inputs_te, norm_outputs_tr, norm_outputs_te = rr.ReadMITGCM(MITGCM_filename, 0.7, exp_name, run_vars)
else:
   inputsoutputs_file = '/data/hpcdata/users/racfur/DynamicPrediction/INPUT_OUTPUT_ARRAYS/'+exp_name+'_InputsOutputs.npz'
   norm_inputs_tr, norm_inputs_te, norm_outputs_tr, norm_outputs_te = np.load(inputsoutputs_file).values()

#-------------------------------------------------------------------------------------------------------------
# Plot temp at time t against outputs (temp at time t+1), to see if variance changes with input values
#-------------------------------------------------------------------------------------------------------------
fig = plt.figure(figsize=(14, 7))

ax1 = fig.add_subplot(121)
ax1.scatter(norm_inputs_tr[:,xy_pos], norm_outputs_tr, edgecolors=(0, 0, 0))
ax1.set_xlabel('Inputs (t)')
ax1.set_ylabel('Outputs (t+1)')
ax1.set_title('Training Data')

ax2 = fig.add_subplot(122)
ax2.scatter(norm_inputs_te[:,xy_pos], norm_outputs_te, edgecolors=(0, 0, 0))
ax2.set_xlabel('Inputs (t)')
ax2.set_ylabel('Outputs (t+1)')
ax2.set_title('Validation Data')

plt.savefig('../../RegressionOutputs/PLOTS/'+model_type+'_'+exp_name+'_InputsvsOutputs', bbox_inches = 'tight', pad_inches = 0.1)

##----------------------------------------------------------------------------------------------------------------
## Calculate 'persistance' score, to give a baseline - this is just zero everywhere as we're predicting the trend
##----------------------------------------------------------------------------------------------------------------
#
## For training data
#predict_persistance_tr = np.zeros(norm_outputs_tr.shape)
#pers_tr_mse = metrics.mean_squared_error(norm_outputs_tr, predict_persistance_tr)
## For validation data
#predict_persistance_te = np.zeros(norm_outputs_te.shape)  
#pers_te_mse = metrics.mean_squared_error(norm_outputs_te, predict_persistance_te)
#
#
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
pkl_filename = '/data/hpcdata/users/racfur/DynamicPrediction/RegressionOutputs/MODELS/pickle_'+model_type+'_'+exp_name+'.pkl'
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
am.plot_results(model_type, exp_name, norm_outputs_tr, norm_outputs_te, lr_predicted_tr, lr_predicted_te)

