#!/usr/bin/env python
# coding: utf-8

#----------------------------
# Import neccessary packages
#----------------------------
import sys
sys.path.append('/data/hpcdata/users/racfur/DynamicPrediction/code_git/')
from Tools import CreateDataName as cn
#from Tools import ReadRoutines as rr
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

#----------------------------
# Set variables for this run
#----------------------------
run_vars={'dimension':3, 'lat':True , 'lon':True , 'dep':True , 'current':True , 'sal':True , 'eta':False, 'poly_degree':2}
model_type = 'lr'
#exp_prefix = 'BalancedTrainingSet_'
exp_prefix = ''

ModifyTrainingBalance = False

DIR = '/data/hpcdata/users/racfur/MITGCM_OUTPUT/20000yr_Windx1.00_mm_diag/'
MITGCM_filename=DIR+'cat_tave_2000yrs_SelectedVars_masked.nc'

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
print('reading data')
inputsoutputs_file = '/data/hpcdata/users/racfur/DynamicPrediction/INPUT_OUTPUT_ARRAYS/SinglePoint_'+data_name+'_InputsOutputs.npz'
norm_inputs_tr, norm_inputs_te, norm_outputs_tr, norm_outputs_te = np.load(inputsoutputs_file).values()

# if neccessary, modify the training data to give a better balance of dynamicity of points
#if ModifyTrainingBalance:
   # use where on outputs to find training samples which fall into certain bins of dynamic activity.
   # take first n of each of these (samples already random, so shouldn't create any systematic bias/issue here)

#-------------------------------------------------------------
# Set up a model in scikitlearn to predict deltaT (the trend)
# Run ridge regression tuning alpha through cross val
#-------------------------------------------------------------
print('training model')

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
print('predict values')
lr_predicted_tr = lr.predict(norm_inputs_tr)
lr_tr_mse = metrics.mean_squared_error(norm_outputs_tr, lr_predicted_tr)

lr_predicted_te = lr.predict(norm_inputs_te)
lr_te_mse = metrics.mean_squared_error(norm_outputs_te, lr_predicted_te)

#------------------
# Assess the model
#------------------
print('get stats')
am.get_stats(model_type, data_name, exp_name, norm_outputs_tr, norm_outputs_te, lr_predicted_tr, lr_predicted_te)
print('plot results')
am.plot_results(model_type, data_name, exp_name, norm_outputs_tr, norm_outputs_te, lr_predicted_tr, lr_predicted_te)


#-------------------------------------------------------------------
# Plot some extra stuff.....
#-------------------------------------------------

#first de-normalise
#Read in mean and std to normalise inputs
norm_file=open('/data/hpcdata/users/racfur/DynamicPrediction/NORMALISING_PARAMS/NormalisingParameters_SinglePoint_'+data_name+'.txt',"r")
count = len(norm_file.readlines(  ))
input_mean=[]
input_std =[]
norm_file.seek(0)
for i in range( int( (count-4)/4) ):
   a_str = norm_file.readline()
   a_str = norm_file.readline() ;  input_mean.append(a_str.split())
   a_str = norm_file.readline()
   a_str = norm_file.readline() ;  input_std.append(a_str.split())
a_str = norm_file.readline()
a_str = norm_file.readline() ;  output_mean = float(a_str.split()[0])
a_str = norm_file.readline()
a_str = norm_file.readline() ;  output_std = float(a_str.split()[0])
norm_file.close()
input_mean = np.array(input_mean).astype(float)
input_std  = np.array(input_std).astype(float)
input_mean = input_mean.reshape(1,input_mean.shape[0])
input_std  = input_std.reshape(1,input_std.shape[0])
print(output_mean, output_std)
print(input_mean.shape, input_std.shape)
print(input_mean, input_std)

# denormalise the predictions and truth   
denorm_lr_predicted_tr = lr_predicted_tr*output_std+output_mean
denorm_lr_predicted_te = lr_predicted_te*output_std+output_mean
denorm_outputs_tr = norm_outputs_tr*output_std+output_mean
denorm_outputs_te = norm_outputs_te*output_std+output_mean

# plot histograms:
fig = rfplt.Plot_Histogram(denorm_outputs_tr, 100)  # Remove points not being predicted (boundaries) from this
plt.savefig('../../'+model_type+'_Outputs/PLOTS/'+model_type+'_'+exp_name+'_denorm_truth_tr_hist', bbox_inches = 'tight', pad_inches = 0.1)

fig = rfplt.Plot_Histogram(denorm_outputs_te, 100)  # Remove points not being predicted (boundaries) from this
plt.savefig('../../'+model_type+'_Outputs/PLOTS/'+model_type+'_'+exp_name+'_denorm_truth_te_hist', bbox_inches = 'tight', pad_inches = 0.1)

fig = rfplt.Plot_Histogram(denorm_lr_predicted_tr, 100)  # Remove points not being predicted (boundaries) from this
plt.savefig('../../'+model_type+'_Outputs/PLOTS/'+model_type+'_'+exp_name+'_denorm_prediction_tr_hist', bbox_inches = 'tight', pad_inches = 0.1)

fig = rfplt.Plot_Histogram(denorm_lr_predicted_te, 100)  # Remove points not being predicted (boundaries) from this
plt.savefig('../../'+model_type+'_Outputs/PLOTS/'+model_type+'_'+exp_name+'_denorm_predictons_te_hist', bbox_inches = 'tight', pad_inches = 0.1)

# Plot a histogram of predictions minus the truth
fig = rfplt.Plot_Histogram(denorm_lr_predicted_tr-denorm_outputs_tr, 100)  # Remove points not being predicted (boundaries) from this
plt.savefig('../../'+model_type+'_Outputs/PLOTS/'+model_type+'_'+exp_name+'_denorm_errors_tr_hist', bbox_inches = 'tight', pad_inches = 0.1)

fig = rfplt.Plot_Histogram(denorm_lr_predicted_te-denorm_outputs_te, 100)  # Remove points not being predicted (boundaries) from this
plt.savefig('../../'+model_type+'_Outputs/PLOTS/'+model_type+'_'+exp_name+'_denorm_errors_te_hist', bbox_inches = 'tight', pad_inches = 0.1)
