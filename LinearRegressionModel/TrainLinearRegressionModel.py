#!/usr/bin/env python
# coding: utf-8

#----------------------------
# Import neccessary packages
#----------------------------
import sys
sys.path.append('/data/hpcdata/users/racfur/DynamicPrediction/code_git/')
from Tools import CreateDataName as cn
from Tools import AssessModel as am
from Tools import Model_Plotting as rfplt

import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn import linear_model
import sklearn.metrics as metrics
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

import torch as torch

torch.cuda.empty_cache()
plt.rcParams.update({'font.size': 14})

#----------------------------
# Set variables for this run
#----------------------------
run_vars = {'dimension':2, 'lat':True , 'lon':True , 'dep':True , 'current':True , 'bolus_vel':False,'sal':True , 'eta':True , 'density':False, 'poly_degree':2}
model_type = 'lr'
time_step = '24hrs'
#time_step = '1mnth'
data_prefix = ''
exp_prefix = ''

TrainModel = True

if time_step == '1mnth':
   DIR = '/data/hpcdata/users/racfur/MITGCM_OUTPUT/20000yr_Windx1.00_mm_diag/'
   MITGCM_filename=DIR+'cat_tave_2000yrs_SelectedVars_masked_withBolus.nc'
elif time_step == '24hrs':
   DIR = '/data/hpcdata/users/racfur/MITGCM_OUTPUT/100yr_Windx1.00_FrequentOutput/'
   MITGCM_filename=DIR+'cat_tave_50yr_SelectedVars_masked_withBolus.nc'
else:
   print('ERROR - No proper time step given')

#---------------------------
# calculate other variables 
#---------------------------
data_name = cn.create_dataname(run_vars)
data_name = time_step+'_'+data_prefix+data_name

exp_name = exp_prefix+data_name

#--------------------------------------------------------------
# Open data from saved array
#--------------------------------------------------------------
print('reading data')
inputsoutputs_file = '/data/hpcdata/users/racfur/DynamicPrediction/INPUT_OUTPUT_ARRAYS/SinglePoint_'+data_name+'_InputsOutputs.npz'
norm_inputs_tr, norm_inputs_val, norm_inputs_te, norm_outputs_tr, norm_outputs_val, norm_outputs_te = np.load(inputsoutputs_file).values()

print(norm_inputs_tr.shape)
print(norm_inputs_val.shape)
print(norm_outputs_tr.shape)
print(norm_outputs_val.shape)
#-------------------------------------------------------------
# Set up a model in scikitlearn to predict deltaT (the trend)
# Run ridge regression tuning alpha through cross val
#-------------------------------------------------------------
pkl_filename = '/data/hpcdata/users/racfur/DynamicPrediction/lr_Outputs/MODELS/pickle_'+exp_name+'.pkl'
if TrainModel:
    print('training model')
    
    alpha_s = [0.0001, 0.001, 0.01, 0.1, 1.0]
    parameters = [{'alpha': alpha_s}]
    n_folds=3
    
    lr = linear_model.Ridge(fit_intercept=True)
    
    # set up regressor
    lr = GridSearchCV(lr, param_grid=parameters, cv=n_folds, scoring='neg_mean_squared_error', refit=True)
    
    # fit the model
    lr.fit(norm_inputs_tr, norm_outputs_tr)
    lr.get_params()
    
    print("Best parameters set found on development set:")
    print()
    print(lr.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = lr.cv_results_['mean_test_score']
    stds = lr.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, lr.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    # pickle it
    with open(pkl_filename, 'wb') as pckl_file:
        pickle.dump(lr, pckl_file)

with open(pkl_filename, 'rb') as file:
   print('opening '+pkl_filename)
   lr = pickle.load(file)

# predict values
print('predict values')
norm_lr_predicted_tr = lr.predict(norm_inputs_tr)

norm_lr_predicted_val = lr.predict(norm_inputs_val)

#------------------------------------------
# De-normalise the outputs and predictions
#------------------------------------------
mean_std_file = '/data/hpcdata/users/racfur/DynamicPrediction/INPUT_OUTPUT_ARRAYS/SinglePoint_'+data_name+'_MeanStd.npz'
input_mean, input_std, output_mean, output_std = np.load(mean_std_file).values()

# denormalise inputs
def denormalise_data(norm_data,mean,std):
    denorm_data = norm_data * std + mean
    return denorm_data

denorm_inputs_tr  = np.zeros(norm_inputs_tr.shape)
denorm_inputs_val = np.zeros(norm_inputs_val.shape)
for i in range(norm_inputs_tr.shape[1]):  
    denorm_inputs_tr[:, i]  = denormalise_data(norm_inputs_tr[:, i] , input_mean[i], input_std[i])
    denorm_inputs_val[:, i] = denormalise_data(norm_inputs_val[:, i], input_mean[i], input_std[i])

# denormalise the predictions and true outputs   
denorm_lr_predicted_tr = denormalise_data(norm_lr_predicted_tr, output_mean, output_std)
denorm_lr_predicted_val = denormalise_data(norm_lr_predicted_val, output_mean, output_std)
denorm_outputs_tr = denormalise_data(norm_outputs_tr, output_mean, output_std)
denorm_outputs_val = denormalise_data(norm_outputs_val, output_mean, output_std)

#------------------
# Assess the model
#------------------
# Calculate 'persistance' score - persistence prediction is just zero everywhere as we're predicting the trend
predict_persistance_tr = np.zeros(denorm_outputs_tr.shape)
predict_persistance_val = np.zeros(denorm_outputs_val.shape)

print('get stats')
am.get_stats(model_type, exp_name, name1='Training', truth1=denorm_outputs_tr, exp1=denorm_lr_predicted_tr, pers1=predict_persistance_tr,
                                name2='Validation',  truth2=denorm_outputs_val, exp2=denorm_lr_predicted_val, pers2=predict_persistance_val, name='TrainVal')

print('plot results')
am.plot_results(model_type, exp_name, denorm_outputs_tr, denorm_lr_predicted_tr, name='training')
am.plot_results(model_type, exp_name, denorm_outputs_val, denorm_lr_predicted_val, name='val')

#-------------------------------------------------------------------
# plot histograms:
#-------------------------------------------------
fig = rfplt.Plot_Histogram(denorm_lr_predicted_tr, 100) 
plt.savefig('../../'+model_type+'_Outputs/PLOTS/'+exp_name+'_histogram_train_predictions', bbox_inches = 'tight', pad_inches = 0.1)

fig = rfplt.Plot_Histogram(denorm_lr_predicted_val, 100)
plt.savefig('../../'+model_type+'_Outputs/PLOTS/'+exp_name+'_histogram_val_predictions', bbox_inches = 'tight', pad_inches = 0.1)

fig = rfplt.Plot_Histogram(denorm_lr_predicted_tr-denorm_outputs_tr, 100)
plt.savefig('../../'+model_type+'_Outputs/PLOTS/'+exp_name+'_histogram_train_errors', bbox_inches = 'tight', pad_inches = 0.1)

fig = rfplt.Plot_Histogram(denorm_lr_predicted_val-denorm_outputs_val, 100) 
plt.savefig('../../'+model_type+'_Outputs/PLOTS/'+exp_name+'_histogram_val_errors', bbox_inches = 'tight', pad_inches = 0.1)

#---------------------------------------------------------
# Plot scatter plots of errors against outputs and inputs
#---------------------------------------------------------
am.plot_results(model_type, exp_name, denorm_outputs_tr, denorm_lr_predicted_tr-denorm_outputs_tr, name='training', xlabel='DeltaT', ylabel='Errors', exp_cor=False)
am.plot_results(model_type, exp_name, denorm_outputs_val, denorm_lr_predicted_val-denorm_outputs_val, name='val', xlabel='DeltaT', ylabel='Errors', exp_cor=False)

print('')
print(norm_inputs_tr.shape)
print(norm_inputs_val.shape)
print(norm_outputs_tr.shape)
print(norm_outputs_val.shape)

