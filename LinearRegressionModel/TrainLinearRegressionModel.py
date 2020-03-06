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
from sklearn.metrics import classification_report

import torch as torch

torch.cuda.empty_cache()
plt.rcParams.update({'font.size': 14})

#----------------------------
# Set variables for this run
#----------------------------
run_vars={'dimension':3, 'lat':True , 'lon':True , 'dep':True , 'current':True , 'sal':True , 'eta':True , 'poly_degree':2}
model_type = 'lr'
#exp_prefix = 'BalancedTrainingSet_'
exp_prefix = ''

TrainModel=True 

DIR = '/data/hpcdata/users/racfur/MITGCM_OUTPUT/20000yr_Windx1.00_mm_diag/'
MITGCM_filename=DIR+'cat_tave_2000yrs_SelectedVars_masked.nc'

#---------------------------
# calculate other variables 
#---------------------------
data_name = cn.create_dataname(run_vars)

# RF BUG TESTING!
data_name = 'TEST_'+data_name

exp_name = exp_prefix+data_name

### Calculate x,y position in each feature list for extracting 'now' value
# This probably doesn't work with the Polyfeatures.... need to check and/or rethink!
#if run_vars['dimension'] == 2:
#   xy_pos = 4
#elif run_vars['dimension'] == 3:
#   xy_pos = 13
#else:
#   print('ERROR, dimension neither two or three!!!')

#--------------------------------------------------------------
# Open data from saved array
#--------------------------------------------------------------
print('reading data')
inputsoutputs_file = '/data/hpcdata/users/racfur/DynamicPrediction/INPUT_OUTPUT_ARRAYS/SinglePoint_'+data_name+'_InputsOutputs.npz'
norm_inputs_tr, norm_inputs_val, norm_inputs_te, norm_outputs_tr, norm_outputs_val, norm_outputs_te = np.load(inputsoutputs_file).values()

#-------------------------------------------------------------
# Set up a model in scikitlearn to predict deltaT (the trend)
# Run ridge regression tuning alpha through cross val
#-------------------------------------------------------------
pkl_filename = '/data/hpcdata/users/racfur/DynamicPrediction/lr_Outputs/MODELS/pickle_'+model_type+'_'+exp_name+'.pkl'
if TrainModel:
    print('training model')
    
    #alpha_s = [0.00, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10, 30]
    alpha_s = [0.00]
    parameters = [{'alpha': alpha_s}]
    n_folds=3
    
    lr = linear_model.Ridge(fit_intercept=True)
    
    # set up regressor
    lr = GridSearchCV(lr, param_grid=parameters, cv=n_folds, scoring='neg_mean_squared_error', refit=True)
    #lr = GridSearchCV(lr, param_grid=parameters, cv=n_folds, scoring='max_error', refit=True)
    
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

# predict values and calculate the error
print('predict values')
norm_lr_predicted_tr = lr.predict(norm_inputs_tr)
lr_tr_mse = metrics.mean_squared_error(norm_outputs_tr, norm_lr_predicted_tr)

norm_lr_predicted_val = lr.predict(norm_inputs_val)
lr_val_mse = metrics.mean_squared_error(norm_outputs_val, norm_lr_predicted_val)

# run a few simple tests to check against iterator:
in1 = np.zeros((1,norm_inputs_tr.shape[1]))
in2 = np.ones((1,norm_inputs_tr.shape[1]))

out1 = lr.predict(in1)
out2 = lr.predict(in2)

print(out1)
print(out2)

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
am.get_stats(model_type, exp_name, name1='Training', truth1=norm_outputs_tr, exp1=norm_lr_predicted_tr, pers1=predict_persistance_tr,
                                name2='Validation',  truth2=norm_outputs_val, exp2=norm_lr_predicted_val, pers2=predict_persistance_val, name='TrainVal_norm')
am.get_stats(model_type, exp_name, name1='Training', truth1=denorm_outputs_tr, exp1=denorm_lr_predicted_tr, pers1=predict_persistance_tr,
                                name2='Validation',  truth2=denorm_outputs_val, exp2=denorm_lr_predicted_val, pers2=predict_persistance_val, name='TrainVal_denorm')

print('plot results')
am.plot_results(model_type, exp_name, norm_outputs_tr, norm_lr_predicted_tr, name='training_norm')
am.plot_results(model_type, exp_name, norm_outputs_val, norm_lr_predicted_val, name='val_norm')
am.plot_results(model_type, exp_name, denorm_outputs_tr, denorm_lr_predicted_tr, name='training_denorm')
am.plot_results(model_type, exp_name, denorm_outputs_val, denorm_lr_predicted_val, name='val_denorm')

#-------------------------------------------------------------------
# plot histograms:
#-------------------------------------------------
fig = rfplt.Plot_Histogram(denorm_lr_predicted_tr, 100) 
plt.savefig('../../'+model_type+'_Outputs/PLOTS/'+model_type+'_'+exp_name+'_denorm_prediction_train_histogram', bbox_inches = 'tight', pad_inches = 0.1)

fig = rfplt.Plot_Histogram(denorm_lr_predicted_val, 100)
plt.savefig('../../'+model_type+'_Outputs/PLOTS/'+model_type+'_'+exp_name+'_denorm_prediction_val_histogram', bbox_inches = 'tight', pad_inches = 0.1)

fig = rfplt.Plot_Histogram(denorm_lr_predicted_tr-denorm_outputs_tr, 100)
plt.savefig('../../'+model_type+'_Outputs/PLOTS/'+model_type+'_'+exp_name+'_denorm_errors_train_histogram', bbox_inches = 'tight', pad_inches = 0.1)

fig = rfplt.Plot_Histogram(denorm_lr_predicted_val-denorm_outputs_val, 100) 
plt.savefig('../../'+model_type+'_Outputs/PLOTS/'+model_type+'_'+exp_name+'_denorm_errors_val_histogram', bbox_inches = 'tight', pad_inches = 0.1)


#---------------------------------------------------------
# Plot scatter plots of errors against outputs and inputs
#---------------------------------------------------------
am.plot_results(model_type, exp_name, denorm_outputs_tr, denorm_lr_predicted_tr-denorm_outputs_tr, name='training_denorm', xlabel='DeltaT', ylabel='Errors')
am.plot_results(model_type, exp_name, denorm_outputs_val, denorm_lr_predicted_val-denorm_outputs_val, name='val_denorm', xlabel='DeltaT', ylabel='Errors')

#fig = plt.figure(figsize=(9,9))
#ax1 = fig.add_subplot(111)
#ax1.scatter(denorm_inputs_tr[:,xy_pos], denorm_lr_predicted_tr-denorm_outputs_tr, edgecolors=(0, 0, 0))
#ax1.set_xlabel('InputTemp')
#ax1.set_ylabel('Errors')
#plt.savefig('../../'+model_type+'_Outputs/PLOTS/'+model_type+'_'+exp_name+'_inputtempVerrors_training_denorm.png', bbox_inches = 'tight', pad_inches = 0.1)
#
#fig = plt.figure(figsize=(9,9))
#ax1 = fig.add_subplot(111)
#ax1.scatter(denorm_inputs_val[:,xy_pos], denorm_lr_predicted_val-denorm_outputs_val, edgecolors=(0, 0, 0))
#ax1.set_xlabel('InputTemp')
#ax1.set_ylabel('Errors')
#plt.savefig('../../'+model_type+'_Outputs/PLOTS/'+model_type+'_'+exp_name+'_inputtempVerrors_val_denorm.png', bbox_inches = 'tight', pad_inches = 0.1)

