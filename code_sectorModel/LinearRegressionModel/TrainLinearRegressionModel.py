#!/usr/bin/env python
# coding: utf-8
# Script written by Rachel Furner
# Creates a linear regressor to forecast temperature change
# based on data from from an MITgcm sector model.
# Assesment is then done over the training and validation datasets

#----------------------------
# Import neccessary packages
#----------------------------
import sys
sys.path.append('../Tools')
import CreateDataName as cn
import AssessModel as am
import Model_Plotting as rfplt
import ReadRoutines as rr

import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn import linear_model
from sklearn.model_selection import GridSearchCV

import gc

import os

plt.rcParams.update({'font.size': 10})
plt.rc('font', family='sans serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')

#----------------------------
# Set variables for this run
#----------------------------
run_vars = {'dimension':3, 'lat':True , 'lon':True , 'dep':True , 'current':True , 'bolus_vel':True , 'sal':True , 'eta':True , 'density':True , 'poly_degree':1}
data_prefix = ''
model_prefix = 'alpha.001_'

TrainModel = True  

DIR = '/data/hpcdata/users/racfur/MITgcm/verification/MundaySectorConfig_2degree/runs/100yrs/'
MITGCM_filename=DIR+'mnc_test_0002/cat_tave.nc'
density_file = DIR+'DensityData.npy'

print('run_vars; '+str(run_vars))
print('model_prefix ; '+str(model_prefix))

#------------------------------------------------------------
# calculate other variables - these dont change between runs
#------------------------------------------------------------
data_name = data_prefix + cn.create_dataname(run_vars)

model_name = model_prefix+data_name

plot_dir = '../../../lr_Outputs/PLOTS/'+model_name
if not os.path.isdir(plot_dir):
   os.system("mkdir %s" % (plot_dir))

#--------------------------------------------------------------
# Read in data
#--------------------------------------------------------------
print('reading in data')
norm_inputs_tr, norm_inputs_val, norm_inputs_te, norm_outputs_tr, norm_outputs_val, norm_outputs_te = rr.ReadMITGCM(MITGCM_filename, density_file, 0.7, 0.9, data_name, run_vars, plot_histograms=True)
del norm_inputs_te
del norm_outputs_te

print(norm_inputs_tr.shape)
print(norm_inputs_val.shape)
print(norm_outputs_tr.shape)
print(norm_outputs_val.shape)

#-------------------------------------------------------------
# Set up a model in scikitlearn to predict deltaT (the trend)
# Run ridge regression tuning alpha through cross val
#-------------------------------------------------------------
print('setting up model')
pkl_filename = '../../../lr_Outputs/MODELS/'+model_name+'_pickle.pkl'
if TrainModel:
    print('training model')
    
    alpha_s = [0.001]
    parameters = [{'alpha': alpha_s}]
    n_folds=3
   
    lr = linear_model.Ridge(fit_intercept=False)
    
    lr = GridSearchCV(lr, param_grid=parameters, cv=n_folds, scoring='neg_mean_squared_error', refit=True)
    
    # fit the model
    lr.fit(norm_inputs_tr, norm_outputs_tr)
    lr.get_params()

    # Write info on Alpha in file    
    info_filename = '../../../lr_Outputs/MODELS/'+model_name+'_info.txt'
    info_file=open(info_filename,"w")
    info_file.write("Best parameters set found on development set:\n")
    info_file.write('\n')
    info_file.write(str(lr.best_params_)+'\n')
    info_file.write('\n')
    info_file.write("Grid scores on development set:\n")
    info_file.write('\n')
    means = lr.cv_results_['mean_test_score']
    stds = lr.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, lr.cv_results_['params']):
        info_file.write("%0.3f (+/-%0.03f) for %r \n" % (mean, std * 2, params))
    info_file.write('')

    # Store coeffs in an npz file
    coef_filename = '../../../lr_Outputs/MODELS/'+model_name+'_coefs.npz'
    np.savez( coef_filename, np.asarray(lr.best_estimator_.intercept_), np.asarray(lr.best_estimator_.coef_) )

    # pickle it
    with open(pkl_filename, 'wb') as pckl_file:
        pickle.dump(lr, pckl_file)

#--------------------------------------------------
# Make predictions and denormalise ready to assess
#--------------------------------------------------

with open(pkl_filename, 'rb') as file:
    print('opening '+pkl_filename)
    lr = pickle.load(file)

# predict values
print('predict values')
norm_lr_predicted_tr = lr.predict(norm_inputs_tr).reshape(-1,1)
norm_lr_predicted_val = lr.predict(norm_inputs_val).reshape(-1,1)

del norm_inputs_tr
del norm_inputs_val
gc.collect()

# De-normalise the outputs and predictions

# Read in mean and std
mean_std_file = '../../../INPUT_OUTPUT_ARRAYS/SinglePoint_'+data_name+'_MeanStd.npz'
zip_mean_std_file = mean_std_file+'.gz' 
if os.path.isfile(mean_std_file):
   mean_std_data = np.load(mean_std_file)
elif os.path.isfile(zip_mean_std_file):
   os.system("gunzip %s" % (zip_mean_std_file))
   mean_std_data = np.load(mean_std_file)
   os.system("gunzip %s" % (mean_std_file))
input_mean  = mean_std_data['arr_0']
input_std   = mean_std_data['arr_1']
output_mean = mean_std_data['arr_2']
output_std  = mean_std_data['arr_3']

# define denormalising function
def denormalise_data(norm_data,mean,std):
    denorm_data = norm_data * std + mean
    return denorm_data

# denormalise the predictions and true outputs   
denorm_lr_predicted_tr = denormalise_data(norm_lr_predicted_tr, output_mean, output_std)
denorm_lr_predicted_val = denormalise_data(norm_lr_predicted_val, output_mean, output_std)
denorm_outputs_tr = denormalise_data(norm_outputs_tr, output_mean, output_std)
denorm_outputs_val = denormalise_data(norm_outputs_val, output_mean, output_std)

del norm_lr_predicted_tr
del norm_lr_predicted_val
del norm_outputs_tr
del norm_outputs_val
gc.collect()

#------------------
# Assess the model
#------------------
# Calculate 'persistance' score - persistence prediction is just zero everywhere as we're predicting the trend
predict_persistance_tr = np.zeros(denorm_outputs_tr.shape)
predict_persistance_val = np.zeros(denorm_outputs_val.shape)

print('get stats')
am.get_stats(model_name, 
             name1='Training', truth1=denorm_outputs_tr, exp1=denorm_lr_predicted_tr, pers1=predict_persistance_tr,
             name2='Validation', truth2=denorm_outputs_val, exp2=denorm_lr_predicted_val, pers2=predict_persistance_val,
             name='TrainVal')

print('plot results')
top    = max(max(denorm_outputs_tr), max(denorm_lr_predicted_tr), max(denorm_outputs_val), max(denorm_lr_predicted_val))
top    = top + 0.1*abs(top)
bottom = min(min(denorm_outputs_tr), min(denorm_lr_predicted_tr), min(denorm_outputs_val), min(denorm_lr_predicted_val))
bottom = bottom - 0.1*abs(top)
am.plot_scatter(model_name, denorm_outputs_tr, denorm_lr_predicted_tr, name='train', top=top, bottom=bottom, text='(a)')
am.plot_scatter(model_name, denorm_outputs_val, denorm_lr_predicted_val, name='val', top=top, bottom=bottom, text='(b)')

##------------------
## plot histograms:
##------------------
#fig = rfplt.Plot_Histogram(denorm_lr_predicted_tr, 100) 
#plt.savefig(plot_dir+'/'+model_name+'_histogram_train_predictions.png', bbox_inches = 'tight', pad_inches = 0.1)
#
#fig = rfplt.Plot_Histogram(denorm_lr_predicted_val, 100)
#plt.savefig(plot_dir+'/'+model_name+'_histogram_val_predictions.png', bbox_inches = 'tight', pad_inches = 0.1)
#
#fig = rfplt.Plot_Histogram(denorm_lr_predicted_tr-denorm_outputs_tr, 100)
#plt.savefig(plot_dir+'/'+model_name+'_histogram_train_errors.png', bbox_inches = 'tight', pad_inches = 0.1)
#
#fig = rfplt.Plot_Histogram(denorm_lr_predicted_val-denorm_outputs_val, 100) 
#plt.savefig(plot_dir+'/'+model_name+'_histogram_val_errors.png', bbox_inches = 'tight', pad_inches = 0.1)
#
##----------------------------------------------
## Plot scatter plots of errors against outputs
##----------------------------------------------
#am.plot_scatter(model_name, denorm_outputs_tr, denorm_lr_predicted_tr-denorm_outputs_tr, name='train', xlabel='DeltaT', ylabel='Errors', exp_cor=False)
#am.plot_scatter(model_name, denorm_outputs_val, denorm_lr_predicted_val-denorm_outputs_val, name='val', xlabel='DeltaT', ylabel='Errors', exp_cor=False)
#

