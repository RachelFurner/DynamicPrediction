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
from Tools import ReadRoutines as rr

import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn import linear_model
import sklearn.metrics as metrics
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

import torch as torch
import gc

import os

torch.cuda.empty_cache()
plt.rcParams.update({'font.size': 14})

#----------------------------
# Set variables for this run
#----------------------------
run_vars = {'dimension':3, 'lat':True , 'lon':True , 'dep':True , 'current':False, 'bolus_vel':False, 'sal':False, 'eta':False, 'density':False, 'poly_degree':2}
model_type = 'lr'
time_step = '24hrs'
data_prefix = ''
model_prefix = ''

TrainModel = True  

if time_step == '1mnth':
   DIR = '/data/hpcdata/users/racfur/MITGCM_OUTPUT/20000yr_Windx1.00_mm_diag/'
   MITGCM_filename=DIR+'cat_tave_2000yrs_SelectedVars_masked_withBolus.nc'
   density_file = DIR+'DensityData.npy'
elif time_step == '24hrs':
   DIR = '/data/hpcdata/users/racfur/MITGCM_OUTPUT/100yr_Windx1.00_FrequentOutput/'
   MITGCM_filename=DIR+'cat_tave_50yr_SelectedVars_masked_withBolus.nc'
   density_file = DIR+'DensityData.npy'
else:
   print('ERROR - No proper time step given')

#---------------------------
# calculate other variables 
#---------------------------
data_name = cn.create_dataname(run_vars)
data_name = data_prefix+data_name+'_'+time_step

model_name = model_prefix+data_name

plot_dir = '../../'+model_type+'_Outputs/PLOTS/'+model_name
if not os.path.isdir(plot_dir):
   os.system("mkdir %s" % (plot_dir))

#--------------------------------------------------------------
# Read in data or load from saved array
#--------------------------------------------------------------
print('reading data')
inputs_tr_file = '/data/hpcdata/users/racfur/DynamicPrediction/INPUT_OUTPUT_ARRAYS/SinglePoint_'+data_name+'_InputsTr.npy'
zip_inputs_tr_file = inputs_tr_file+'.gz'
inputs_val_file = '/data/hpcdata/users/racfur/DynamicPrediction/INPUT_OUTPUT_ARRAYS/SinglePoint_'+data_name+'_InputsVal.npy'
zip_inputs_val_file = inputs_val_file+'.gz'
outputs_tr_file = '/data/hpcdata/users/racfur/DynamicPrediction/INPUT_OUTPUT_ARRAYS/SinglePoint_'+time_step+'_OutputsTr.npy'
zip_outputs_tr_file = outputs_tr_file+'.gz'
outputs_val_file = '/data/hpcdata/users/racfur/DynamicPrediction/INPUT_OUTPUT_ARRAYS/SinglePoint_'+time_step+'_OutputsVal.npy'
zip_outputs_val_file = outputs_val_file+'.gz'

# If datasets already exist (zipped or unzipped) load these, otherwise create them but don't save - disc space too short!!
if not ( (os.path.isfile(inputs_tr_file)   or os.path.isfile(zip_inputs_tr_file))   and
         (os.path.isfile(inputs_val_file)  or os.path.isfile(zip_inputs_val_file))  and
         (os.path.isfile(outputs_tr_file)  or os.path.isfile(zip_outputs_tr_file))  and
         (os.path.isfile(outputs_val_file) or os.path.isfile(zip_outputs_val_file)) ):
   norm_inputs_tr, norm_inputs_val, norm_inputs_te, norm_outputs_tr, norm_outputs_val, norm_outputs_te = rr.ReadMITGCM(MITGCM_filename, density_file, 0.7, 0.9, data_name, run_vars, time_step=time_step, save_arrays=False, plot_histograms=False)
   del norm_inputs_te
   del norm_outputs_te
else:
   
   if os.path.isfile(inputs_tr_file):
      norm_inputs_tr = np.load(inputs_tr_file, mmap_mode='r')
   elif os.path.isfile(zip_inputs_tr_file):
      os.system("gunzip %s" % (zip_inputs_tr_file))
      norm_inputs_tr = np.load(inputs_tr_file, mmap_mode='r')
      os.system("gzip %s" % (inputs_tr_file))

   if os.path.isfile(inputs_val_file):
      norm_inputs_val = np.load(inputs_val_file, mmap_mode='r')
   elif os.path.isfile(zip_inputs_val_file):
      os.system("gunzip %s" % (zip_inputs_val_file))
      norm_inputs_val = np.load(inputs_val_file, mmap_mode='r')
      os.system("gzip %s" % (inputs_val_file))

   if os.path.isfile(outputs_tr_file):
      norm_outputs_tr = np.load(outputs_tr_file, mmap_mode='r')
   elif os.path.isfile(zip_outputs_tr_file):
      os.system("gunzip %s" % (zip_outputs_tr_file))
      norm_outputs_tr = np.load(outputs_tr_file, mmap_mode='r')
      os.system("gzip %s" % (outputs_tr_file))
   
   if os.path.isfile(outputs_val_file):
      norm_outputs_val = np.load(outputs_val_file, mmap_mode='r')
   elif os.path.isfile(zip_outputs_val_file):
      os.system("gunzip %s" % (zip_outputs_val_file))
      norm_outputs_val = np.load(outputs_val_file, mmap_mode='r')
      os.system("gzip %s" % (outputs_val_file))

#norm_inputs_tr   = norm_inputs_tr[:10]
#norm_inputs_val  = norm_inputs_val[:10]
#norm_outputs_tr  = norm_outputs_tr[:10]
#norm_outputs_val = norm_outputs_val[:10]

print(norm_inputs_tr.shape)
print(norm_inputs_val.shape)
print(norm_outputs_tr.shape)
print(norm_outputs_val.shape)
#-------------------------------------------------------------
# Set up a model in scikitlearn to predict deltaT (the trend)
# Run ridge regression tuning alpha through cross val
#-------------------------------------------------------------
pkl_filename = '/data/hpcdata/users/racfur/DynamicPrediction/lr_Outputs/MODELS/'+model_name+'_pickle.pkl'
if TrainModel:
    print('training model')
    
    alpha_s = [0.0001, 0.001, 0.01, 0.1, 1.0]
    alpha_s = [0.0001]
    parameters = [{'alpha': alpha_s}]
    n_folds=3
   
    lr = linear_model.Ridge(fit_intercept=False)
    #lr = linear_model.Lasso(fit_intercept=False, max_iter=2000000)
    
    lr = GridSearchCV(lr, param_grid=parameters, cv=n_folds, scoring='neg_mean_squared_error', refit=True)
    
    # fit the model
    lr.fit(norm_inputs_tr, norm_outputs_tr)
    lr.get_params()

    # Write info on Alpha in file    
    info_filename = '../../'+model_type+'_Outputs/MODELS/'+model_name+'_info.txt'
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

    # Store coeffs in an npz file and print to info file
    coef_filename = '../../'+model_type+'_Outputs/MODELS/'+model_name+'_coefs.npz'
    np.savez( coef_filename, np.asarray(lr.best_estimator_.intercept_), np.asarray(lr.best_estimator_.coef_) )
    info_file.write("lr.best_estimator_.intercept_: \n")
    info_file.write(str(lr.best_estimator_.intercept_))
    info_file.write('\n')
    info_file.write("lr.best_estimator_.coef_: \n")
    info_file.write(str(lr.best_estimator_.coef_))
    info_file.write('\n')
    info_file.write('\n')

    # pickle it
    with open(pkl_filename, 'wb') as pckl_file:
        pickle.dump(lr, pckl_file)

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

#------------------------------------------
# De-normalise the outputs and predictions
#------------------------------------------
mean_std_file = '/data/hpcdata/users/racfur/DynamicPrediction/INPUT_OUTPUT_ARRAYS/SinglePoint_'+data_name+'_MeanStd.npz'
zip_mean_std_file = mean_std_file+'.gz' 
if os.path.isfile(mean_std_file):
   input_mean, input_std, output_mean, output_std = np.load(mean_std_file).values()
elif os.path.isfile(zip_mean_std_file):
   os.system("gunzip %s" % (zip_mean_std_file))
   input_mean, input_std, output_mean, output_std = np.load(mean_std_file).values()
   os.system("gunzip %s" % (mean_std_file))

# denormalise data
def denormalise_data(norm_data,mean,std):
    denorm_data = norm_data * std + mean
    return denorm_data

# denormalise the predictions and true outputs   
denorm_lr_predicted_tr = denormalise_data(norm_lr_predicted_tr, output_mean, output_std)
denorm_lr_predicted_val = denormalise_data(norm_lr_predicted_val, output_mean, output_std)
denorm_outputs_tr = denormalise_data(norm_outputs_tr, output_mean, output_std)
denorm_outputs_val = denormalise_data(norm_outputs_val, output_mean, output_std)

# Free up memory
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
am.get_stats(model_type, model_name, name1='Training', truth1=denorm_outputs_tr, exp1=denorm_lr_predicted_tr, pers1=predict_persistance_tr,
                                name2='Validation',  truth2=denorm_outputs_val, exp2=denorm_lr_predicted_val, pers2=predict_persistance_val, name='TrainVal')

print('plot results')
am.plot_results(model_type, model_name, denorm_outputs_tr, denorm_lr_predicted_tr, name='training')
am.plot_results(model_type, model_name, denorm_outputs_val, denorm_lr_predicted_val, name='val')

#-------------------------------------------------------------------
# plot histograms:
#-------------------------------------------------
fig = rfplt.Plot_Histogram(denorm_lr_predicted_tr, 100) 
plt.savefig(plot_dir+'/'+model_name+'_histogram_train_predictions', bbox_inches = 'tight', pad_inches = 0.1)

fig = rfplt.Plot_Histogram(denorm_lr_predicted_val, 100)
plt.savefig(plot_dir+'/'+model_name+'_histogram_val_predictions', bbox_inches = 'tight', pad_inches = 0.1)

fig = rfplt.Plot_Histogram(denorm_lr_predicted_tr-denorm_outputs_tr, 100)
plt.savefig(plot_dir+'/'+model_name+'_histogram_train_errors', bbox_inches = 'tight', pad_inches = 0.1)

fig = rfplt.Plot_Histogram(denorm_lr_predicted_val-denorm_outputs_val, 100) 
plt.savefig(plot_dir+'/'+model_name+'_histogram_val_errors', bbox_inches = 'tight', pad_inches = 0.1)

#----------------------------------------------
# Plot scatter plots of errors against outputs
#----------------------------------------------
am.plot_results(model_type, model_name, denorm_outputs_tr, denorm_lr_predicted_tr-denorm_outputs_tr, name='training', xlabel='DeltaT', ylabel='Errors', exp_cor=False)
am.plot_results(model_type, model_name, denorm_outputs_val, denorm_lr_predicted_val-denorm_outputs_val, name='val', xlabel='DeltaT', ylabel='Errors', exp_cor=False)


