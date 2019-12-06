#!/usr/bin/env python
# coding: utf-8

#----------------------------
# Import neccessary packages
#----------------------------

from sklearn import linear_model
import numpy as np
import os
import xarray as xr
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report
import sklearn.metrics as metrics
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import pickle


#---------------
# Set variables
#---------------

StepSize = 1 # how many output steps (months!) to predict over

halo_size = 1
halo_list = (range(-halo_size, halo_size+1))
##Calculate x,y position in each feature list to use as 'now' value
if halo_size == 1:
   xy = 13
elif halo_size == 2:
   xy == 62
else:
   print('error in halo size....')
#xy=2*(halo_size^2+halo_size) # if using a 2-d halo in x and y....if using a 3-d halo need to use above.

# region at start of run to learn from
tr_split_index_yr = 200                    # how many years to learn from
tr_split_index    = 12*tr_split_index_yr   # convert to from years to months

te_split_index_yr = 2000                   # Define first set of test data from period when model is still dynamically active
te_split_index    = 12*te_split_index_yr  

temp = 'Ttave'
#my_var = 'uVeltave'
#my_var = 'Stave'
#my_var = 'vVeltave'

#exp_name='T_int3dLatPastPlusCurrent_tr'+str(tr_split_index_yr)+'_halo'+str(halo_size)+'_pred'+str(StepSize)
exp_name='T_Int3dLatVarsDepth'+str(tr_split_index_yr)+'_halo'+str(halo_size)+'_pred'+str(StepSize)


#------------------
# Read in the data
#------------------

DIR = '/data/hpcdata/users/racfur/MITGCM_OUTPUT/'
exp = '20000yr_Windx1.00_mm_diag/'
filename=DIR+exp+'cat_tave_5000yrs_SelectedVars.nc'
print(filename)
ds   = xr.open_dataset(filename)
#print(ds)

da_T=ds['Ttave'].values
da_S=ds['Stave'].values
da_U=ds['uVeltave'].values
da_V=ds['vVeltave'].values
da_Eta=ds['ETAtave'].values
da_lat=ds['Y'].values
da_depth=ds['Z'].values

##------------------------------------------------------------------------------------------
## Read in data as ione dataset, which we randomly split into train and test later
##------------------------------------------------------------------------------------------
#inputs = []
#outputs = []
#print('read in data')
#
#for z in range(2,38,8):
#    for x in range(2,7,3):
#        for y in range(2,74,10):
#            for time in range(0, te_split_index, 10):
#                input_temp = []
#                [input_temp.append(da_T[time,z+z_offset,y+y_offset,x+x_offset]) for x_offset in halo_list for y_offset in halo_list for z_offset in halo_list]
#                [input_temp.append(da_S[time,z+z_offset,y+y_offset,x+x_offset]) for x_offset in halo_list for y_offset in halo_list for z_offset in halo_list]
#                [input_temp.append(da_U[time,z+z_offset,y+y_offset,x+x_offset]) for x_offset in halo_list for y_offset in halo_list for z_offset in halo_list]
#                [input_temp.append(da_V[time,z+z_offset,y+y_offset,x+x_offset]) for x_offset in halo_list for y_offset in halo_list for z_offset in halo_list]
#                [input_temp.append(da_Eta[time,y+y_offset,x+x_offset]) for x_offset in halo_list for y_offset in halo_list]
#                input_temp.append(da_lat[y])
#                input_temp.append(da_depth[z])
#                inputs.append(input_temp)
#                outputs.append([da_T[time+StepSize,z,y,x]])
#               
#print('now save them')
#inputs=np.asarray(inputs)
#outputs=np.asarray(outputs)
#np.savez('inputs_outputs_arrays.npz', inputs,outputs)
#print('saved')
#
##------------------------------------------------------------------------------------------
## Read in data as training and test data - test data is split over two validation periods;
## 1 is while the model is still dynamically active, 2 is when it is near/at equilibrium
##------------------------------------------------------------------------------------------
# Read in as training and test data (rather than reading in all data and splitting),
# so we can learn on first n time steps, and test on rest
# substep in space and time

inputs_tr = []
outputs_tr = []

inputs_te1 = []
outputs_te1 = []

inputs_te2 = []
outputs_te2 = []

for z in range(2,38,8):
    for x in range(2,7,3):
        for y in range(2,74,10):
            for time in range(0, tr_split_index, 10):
                input_temp = []
                [input_temp.append(da_T[time,z+z_offset,y+y_offset,x+x_offset]) for x_offset in halo_list for y_offset in halo_list for z_offset in halo_list]
                [input_temp.append(da_S[time,z+z_offset,y+y_offset,x+x_offset]) for x_offset in halo_list for y_offset in halo_list for z_offset in halo_list]
                [input_temp.append(da_U[time,z+z_offset,y+y_offset,x+x_offset]) for x_offset in halo_list for y_offset in halo_list for z_offset in halo_list]
                [input_temp.append(da_V[time,z+z_offset,y+y_offset,x+x_offset]) for x_offset in halo_list for y_offset in halo_list for z_offset in halo_list]
                [input_temp.append(da_Eta[time,y+y_offset,x+x_offset]) for x_offset in halo_list for y_offset in halo_list]
                #[input_temp.append(da_T[time-StepSize,z+z_offset,y+y_offset,x+x_offset]) for x_offset in halo_list for y_offset in halo_list for z_offset in halo_list]
                input_temp.append(da_lat[y])
                input_temp.append(da_depth[z])
                inputs_tr.append(input_temp)
                #inputs_tr.append([da_T[time,z+z_offset,y+y_offset,x+x_offset] for x_offset in halo_list for y_offset in halo_list for z_offset in halo_list])
                #inputs_tr.append([da_T[time,z,y+y_offset,x+x_offset] for x_offset in halo_list for y_offset in halo_list])
                outputs_tr.append([da_T[time+StepSize,z,y,x]])
                
            for time in range(tr_split_index, te_split_index, 50):
                input_temp = []
                [input_temp.append(da_T[time,z+z_offset,y+y_offset,x+x_offset]) for x_offset in halo_list for y_offset in halo_list for z_offset in halo_list]
                [input_temp.append(da_S[time,z+z_offset,y+y_offset,x+x_offset]) for x_offset in halo_list for y_offset in halo_list for z_offset in halo_list]
                [input_temp.append(da_U[time,z+z_offset,y+y_offset,x+x_offset]) for x_offset in halo_list for y_offset in halo_list for z_offset in halo_list]
                [input_temp.append(da_V[time,z+z_offset,y+y_offset,x+x_offset]) for x_offset in halo_list for y_offset in halo_list for z_offset in halo_list]
                [input_temp.append(da_Eta[time,y+y_offset,x+x_offset]) for x_offset in halo_list for y_offset in halo_list]
                #[input_temp.append(da_T[time-StepSize,z+z_offset,y+y_offset,x+x_offset]) for x_offset in halo_list for y_offset in halo_list for z_offset in halo_list]
                input_temp.append(da_lat[y])
                input_temp.append(da_depth[z])
                inputs_te1.append(input_temp)
                #inputs_te1.append([da_T[time,z+z_offset,y+y_offset,x+x_offset] for x_offset in halo_list for y_offset in halo_list for z_offset in halo_list])
                #inputs_te1.append([da_T[time,z,y+y_offset,x+x_offset] for x_offset in halo_list for y_offset in halo_list])
                outputs_te1.append([da_T[time+StepSize,z,y,x]])

            for time in range(te_split_index, len(ds.T.data)-StepSize, 100):
                input_temp = []
                [input_temp.append(da_T[time,z+z_offset,y+y_offset,x+x_offset]) for x_offset in halo_list for y_offset in halo_list for z_offset in halo_list]
                [input_temp.append(da_S[time,z+z_offset,y+y_offset,x+x_offset]) for x_offset in halo_list for y_offset in halo_list for z_offset in halo_list]
                [input_temp.append(da_U[time,z+z_offset,y+y_offset,x+x_offset]) for x_offset in halo_list for y_offset in halo_list for z_offset in halo_list]
                [input_temp.append(da_V[time,z+z_offset,y+y_offset,x+x_offset]) for x_offset in halo_list for y_offset in halo_list for z_offset in halo_list]
                [input_temp.append(da_Eta[time,y+y_offset,x+x_offset]) for x_offset in halo_list for y_offset in halo_list]
                #[input_temp.append(da_T[time-StepSize,z+z_offset,y+y_offset,x+x_offset]) for x_offset in halo_list for y_offset in halo_list for z_offset in halo_list]
                input_temp.append(da_lat[y])
                input_temp.append(da_depth[z])
                inputs_te2.append(input_temp)
                #inputs_te2.append([da_T[time,z+z_offset,y+y_offset,x+x_offset] for x_offset in halo_list for y_offset in halo_list for z_offset in halo_list])
                #inputs_te2.append([da_T[time,z,y+y_offset,x+x_offset] for x_offset in halo_list for y_offset in halo_list])
                outputs_te2.append([da_T[time+StepSize,z,y,x]])
                
inputs_tr=np.asarray(inputs_tr)
outputs_tr=np.asarray(outputs_tr)

inputs_te1=np.asarray(inputs_te1)
outputs_te1=np.asarray(outputs_te1)

inputs_te2=np.asarray(inputs_te2)
outputs_te2=np.asarray(outputs_te2)

#Save these to a file
np.savez('test_train1_train2_arrays.npz', inputs_tr,outputs_tr,inputs_te1,outputs_te1, inputs_te2, outputs_te2)

#Load the data from file instead - havign saved it on previous run of the above code
#inputs, outputs = np.load('inputs_outputs_arrays.npz').values()

#print('shape for inputs and outputs')
#print(inputs.shape, outputs.shape)
print('shape for inputs and outputs: Tr; test1; test2')
print(inputs_tr.shape, outputs_tr.shape)
print(inputs_te1.shape, outputs_te1.shape)
print(inputs_te2.shape, outputs_te2.shape)

#----------------------------------------------
# Normalise Data (based on training data only)
#----------------------------------------------
print('normalise data')
#def normalise_data(train):
#    train_mean, train_std = np.mean(train), np.std(train)
#    norm_train = (train - train_mean) / train_std
#    return norm_train

## normalise inputs
#norm_inputs  = np.zeros(inputs.shape)
#for i in range(inputs.shape[1]):  #loop over each feature, normalising individually
#    norm_inputs[:, i] = normalise_data(inputs[:, i])

##normalise outputs
#norm_outputs = normalise_data(outputs[:])

##Calc mean and std of outputs re-forming predictions
#outputs_mean = np.mean(outputs)
#outputs_std = np.std(outputs)

def normalise_data(train,test1,test2):
    train_mean, train_std = np.mean(train), np.std(train)
    norm_train = (train - train_mean) / train_std
    norm_test1 = (test1 - train_mean) / train_std
    norm_test2 = (test2 - train_mean) / train_std
    return norm_train, norm_test1, norm_test2

## normalise inputs
norm_inputs_tr  = np.zeros(inputs_tr.shape)
norm_inputs_te1 = np.zeros(inputs_te1.shape)
norm_inputs_te2 = np.zeros(inputs_te2.shape)
for i in range(inputs_tr.shape[1]):  #loop over each feature, normalising individually
    norm_inputs_tr[:, i], norm_inputs_te1[:, i], norm_inputs_te2[:, i] = normalise_data(inputs_tr[:, i], inputs_te1[:, i], inputs_te2[:, i])

##normalise outputs
norm_outputs_tr, norm_outputs_te1, norm_outputs_te2 = normalise_data(outputs_tr[:], outputs_te1[:], outputs_te2[:])

## Calc mean and std of outputs re-forming predictions
outputs_tr_mean = np.mean(outputs_tr)
outputs_tr_std = np.std(outputs_tr)

#---------------------------------------------------------------------
# If needed, randomise the sample order, and split into test and train data
#---------------------------------------------------------------------
#split=int( 0.8 * inputs.shape[0])
#
#np.random.seed(0)
#ordering = np.random.permutation(inputs.shape[0])
#
#norm_inputs_tr = norm_inputs[ordering][:split]
#norm_outputs_tr = norm_outputs[ordering][:split]
#norm_inputs_te = norm_inputs[ordering][split:]
#norm_outputs_te = norm_outputs[ordering][split:]

#---------------------------------------------------------------------
#---------------------------------------------------------------------
# Set up a model to directly predict variable value at next time step
#---------------------------------------------------------------------
#---------------------------------------------------------------------

#---------------------------------------------------------
# First calculate 'persistance' score, to give a baseline
#---------------------------------------------------------
print('calc persistance')
## For single combined dataset
## For training data
#predict_persistance_train = norm_inputs_tr[:,xy]
#pers_train_mse = metrics.mean_squared_error(norm_outputs_tr, predict_persistance_train)
## For Testi data
#predict_persistance_te = norm_inputs_te[:,xy]
#pers_te_mse = metrics.mean_squared_error(norm_outputs_te1, predict_persistance_te)


## For training data
predict_persistance_train = norm_inputs_tr[:,xy]
pers_train_mse = metrics.mean_squared_error(norm_outputs_tr, predict_persistance_train)

## For validation period 1
predict_persistance_te1 = norm_inputs_te1[:,xy]
pers_te1_mse = metrics.mean_squared_error(norm_outputs_te1, predict_persistance_te1)

## For validation period 2
predict_persistance_te2 = norm_inputs_te2[:,xy]
pers_te2_mse = metrics.mean_squared_error(norm_outputs_te2, predict_persistance_te2)


#------------------------------------------------------------------------------------------------------------------------------------------------
# Run ridge regression with interaction between terms, tuning alpha through cross val
#------------------------------------------------------------------------------------------------------------------------------------------------

alpha_s = [0.00, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10, 30]
parameters = {'alpha': alpha_s}
n_folds=3

lr = linear_model.Ridge()

print('set up regressor')
lr_cv = GridSearchCV(lr, param_grid=parameters, cv=n_folds, scoring='neg_mean_squared_error', refit=True)

print('do poly features')
polynomial_features = PolynomialFeatures(degree=2, interaction_only=True, include_bias=True)

lrpipe = Pipeline([("polynomial_features", polynomial_features), ("linear_regression", lr_cv)])
#lrpipe = lr_cv # don't include polynomial features

print('fit the model')
lrpipe.fit(norm_inputs_tr, norm_outputs_tr )  # train to evaluate the value at the next time step...
lrpipe.get_params()

print('pickle it')
pkl_filename = 'PICKLED_MODELS/pickle_lr_'+exp_name+'.pkl'
with open(pkl_filename, 'wb') as file:
    pickle.dump(lrpipe, file)

print('predict values and calc error')
#lr_predicted_train = lrpipe.predict(norm_inputs_tr)
#lrpipe_train_mse = metrics.mean_squared_error(norm_outputs_tr, lr_predicted_train)
#
#lr_predicted_te = lrpipe.predict(norm_inputs_te)
#lrpipe_te_mse = metrics.mean_squared_error(norm_outputs_te, lr_predicted_te)

lr_predicted_train = lrpipe.predict(norm_inputs_tr)
lrpipe_train_mse = metrics.mean_squared_error(norm_outputs_tr, lr_predicted_train)

lr_predicted_te1 = lrpipe.predict(norm_inputs_te1)
lrpipe_te1_mse = metrics.mean_squared_error(norm_outputs_te1, lr_predicted_te1)

lr_predicted_te2 = lrpipe.predict(norm_inputs_te2)
lrpipe_te2_mse = metrics.mean_squared_error(norm_outputs_te2, lr_predicted_te2)


##------------------------------------------
## Plot normalised prediction against truth
##------------------------------------------
#fig = plt.figure(figsize=(20,9.4))
#ax1 = fig.add_subplot(121)
#ax1.scatter(norm_outputs_te1, lr_predicted_1, edgecolors=(0, 0, 0))
#ax1.plot([norm_outputs_te1.min(), norm_outputs_te1.max()], [norm_outputs_te1.min(), norm_outputs_te1.max()], 'k--', lw=1)
#ax1.set_xlabel('Truth')
#ax1.set_ylabel('Predicted')
#ax1.set_title('Validation period 1')
#ax1.text(.05,.9,'Mean Squared Error: {:.4e}'.format(lrpipe1_mse),transform=ax1.transAxes)
#
#ax2 = fig.add_subplot(122)
#ax2.scatter(norm_outputs_te2, lr_predicted_te2, edgecolors=(0, 0, 0))
#ax2.plot([norm_outputs_te2.min(), norm_outputs_te2.max()], [norm_outputs_te2.min(), norm_outputs_te2.max()], 'k--', lw=1)
#ax2.set_xlabel('Truth')
#ax2.set_ylabel('Predicted')
#ax2.set_title('Validation period 2')
#ax2.text(.05,.9,'Mean Squared Error: {:.4e}'.format(lrpipe_te2_mse),transform=ax2.transAxes)
#
#plt.suptitle('Normalised Predicted values against GCM values network\ntuned to predict future value of '+temp, x=0.5, y=0.94, fontsize=15)
#
#plt.savefig('../../RegressionOutputs/'+exp_name+'_norm_predictedVtruth.png', bbox_inches = 'tight', pad_inches = 0.1)
#

#------------------------------------------------------
# de-normalise predicted values and plot against truth
#------------------------------------------------------
#
#fig = plt.figure(figsize=(20,9.4))
#
#denorm_lr_predicted_te1 = lr_predicted_te1*outputs_tr_std+outputs_tr_mean
#ax1 = fig.add_subplot(121)
#ax1.scatter(outputs_te1, denorm_lr_predicted_te1, edgecolors=(0, 0, 0))
#ax1.plot([outputs_te1.min(), outputs_te1.max()], [outputs_te1.min(), outputs_te1.max()], 'k--', lw=1)
#ax1.set_xlabel('Truth')
#ax1.set_ylabel('Predicted')
#ax1.set_title('Validation period 1')
#ax1.text(.05,.9,'Mean Squared Error: {:.4e}'.format(lrpipe_te1_mse),transform=ax1.transAxes)
#
#denorm_lr_predicted_te2 = lr_predicted_te2*outputs_tr_std+outputs_tr_mean
#ax2 = fig.add_subplot(122)
#ax2.scatter(outputs_te2, denorm_lr_predicted_te2, edgecolors=(0, 0, 0))
#ax2.plot([outputs_te2.min(), outputs_te2.max()], [outputs_te2.min(), outputs_te2.max()], 'k--', lw=1)
#ax2.set_xlabel('Truth')
#ax2.set_ylabel('Predicted')
#ax2.set_title('Validation period 2')
#ax2.text(.05,.9,'Mean Squared Error: {:.4e}'.format(lrpipe_te2_mse),transform=ax2.transAxes)
#
#plt.suptitle('Temperature predicted by Linear Regression model\nagainst GCM data (truth)', x=0.5, y=0.94, fontsize=15)
#
#plt.savefig('../../RegressionOutputs/'+exp_name+'_predictedVtruth.png', bbox_inches = 'tight', pad_inches = 0.1)
#
#
#------------------------------------------------------------------------------------------------------------------
# plot normalised prediction-original value, against truth-original value (convert to look like tendancy plot....)
#------------------------------------------------------------------------------------------------------------------
#
#fig = plt.figure(figsize=(20,9.4))
#ax1 = fig.add_subplot(121)
#ax1.scatter(outputs_te1-inputs_tr[:,xy], denorm_lr_predicted_te1-inputs_tr[:,xy], edgecolors=(0, 0, 0))
#ax1.set_xlabel('Truth')
#ax1.set_ylabel('Predicted')
#ax1.set_title('Validation period 1')
#ax1.text(.05,.9,'Mean Squared Error: {:.4e}'.format(lrpipe_te1_mse),transform=ax1.transAxes)
#
#ax2 = fig.add_subplot(122)
#ax2.scatter(outputs_te2-inputs_tr[:,xy], denorm_lr_predicted_te2-inputs_tr[:,xy], edgecolors=(0, 0, 0))
#ax2.set_xlabel('Truth')
#ax2.set_ylabel('Predicted')
#ax2.set_title('Validation period 2')
#ax2.text(.05,.9,'Mean Squared Error: {:.4e}'.format(lrpipe_te2_mse),transform=ax2.transAxes)
#
#plt.suptitle('Normalised Predicted values against GCM values network\ntuned to predict future value of '+temp, x=0.5, y=0.94, fontsize=15)
#plt.savefig('../../RegressionOutputs/'+exp_name+'_norm_predictedVtruth_diffastend.png', bbox_inches = 'tight', pad_inches = 0.1)
#
#
##-----------------------------
## Run random forest regressor
##-----------------------------
#
#n_estimators_set=[10,50,100,200]
#max_depth_set = [None,2,3,4,7,10]
#
#parameters = {'max_depth': max_depth_set, 'n_estimators':n_estimators_set}
#n_folds=3
#
#Rf = RandomForestRegressor()
#Rf_cv = GridSearchCV(Rf, param_grid=parameters, cv=n_folds, scoring='neg_mean_squared_error', refit=True)
#Rf_cv.fit(norm_inputs_tr, np.ravel(norm_outputs_tr))
#
#pkl_filename = 'PICKLED_MODELS/pickle_rfr_'+exp_name+'.pkl'
#with open(pkl_filename, 'wb') as file:
#    pickle.dump(lrpipe, file)
#
#results = Rf_cv.cv_results_
#best_params=Rf_cv.best_params_
#
#rf_predicted_train = Rf_cv.predict(norm_inputs_tr)
#rf_train_mse = metrics.mean_squared_error(norm_outputs_tr, rf_predicted_train)
#
#rf_predicted_te1 = Rf_cv.predict(norm_inputs_te1)
#rf_te1_mse = metrics.mean_squared_error(norm_outputs_te1, rf_predicted_te1)
#
#rf_predicted_te2 = Rf_cv.predict(norm_inputs_te2)
#rf_te2_mse = metrics.mean_squared_error(norm_outputs_te2, rf_predicted_te2)
#
#params_file.write('Random Forest Regressor \n')
#params_file.write(str(Rf_cv.best_params_))
#params_file.write('\n')
#params_file.write('\n')
#
##---------------------------------
## Run gradient boosting regressor
##---------------------------------
#
#max_depth_set = [2,3,4,7,10,15]
#n_estimators_set = [50,75,100,150]
#learning_rate_set = [0.05, 0.1, 0.2]
#
#parameters = {'max_depth': max_depth_set, 'n_estimators':n_estimators_set, 'learning_rate':learning_rate_set}
#n_folds=3
#
#gbr = GradientBoostingRegressor()
#gbr_cv = GridSearchCV(gbr, parameters, cv=n_folds, scoring='neg_mean_squared_error', refit=True)
#gbr_cv.fit(norm_inputs_tr, np.ravel(norm_outputs_tr))
#
#pkl_filename = 'PICKLED_MODELS/pickle_gbr_'+exp_name+'.pkl'
#with open(pkl_filename, 'wb') as file:
#    pickle.dump(lrpipe, file)
#
#results = gbr_cv.cv_results_
#best_params=gbr_cv.best_params_
#
#gbr_predicted_train = gbr_cv.predict(norm_inputs_tr)
#gbr_train_mse = metrics.mean_squared_error(norm_outputs_tr, gbr_predicted_train)
#
#gbr_predicted_te1 = gbr_cv.predict(norm_inputs_te1)
#gbr_te1_mse = metrics.mean_squared_error(norm_outputs_te1, gbr_predicted_te1)
#
#gbr_predicted_te2 = gbr_cv.predict(norm_inputs_te2)
#gbr_te2_mse = metrics.mean_squared_error(norm_outputs_te2, gbr_predicted_te2)
#
#params_file.write('Gradient Boosting Regressor \n')
#params_file.write(str(gbr_cv.best_params_))
#params_file.write('\n')
#params_file.write('\n')
#
#
##---------------------------------
## Run MLP NN regressor
##---------------------------------
#
#hidden_layer_sizes_set = [(50,), (50,50), (50,50,50), (100,), (100,100), (100,100,100), (100,100,100,100), (200,), (200,200), (200,200,200)]
#alpha_set = [0.00, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10, 30]
#
#parameters = {'alpha': alpha_set, 'hidden_layer_sizes': hidden_layer_sizes_set}
#n_folds=3
#
#mlp = MLPRegressor()
#mlp_cv = GridSearchCV(mlp, param_grid=parameters, cv=n_folds, scoring='neg_mean_squared_error', refit=True)
#mlp_cv.fit(norm_inputs_tr, np.ravel(norm_outputs_tr))
#
#pkl_filename = 'PICKLED_MODELS/pickle_mlp_'+exp_name+'.pkl'
#with open(pkl_filename, 'wb') as file:
#    pickle.dump(lrpipe, file)
#
#results = mlp_cv.cv_results_
#best_params=mlp_cv.best_params_
#
#mlp_predicted_train = mlp_cv.predict(norm_inputs_tr)
#mlp_train_mse = metrics.mean_squared_error(norm_outputs_tr, mlp_predicted_train)
#
#mlp_predicted_te1 = mlp_cv.predict(norm_inputs_te1)
#mlp_te1_mse = metrics.mean_squared_error(norm_outputs_te1, mlp_predicted_te1)
#
#mlp_predicted_te2 = mlp_cv.predict(norm_inputs_te2)
#mlp_te2_mse = metrics.mean_squared_error(norm_outputs_te2, mlp_predicted_te2)
#
#params_file.write('MLP Regressor \n')
#params_file.write(str(mlp_cv.best_params_))
#params_file.write('\n')
#params_file.write('\n')
#
##--------------------------------------------
## Print all scores to file
##--------------------------------------------
#
file=open('../../RegressionOutputs/'+exp_name+'.txt',"w+")
#file.write('shape for inputs and outputs.\n')
#file.write(str(inputs.shape)+'  '+str(outputs.shape)+'\n')
file.write('shape for inputs and outputs. Tr; test1; test2 \n')
file.write(str(inputs_tr.shape)+'  '+str(outputs_tr.shape)+'\n')
file.write(str(inputs_te1.shape)+'  '+str(outputs_te1.shape)+'\n')
file.write(str(inputs_te2.shape)+'  '+str(outputs_te2.shape)+'\n')
file.write('\n')
file.write('--------------------------------------------------------')
file.write('\n')
file.write('Training Scores: \n')
file.write('\n')
file.write(' persistance rms score %.10f; \n' %  np.sqrt(pers_train_mse))
file.write('          lr rms score %.10f; \n' %  np.sqrt(lrpipe_train_mse))
#file.write(' rf   rms score %.10f; \n' %  np.sqrt(rf_train_mse))
#file.write(' gbr  rms score %.10f; \n' %  np.sqrt(gbr_train_mse))
#file.write(' mlp  rms score %.10f; \n' %  np.sqrt(mlp_train_mse))
#file.write('\n')
#file.write('--------------------------------------------------------')
#file.write('\n')
#file.write('Test Scores: \n')
#file.write(' persistance rms score %.10f; \n' %  np.sqrt(pers_te_mse))
#file.write('     lr test rms error %.10f; \n' % np.sqrt(lrpipe_te_mse))
#file.write('\n')
file.write('--------------------------------------------------------')
file.write('\n')
file.write('Validation Period 1: \n')
file.write('\n')
file.write(' persistance rms  error %.10f; \n' % np.sqrt(pers_te1_mse))
file.write(' lr test rms error %.10f; \n' % np.sqrt(lrpipe_te1_mse))
#file.write(' rf rms  error %.10f; \n' % np.sqrt(rf_te1_mse))
#file.write(' gbr rms  error %.10f; \n' % np.sqrt(gbr_te1_mse))
#file.write(' mlp rms  error %.10f; \n' % np.sqrt(mlp_te1_mse))
file.write('\n')
file.write('--------------------------------------------------------')
file.write('\n')
file.write('Validation Period 2:\n')
file.write('\n')
file.write(' persistance rms  error %.10f; \n' % np.sqrt(pers_te2_mse))
file.write(' lr rms  error %.10f; \n' % np.sqrt(lrpipe_te2_mse))
#file.write(' rf rms  error %.10f; \n' % np.sqrt(rf_te2_mse))
#file.write(' gbr rms  error %.10f; \n' % np.sqrt(gbr_te2_mse))
#file.write(' mlp rms  error %.10f; \n' % np.sqrt(mlp_te2_mse))
#file.write('\n')
file.close() 
