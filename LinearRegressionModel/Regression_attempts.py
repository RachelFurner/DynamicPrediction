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
from sklearn.metrics import classification_report, r2_score

import sklearn.metrics as metrics
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.neural_network import MLPRegressor

import pickle

#---------------
# Set variables
#---------------

StepSize = 1 # how many output steps (months!) to predict over

halo_size = 2
halo_list = (range(-halo_size, halo_size+1))
#Calculate x,y position in each feature list to use as 'now' value
xy=2*(halo_size^2+halo_size) # if using a 2-d halo in x and y....if using a 3-d halo need to recalculate!

# region at start of run to learn from
tr_split_index_yr = 200                    # how many years to learn from
tr_split_index    = 12*tr_split_index_yr   # convert to from years to months

te_split_index_yr = 2000                   # Define first set of test data from period when model is still dynamically active
te_split_index    = 12*te_split_index_yr  

my_var = 'Ttave'
#my_var = 'uVeltave'
#my_var = 'Stave'
#my_var = 'vVeltave'

exp_name=my_var+'_tr'+str(tr_split_index_yr)+'_halo'+str(halo_size)+'_pred'+str(StepSize)


#------------------
# Read in the data
#------------------

DIR = '/data/hpcdata/users/racfur/MITGCM_OUTPUT/'
exp = '20000yr_Windx1.00_mm_diag/'
filename=DIR+exp+'cat_tave_selectvars_5000yrs.nc'
print(filename)
ds   = xr.open_dataset(filename)

ds_var=ds[my_var]


#------------------------------------------------------------------------
# Plot a timeseries of variable value at set grid point and its halo, 
# to use this to predict the value of that variable at a later time step
#------------------------------------------------------------------------

#z=np.random.randint(halo_size,41-halo_size) # 42 points in z-dir - skip edge points, don't want to be near surface or bottom boundary *for now*.
#x=np.random.randint(halo_size,10-1-halo_size)  # 11 points in x dir - skip edge points to allow non-boundary halo - skip extra point as its land
#y=np.random.randint(halo_size,77-1-halo_size)  # 78 points in y dir - skip edge points to allow non-boundary halo - skip extra point as its land

#colours=['blue' , 'green', 'red', 'cyan', 'magenta', 'yellow', 'orange', 'purple', 'aquamarine',
#         'gold', 'brown', 'pink', 'grey', 'olive', 'yellowgreen', 'violet', 'teal', 'sienna',
#         'salmon', 'plum', 'navy', 'orangered', 'tan', 'lightblue', 'khaki', 'indigo', 
#         'darkgreen', 'crimson']
#colours.insert(xy, 'black')
#styles=['dotted'] * 28
#styles.insert(xy, 'solid')
#alphas=[0.4] * 28
#alphas.insert(xy, 1.)
#thicks=['0.5'] * 28
#thicks.insert(xy, '2')
#
#fig1 = plt.figure(figsize=(18,4))
#i=0
#for x_offset in halo_list:
#    for y_offset in halo_list:
#        ds_var[:tr_split_index,z,y+y_offset,x+x_offset].plot(label=('x+%d,y+%d' % (x_offset, y_offset)), 
#                                                          alpha=1.0, xscale='log', color=colours[i], lw=thicks[i])
#        ds_var[tr_split_index:,z,y+y_offset,x+x_offset].plot(alpha=0.4, xscale='log', color=colours[i], lw=thicks[i])
#        plt.xlabel('model time [s] (log scale)')
#        i=i+1
#
#fig1.legend(loc='center right')
#fig1.savefig('../../regression_plots/'+exp_name+'_'+str(x)+'.'+str(y)+'.'+str(z)+'.png')


#------------------------------------------------------------------------------------------
# Read in data as training and test data - test data is split over two validation periods;
# 1 is while the model is still dynamically active, 2 is when it is near/at equilibrium
#------------------------------------------------------------------------------------------
inputs_tr = []
nxt_outputs_tr = []

inputs_te1 = []
nxt_outputs_te1 = []

inputs_te2 = []
nxt_outputs_te2 = []

# Read in as training and test data (rather than reading in all data and splitting),
# so we can learn on first n time steps, and test on rest
# supstep in space and time

for z in range(1,40,10):
    for x in range(2,7,3):
        for y in range(2,74,10):
            for time in range(0, tr_split_index, 10):
                inputs_tr.append([ds_var.isel(T=time)[z,y+y_offset,x+x_offset] for x_offset in halo_list for y_offset in halo_list])
                nxt_outputs_tr.append([ds_var.isel(T=time+StepSize)[z,y,x]])
                
            for time in range(tr_split_index, te_split_index, 50):
                inputs_te1.append([ds_var.isel(T=time)[z,y+y_offset,x+x_offset] for x_offset in halo_list for y_offset in halo_list])
                nxt_outputs_te1.append([ds_var.isel(T=time+StepSize)[z,y,x]])

            for time in range(te_split_index, len(ds.T.data)-StepSize, 100):
                inputs_te2.append([ds_var.isel(T=time)[z,y+y_offset,x+x_offset] for x_offset in halo_list for y_offset in halo_list])
                nxt_outputs_te2.append([ds_var.isel(T=time+StepSize)[z,y,x]])
                
inputs_tr=np.asarray(inputs_tr)
nxt_outputs_tr=np.asarray(nxt_outputs_tr)

inputs_te1=np.asarray(inputs_te1)
nxt_outputs_te1=np.asarray(nxt_outputs_te1)

inputs_te2=np.asarray(inputs_te2)
nxt_outputs_te2=np.asarray(nxt_outputs_te2)

###calculate outputs for tendancy model as the difference between 'next' and 'now'
#tnd_outputs_tr  = nxt_outputs_tr[:,0]  - inputs_tr[:,xy] 
#tnd_outputs_te1 = nxt_outputs_te1[:,0] - inputs_te1[:,xy]
#tnd_outputs_te2 = nxt_outputs_te2[:,0] - inputs_te2[:,xy]

print('shape for inputs and outputs. Tr, test1, test2')
print(inputs_tr.shape, nxt_outputs_tr.shape)
print(inputs_te1.shape, nxt_outputs_te1.shape)
print(inputs_te2.shape, nxt_outputs_te2.shape)
#print(inputs_tr.shape, nxt_outputs_tr.shape, tnd_outputs_tr.shape)
#print(inputs_te1.shape, nxt_outputs_te1.shape, tnd_outputs_te1.shape)
#print(inputs_te2.shape, nxt_outputs_te2.shape, tnd_outputs_te2.shape)


##-------------------------------------------------------------------------------
## Check the data by outputting and plotting a few random test and train samples
##-------------------------------------------------------------------------------
#
#sample_no=np.random.randint(0,inputs_tr.shape[0])
#print(sample_no)
#print(inputs_tr[sample_no,:])
#print(inputs_tr[sample_no,xy])
#print(nxt_outputs_tr[sample_no])
##print(tnd_outputs_tr[sample_no])
#
#features = range(inputs_tr.shape[1])
#fig = plt.figure()
#ax=plt.subplot()
#for feature in features:
#    plt.scatter(1,inputs_tr[sample_no,feature])
#plt.scatter(2,nxt_outputs_tr[sample_no])
##plt.scatter(3,tnd_outputs_tr[sample_no])
#ax.set_xlim(0,4)
#
#sample_no=np.random.randint(0,inputs_te1.shape[0])
#print('')
#print(sample_no)
#print(inputs_te1[sample_no,:])
#print(inputs_te1[sample_no,xy])
#print(nxt_outputs_te1[sample_no])
##print(tnd_outputs_te1[sample_no])
#
#features = range(inputs_te1.shape[1])
#fig = plt.figure()
#ax=plt.subplot()
#for feature in features:
#    plt.scatter(1,inputs_te1[sample_no,feature])
#plt.scatter(2,nxt_outputs_te1[sample_no])
##plt.scatter(3,tnd_outputs_te1[sample_no])
#ax.set_xlim(0,4)
#
#sample_no=np.random.randint(0,inputs_te2.shape[0])
#print('')
#print(sample_no)
#print(inputs_te2[sample_no,:])
#print(inputs_te2[sample_no,xy])
#print(nxt_outputs_te2[sample_no])
##print(tnd_outputs_te2[sample_no])
#
#features = range(inputs_te2.shape[1])
#fig = plt.figure()
#ax=plt.subplot()
#for feature in features:
#    plt.scatter(1,inputs_te2[sample_no,feature])
#plt.scatter(2,nxt_outputs_te2[sample_no])
##plt.scatter(3,tnd_outputs_te2[sample_no])
#ax.set_xlim(0,4)


#----------------------------------------------
# Normalise Data (based on training data only)
#----------------------------------------------

def normalise_data(train,test1,test2):
    train_mean, train_std = np.mean(train), np.std(train)
    norm_train = (train - train_mean) / train_std
    norm_test1 = (test1 - train_mean) / train_std
    norm_test2 = (test2 - train_mean) / train_std
    return norm_train, norm_test1, norm_test2

# normalise inputs
norm_inputs_tr  = np.zeros(inputs_tr.shape)
norm_inputs_te1 = np.zeros(inputs_te1.shape)
norm_inputs_te2 = np.zeros(inputs_te2.shape)

for i in range(inputs_tr.shape[1]):  #loop over each feature, normalising individually
    norm_inputs_tr[:, i], norm_inputs_te1[:, i], norm_inputs_te2[:, i] = normalise_data(inputs_tr[:, i], inputs_te1[:, i], inputs_te2[:, i])

#normalise nxt and tnd outputs
norm_nxt_outputs_tr, norm_nxt_outputs_te1, norm_nxt_outputs_te2 = normalise_data(nxt_outputs_tr[:], nxt_outputs_te1[:], nxt_outputs_te2[:])
#norm_tnd_outputs_tr, norm_tnd_outputs_te1, norm_tnd_outputs_te2 = normalise_data(tnd_outputs_tr[:], tnd_outputs_te1[:], tnd_outputs_te2[:])

# Calc mean and std of outputs re-forming predictions
nxt_outputs_tr_mean = np.mean(nxt_outputs_tr)
nxt_outputs_tr_std = np.std(nxt_outputs_tr)
#tnd_outputs_tr_mean = np.mean(tnd_outputs_tr)
#tnd_outputs_tr_std  = np.std(tnd_outputs_tr)


#---------------------------------------------------------------------
#---------------------------------------------------------------------
# Set up a model to directly predict variable value at next time step
#---------------------------------------------------------------------
#---------------------------------------------------------------------

#---------------------------------------------------------
# First calculate 'persistance' score, to give a baseline
#---------------------------------------------------------
# For training data
predict_persistance_nxt_train = norm_inputs_tr[:,xy]
pers_nxt_train_mse = metrics.mean_squared_error(norm_nxt_outputs_tr, predict_persistance_nxt_train)

# For validation period 1
predict_persistance_nxt1 = norm_inputs_te1[:,xy]

pers_nxt1_r2 = r2_score(norm_nxt_outputs_te1, predict_persistance_nxt1)
pers_nxt1_maxer = metrics.max_error(norm_nxt_outputs_te1, predict_persistance_nxt1)
pers_nxt1_mse = metrics.mean_squared_error(norm_nxt_outputs_te1, predict_persistance_nxt1)

#print('Validation period 1:')
#print('persistance r2 score ; ', pers_nxt1_r2)
#print('persistance max error ; ', pers_nxt1_maxer)
#print('persistance mean squared error ; ', pers_nxt1_mse)

# For validation period 2
predict_persistance_nxt2 = norm_inputs_te2[:,xy]

pers_nxt2_r2 = r2_score(norm_nxt_outputs_te2, predict_persistance_nxt2)
pers_nxt2_maxer = metrics.max_error(norm_nxt_outputs_te2, predict_persistance_nxt2)
pers_nxt2_mse = metrics.mean_squared_error(norm_nxt_outputs_te2, predict_persistance_nxt2)

#print('Validation period 2:')
#print('persistance r2 score ; ', pers_nxt2_r2)
#print('persistance max error ; ', pers_nxt2_maxer)
#print('persistance mean squared error ; ', pers_nxt2_mse)


#---------------------------------------------------------------
# Tune alpha using cross validation, and neg mean squared error
#---------------------------------------------------------------

alpha_s = [0.00, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10, 30]

parameters = {'alpha': alpha_s}
n_folds=3
scoring={'max_error', 'neg_mean_squared_error', 'r2'}

lr_nxt=linear_model.Ridge()

lr_nxt_cv = GridSearchCV(lr_nxt, param_grid=parameters, cv=n_folds, scoring=scoring, refit='neg_mean_squared_error')
lr_nxt_cv.fit(norm_inputs_tr, norm_nxt_outputs_tr)
results = lr_nxt_cv.cv_results_

best_params=lr_nxt_cv.best_params_
best_alpha = (best_params['alpha'])

params_file=open('../../regression_plots/'+exp_name+'_bestparams.txt',"w+")
params_file.write('linear regressor \n')
params_file.write(str(lr_nxt_cv.best_params_))
params_file.write('\n')
params_file.write('\n')

#-------------------------------------------------------------------------------------------
# For neg mean squared error, calculate and plot the mean score, and the region +/- one s.d
#-------------------------------------------------------------------------------------------
#
#neg_mean_squared_error_mean = lr_nxt_cv.cv_results_['mean_test_neg_mean_squared_error']
#neg_mean_squared_error_std = lr_nxt_cv.cv_results_['std_test_neg_mean_squared_error']
#neg_mean_squared_error_std_error = neg_mean_squared_error_std / np.sqrt(n_folds)
#
#plt.figure().set_size_inches(8, 6)
#plt.semilogx(alpha_s, neg_mean_squared_error_mean, label='mean neg-mean-squared-error from CV', color='blue')
#plt.fill_between(alpha_s, neg_mean_squared_error_mean + neg_mean_squared_error_std_error,
#                 neg_mean_squared_error_mean - neg_mean_squared_error_std_error, alpha=0.1, color='blue')
#plt.ylabel('CV score +/- std error')
#plt.xlabel('alpha')
##plt.axhline(np.max(r2_scores), linestyle='--', color='.5')
#
## Also calculate training and validation scores for same values of alpha without CV 
## and overplot
#
#val1_scores = []
#val2_scores = []
#train_scores = []
#for alpha in alpha_s:
#    lr_nxt.set_params(alpha=alpha)
#    lr_nxt.fit(norm_inputs_tr, norm_nxt_outputs_tr)
#    train_scores.append(- metrics.mean_squared_error(norm_nxt_outputs_tr, lr_nxt.predict(norm_inputs_tr)) )
#    val1_scores.append(- metrics.mean_squared_error(norm_nxt_outputs_te1, lr_nxt.predict(norm_inputs_te1)) )
#    val2_scores.append(- metrics.mean_squared_error(norm_nxt_outputs_te2, lr_nxt.predict(norm_inputs_te2)) )
#plt.plot(alpha_s, train_scores, label='training score')
#plt.plot(alpha_s, val1_scores, label='validation period 1 score')
#plt.plot(alpha_s, val2_scores, label='validation period 2 score')
#plt.legend()
#plt.savefig('../../regression_plots/'+exp_name+'_alphacv_mse.png')
#
#
##-----------------------------------------------------------------------------
## For r2 score, calculate and plot the mean score, and the region +/- one s.d
##-----------------------------------------------------------------------------
#
#r2_mean = lr_nxt_cv.cv_results_['mean_test_r2']
#r2_std = lr_nxt_cv.cv_results_['std_test_r2']
#r2_std_error = r2_std / np.sqrt(n_folds)
#
#plt.figure().set_size_inches(8, 6)
#plt.semilogx(alpha_s, r2_mean, label='mean r2 score from CV', color='blue')
#plt.fill_between(alpha_s, r2_mean + r2_std_error, r2_mean - r2_std_error, alpha=0.1, color='blue')
#plt.ylabel('CV score +/- std error')
#plt.xlabel('alpha')
##plt.axhline(np.max(r2_scores), linestyle='--', color='.5')
#
## Also calculate training and validation scores for same values of alpha without CV 
## and overplot
#
#val1_scores = []
#val2_scores = []
#train_scores = []
#for alpha in alpha_s:
#    lr_nxt.set_params(alpha=alpha)
#    lr_nxt.fit(norm_inputs_tr, norm_nxt_outputs_tr)
#    train_scores.append(lr_nxt.score(norm_inputs_tr, norm_nxt_outputs_tr))
#    val1_scores.append(lr_nxt.score(norm_inputs_te1, norm_nxt_outputs_te1))#
#    val2_scores.append(lr_nxt.score(norm_inputs_te2, norm_nxt_outputs_te2))#
#plt.plot(alpha_s, train_scores, label='training score')
#plt.plot(alpha_s, val1_scores, label='validation period 1 score')
#plt.plot(alpha_s, val2_scores, label='validation period 2 score')
#plt.legend()
#plt.savefig('../../regression_plots/'+exp_name+'_alphacv_r2.png')
#
#
##---------------------------------------------------------------------------------------------
## For Max Error, plot the mean score, and the region +/- one s.d, (huge s.d. as you'd expect)
##---------------------------------------------------------------------------------------------
#
#max_error_mean = lr_nxt_cv.cv_results_['mean_test_max_error']
#max_error_std = lr_nxt_cv.cv_results_['std_test_max_error']
#max_error_std_error = max_error_std / np.sqrt(n_folds)
#
#plt.figure().set_size_inches(8, 6)
#plt.semilogx(alpha_s, max_error_mean, label='mean negative-max-error from CV', color='red')
#plt.fill_between(alpha_s, max_error_mean + max_error_std_error, max_error_mean - max_error_std_error, alpha=0.1, color='red')
#plt.ylabel('CV score +/- std error')
#plt.xlabel('alpha')
#
## Also calculate training and validation scores for same values of alpha without CV 
## and overplot 
#
#val1_scores = []
#val2_scores = []
#train_scores = []
#for alpha in alpha_s:
#    lr_nxt.set_params(alpha=alpha)
#    lr_nxt.fit(norm_inputs_tr, norm_nxt_outputs_tr)
#    train_scores.append(- metrics.max_error(norm_nxt_outputs_tr, lr_nxt.predict(norm_inputs_tr)) )
#    val1_scores.append(- metrics.max_error(norm_nxt_outputs_te1, lr_nxt.predict(norm_inputs_te1)) )
#    val2_scores.append(- metrics.max_error(norm_nxt_outputs_te2, lr_nxt.predict(norm_inputs_te2)) )
#plt.plot(alpha_s, train_scores, label='training score')
#plt.plot(alpha_s, val1_scores, label='validation period 1 score')
#plt.plot(alpha_s, val2_scores, label='validation period 2 score')
#plt.legend()
#plt.savefig('../../regression_plots/'+exp_name+'_alphacv_maxer.png')
#
#------------------------------------------------------------------------------------------------------------------------------------------------
# Having Tuned Alpha, run the model: predict next time step value, and assess (using best alpha selected from above based on mean-square scores)
#------------------------------------------------------------------------------------------------------------------------------------------------

lr_nxt= linear_model.Ridge(alpha=best_alpha)
# eventually might want to aim for a zero intercept, but model seems to struggle a lot with this so leave for now

lr_nxt.fit(norm_inputs_tr, norm_nxt_outputs_tr )  # train to evaluate the value at the next time step...
lr_nxt.get_params()

pkl_filename = 'pickle_LRmodel'+exp_name+'.pkl'
with open(pkl_filename, 'wb') as file:
    pickle.dump(lr_nxt, file)

#print('coefs     : ' + str(lr_nxt.coef_))
#print('intercept : ' + str(lr_nxt.intercept_))
#print(' ')

lr_predicted_nxt_train = lr_nxt.predict(norm_inputs_tr)
lr_nxt_train_mse = metrics.mean_squared_error(norm_nxt_outputs_tr, lr_predicted_nxt_train)

lr_predicted_nxt1 = lr_nxt.predict(norm_inputs_te1)
lr_nxt1_r2 = r2_score(norm_nxt_outputs_te1, lr_predicted_nxt1)
lr_nxt1_maxer = metrics.max_error(norm_nxt_outputs_te1, lr_predicted_nxt1)
lr_nxt1_mse = metrics.mean_squared_error(norm_nxt_outputs_te1, lr_predicted_nxt1)

#print('Validation period 1')
#print('persistance r2 score ; ', pers_nxt1_r2)
#print('persistance max error ; ', pers_nxt1_maxer)
#print('persistance mean squared error ; ', pers_nxt1_mse)
#print('')
#print('model r2 score ; ', nxt1_r2)
#print('model max error ; ', nxt1_maxer)
#print('model mean squared error ; ', nxt1_mse)

lr_predicted_nxt2 = lr_nxt.predict(norm_inputs_te2)
lr_nxt2_r2 = r2_score(norm_nxt_outputs_te2, lr_predicted_nxt2)
lr_nxt2_maxer = metrics.max_error(norm_nxt_outputs_te2, lr_predicted_nxt2)
lr_nxt2_mse = metrics.mean_squared_error(norm_nxt_outputs_te2, lr_predicted_nxt2)

#print('Validation period 1')
#print('persistance r2 score ; ', pers_nxt2_r2)
#print('persistance max error ; ', pers_nxt2_maxer)
#print('persistance mean squared error ; ', pers_nxt2_mse)
#print('')
#print('model r2 score ; ', nxt2_r2)
#print('model max error ; ', nxt2_maxer)
#print('model mean squared error ; ', nxt2_mse)


##------------------------------------------
## Plot normalised prediction against truth
##------------------------------------------
#fig = plt.figure(figsize=(20,9.4))
#ax1 = fig.add_subplot(121)
#ax1.scatter(norm_nxt_outputs_te1, lr_predicted_nxt1, edgecolors=(0, 0, 0))
#ax1.plot([norm_nxt_outputs_te1.min(), norm_nxt_outputs_te1.max()], [norm_nxt_outputs_te1.min(), norm_nxt_outputs_te1.max()], 'k--', lw=1)
#ax1.set_xlabel('Truth')
#ax1.set_ylabel('Predicted')
#ax1.set_title('Validation period 1')
#ax1.text(.05,.9,'Mean Squared Error: {:.4e}'.format(lr_nxt1_mse),transform=ax1.transAxes)
#
#ax2 = fig.add_subplot(122)
#ax2.scatter(norm_nxt_outputs_te2, lr_predicted_nxt2, edgecolors=(0, 0, 0))
#ax2.plot([norm_nxt_outputs_te2.min(), norm_nxt_outputs_te2.max()], [norm_nxt_outputs_te2.min(), norm_nxt_outputs_te2.max()], 'k--', lw=1)
#ax2.set_xlabel('Truth')
#ax2.set_ylabel('Predicted')
#ax2.set_title('Validation period 2')
#ax2.text(.05,.9,'Mean Squared Error: {:.4e}'.format(lr_nxt2_mse),transform=ax2.transAxes)
#
#plt.suptitle('Normalised Predicted values against GCM values network\ntuned to predict future value of '+my_var, x=0.5, y=0.94, fontsize=15)
#
#plt.savefig('../../regression_plots/'+exp_name+'_norm_predictedVtruth_nxt.png', bbox_inches = 'tight', pad_inches = 0.1)
#

#------------------------------------------------------
# de-normalise predicted values and plot against truth
#------------------------------------------------------

fig = plt.figure(figsize=(20,9.4))

denorm_lr_predicted_nxt1 = lr_predicted_nxt1*nxt_outputs_tr_std+nxt_outputs_tr_mean
ax1 = fig.add_subplot(121)
ax1.scatter(nxt_outputs_te1, denorm_lr_predicted_nxt1, edgecolors=(0, 0, 0))
ax1.plot([nxt_outputs_te1.min(), nxt_outputs_te1.max()], [nxt_outputs_te1.min(), nxt_outputs_te1.max()], 'k--', lw=1)
ax1.set_xlabel('Truth')
ax1.set_ylabel('Predicted')
ax1.set_title('Validation period 1')
ax1.text(.05,.9,'Mean Squared Error: {:.4e}'.format(lr_nxt1_mse),transform=ax1.transAxes)

denorm_lr_predicted_nxt2 = lr_predicted_nxt2*nxt_outputs_tr_std+nxt_outputs_tr_mean
ax2 = fig.add_subplot(122)
ax2.scatter(nxt_outputs_te2, denorm_lr_predicted_nxt2, edgecolors=(0, 0, 0))
ax2.plot([nxt_outputs_te2.min(), nxt_outputs_te2.max()], [nxt_outputs_te2.min(), nxt_outputs_te2.max()], 'k--', lw=1)
ax2.set_xlabel('Truth')
ax2.set_ylabel('Predicted')
ax2.set_title('Validation period 2')
ax2.text(.05,.9,'Mean Squared Error: {:.4e}'.format(lr_nxt2_mse),transform=ax2.transAxes)

plt.suptitle('Temperature predicted by Linear Regression model\nagainst GCM data (truth)', x=0.5, y=0.94, fontsize=15)

plt.savefig('../../regression_plots/'+exp_name+'_predictedVtruth_nxt.png', bbox_inches = 'tight', pad_inches = 0.1)


##------------------------------------------------------------------------------------------------------------------
## plot normalised prediction-original value, against truth-original value (convert to look like tendancy plot....)
##------------------------------------------------------------------------------------------------------------------
#
#fig = plt.figure(figsize=(20,9.4))
#ax1 = fig.add_subplot(121)
#ax1.scatter(nxt_outputs_te1-inputs_tr[:,xy], denorm_lr_predicted_nxt1-inputs_tr[:,xy], edgecolors=(0, 0, 0))
#ax1.set_xlabel('Truth')
#ax1.set_ylabel('Predicted')
#ax1.set_title('Validation period 1')
#ax1.text(.05,.9,'Mean Squared Error: {:.4e}'.format(lr_nxt1_mse),transform=ax1.transAxes)
#
#ax2 = fig.add_subplot(122)
#ax2.scatter(nxt_outputs_te2-inputs_tr[:,xy], denorm_lr_predicted_nxt2-inputs_tr[:,xy], edgecolors=(0, 0, 0))
#ax2.set_xlabel('Truth')
#ax2.set_ylabel('Predicted')
#ax2.set_title('Validation period 2')
#ax2.text(.05,.9,'Mean Squared Error: {:.4e}'.format(lr_nxt2_mse),transform=ax2.transAxes)
#
#plt.suptitle('Normalised Predicted values against GCM values network\ntuned to predict future value of '+my_var, x=0.5, y=0.94, fontsize=15)
#plt.savefig('../../regression_plots/'+exp_name+'_norm_predictedVtruth_nxt_diffastend.png', bbox_inches = 'tight', pad_inches = 0.1)
#
#
#-----------------------------
# Run random forest regressor
#-----------------------------

n_estimators_set=[10,50,100,200]
max_depth_set = [None,2,3,4,7,10]

parameters = {'max_depth': max_depth_set, 'n_estimators':n_estimators_set}
n_folds=3

Rf_nxt = RandomForestRegressor()
Rf_nxt_cv = GridSearchCV(Rf_nxt, param_grid=parameters, cv=n_folds, scoring='neg_mean_squared_error', refit=True)
Rf_nxt_cv.fit(norm_inputs_tr, np.ravel(norm_nxt_outputs_tr))

results = Rf_nxt_cv.cv_results_
best_params=Rf_nxt_cv.best_params_

rf_predicted_nxt_train = Rf_nxt_cv.predict(norm_inputs_tr)
rf_nxt_train_mse = metrics.mean_squared_error(norm_nxt_outputs_tr, rf_predicted_nxt_train)

rf_predicted_nxt1 = Rf_nxt_cv.predict(norm_inputs_te1)
rf_nxt1_r2 = r2_score(norm_nxt_outputs_te1, rf_predicted_nxt1)
rf_nxt1_maxer = metrics.max_error(norm_nxt_outputs_te1, rf_predicted_nxt1)
rf_nxt1_mse = metrics.mean_squared_error(norm_nxt_outputs_te1, rf_predicted_nxt1)

rf_predicted_nxt2 = Rf_nxt_cv.predict(norm_inputs_te2)
rf_nxt2_r2 = r2_score(norm_nxt_outputs_te2, rf_predicted_nxt2)
rf_nxt2_maxer = metrics.max_error(norm_nxt_outputs_te2, rf_predicted_nxt2)
rf_nxt2_mse = metrics.mean_squared_error(norm_nxt_outputs_te2, rf_predicted_nxt2)

params_file.write('Random Forest Regressor \n')
params_file.write(str(Rf_nxt_cv.best_params_))
params_file.write('\n')
params_file.write('\n')

#---------------------------------
# Run gradient boosting regressor
#---------------------------------

max_depth_set = [2,3,4,7,10,15]
n_estimators_set = [50,75,100,150]
learning_rate_set = [0.05, 0.1, 0.2]

parameters = {'max_depth': max_depth_set, 'n_estimators':n_estimators_set, 'learning_rate':learning_rate_set}
n_folds=3

gbr_nxt = GradientBoostingRegressor()
gbr_nxt_cv = GridSearchCV(gbr_nxt, parameters, cv=n_folds, scoring='neg_mean_squared_error', refit=True)
gbr_nxt_cv.fit(norm_inputs_tr, np.ravel(norm_nxt_outputs_tr))

results = gbr_nxt_cv.cv_results_
best_params=gbr_nxt_cv.best_params_

gbr_predicted_nxt_train = gbr_nxt_cv.predict(norm_inputs_tr)
gbr_nxt_train_mse = metrics.mean_squared_error(norm_nxt_outputs_tr, gbr_predicted_nxt_train)

gbr_predicted_nxt1 = gbr_nxt_cv.predict(norm_inputs_te1)
gbr_nxt1_r2 = r2_score(norm_nxt_outputs_te1, gbr_predicted_nxt1)
gbr_nxt1_maxer = metrics.max_error(norm_nxt_outputs_te1, gbr_predicted_nxt1)
gbr_nxt1_mse = metrics.mean_squared_error(norm_nxt_outputs_te1, gbr_predicted_nxt1)

gbr_predicted_nxt2 = gbr_nxt_cv.predict(norm_inputs_te2)
gbr_nxt2_r2 = r2_score(norm_nxt_outputs_te2, gbr_predicted_nxt2)
gbr_nxt2_maxer = metrics.max_error(norm_nxt_outputs_te2, gbr_predicted_nxt2)
gbr_nxt2_mse = metrics.mean_squared_error(norm_nxt_outputs_te2, gbr_predicted_nxt2)

params_file.write('Gradient Boosting Regressor \n')
params_file.write(str(gbr_nxt_cv.best_params_))
params_file.write('\n')
params_file.write('\n')


#---------------------------------
# Run MLP NN regressor
#---------------------------------

hidden_layer_sizes_set = [(50,), (50,50), (50,50,50), (100,), (100,100), (100,100,100), (100,100,100,100), (200,), (200,200), (200,200,200)]
alpha_set = [0.00, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10, 30]

parameters = {'alpha': alpha_set, 'hidden_layer_sizes': hidden_layer_sizes_set}
n_folds=3

mlp_nxt = MLPRegressor()
mlp_nxt_cv = GridSearchCV(mlp_nxt, param_grid=parameters, cv=n_folds, scoring='neg_mean_squared_error', refit=True)
mlp_nxt_cv.fit(norm_inputs_tr, np.ravel(norm_nxt_outputs_tr))

results = mlp_nxt_cv.cv_results_
best_params=mlp_nxt_cv.best_params_

mlp_predicted_nxt_train = mlp_nxt_cv.predict(norm_inputs_tr)
mlp_nxt_train_mse = metrics.mean_squared_error(norm_nxt_outputs_tr, mlp_predicted_nxt_train)

mlp_predicted_nxt1 = mlp_nxt_cv.predict(norm_inputs_te1)
mlp_nxt1_r2 = r2_score(norm_nxt_outputs_te1, mlp_predicted_nxt1)
mlp_nxt1_maxer = metrics.max_error(norm_nxt_outputs_te1, mlp_predicted_nxt1)
mlp_nxt1_mse = metrics.mean_squared_error(norm_nxt_outputs_te1, mlp_predicted_nxt1)

mlp_predicted_nxt2 = mlp_nxt_cv.predict(norm_inputs_te2)
mlp_nxt2_r2 = r2_score(norm_nxt_outputs_te2, mlp_predicted_nxt2)
mlp_nxt2_maxer = metrics.max_error(norm_nxt_outputs_te2, mlp_predicted_nxt2)
mlp_nxt2_mse = metrics.mean_squared_error(norm_nxt_outputs_te2, mlp_predicted_nxt2)

params_file.write('MLP Regressor \n')
params_file.write(str(mlp_nxt_cv.best_params_))
params_file.write('\n')
params_file.write('\n')

##-------------------------------------------------------------------------------
##-------------------------------------------------------------------------------
## Define a new model to predict tendancy (difference between now and next step)
##-------------------------------------------------------------------------------
##-------------------------------------------------------------------------------
#
##------------------------------------------------------------------
## First calculate and plot 'persistance' score, to give a baseline
##------------------------------------------------------------------
#
## For Validation period 1
#
#predict_persistance_tnd1 = np.zeros([tnd_outputs_te1.shape[0]])
#pers_tnd1_r2 = r2_score(norm_tnd_outputs_te1, predict_persistance_tnd1)
#pers_tnd1_maxer = metrics.max_error(norm_tnd_outputs_te1, predict_persistance_tnd1)
#pers_tnd1_mse = metrics.mean_squared_error(norm_tnd_outputs_te1, predict_persistance_tnd1)
#print('Validation Period 1:')
#print('persistance r2 score ; ', pers_tnd1_r2)
#print('persistance max error ; ', pers_tnd1_maxer)
#print('persistance mean squared error ; ', pers_tnd1_mse)
#
## For Validation period 2
#
#predict_persistance_tnd2 = np.zeros([tnd_outputs_te2.shape[0]])
#pers_tnd2_r2 = r2_score(norm_tnd_outputs_te2, predict_persistance_tnd2)
#pers_tnd2_maxer = metrics.max_error(norm_tnd_outputs_te2, predict_persistance_tnd2)
#pers_tnd2_mse = metrics.mean_squared_error(norm_tnd_outputs_te2, predict_persistance_tnd2)
#print('Validation Period 2:')
#print('persistance r2 score ; ', pers_tnd2_r2)
#print('persistance max error ; ', pers_tnd2_maxer)
#print('persistance mean squared error ; ', pers_tnd2_mse)
#
#
##---------------------------------------------------------------
## Tune alpha using cross validation, and neg mean squared error
##---------------------------------------------------------------
#
#alpha_s = [0.00, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10, 30]
#
#parameters = [{'alpha': alpha_s}]
#n_folds=3
#scoring={'max_error', 'neg_mean_squared_error', 'r2'}
#
#lr_tnd=linear_model.Ridge()
#
#
##-------------------------------------------------------------------------------
## Calculate training scores using cross validation with various values of alpha
##-------------------------------------------------------------------------------
#
#lr_tnd_cv = GridSearchCV(lr_tnd, parameters, cv=n_folds, scoring=scoring, refit='neg_mean_squared_error')
#lr_tnd_cv.fit(norm_inputs_tr, norm_tnd_outputs_tr)
#results = lr_tnd_cv.cv_results_
#
#best_params=lr_tnd_cv.best_params_
#best_alpha = (best_params['alpha'])
#
#
##-------------------------------------------------------------------------------------------
## For neg mean squared error, calculate and plot the mean score, and the region +/- one s.d
##-------------------------------------------------------------------------------------------
#
#neg_mean_squared_error_mean = lr_tnd_cv.cv_results_['mean_test_neg_mean_squared_error']
#neg_mean_squared_error_std = lr_tnd_cv.cv_results_['std_test_neg_mean_squared_error']
#neg_mean_squared_error_std_error = neg_mean_squared_error_std / np.sqrt(n_folds)
#
#plt.figure().set_size_inches(8, 6)
#plt.semilogx(alpha_s, neg_mean_squared_error_mean, label='mean neg-mean-squared-error from CV', color='blue')
#plt.fill_between(alpha_s, neg_mean_squared_error_mean + neg_mean_squared_error_std_error,
#                 neg_mean_squared_error_mean - neg_mean_squared_error_std_error, alpha=0.1, color='blue')
#plt.ylabel('CV score +/- std error')
#plt.xlabel('alpha')
##plt.axhline(np.max(r2_scores), linestyle='--', color='.5')
#
## Also calculate training and validation scores for same values of alpha without CV 
## and overplot
#
#val1_scores = []
#val2_scores = []
#train_scores = []
#for alpha in alpha_s:
#    lr_tnd.set_params(alpha=alpha)
#    lr_tnd.fit(norm_inputs_tr, norm_tnd_outputs_tr)
#    train_scores.append(- metrics.mean_squared_error(norm_tnd_outputs_tr, lr_tnd.predict(norm_inputs_tr)) )
#    val1_scores.append(- metrics.mean_squared_error(norm_tnd_outputs_te1, lr_tnd.predict(norm_inputs_te1)) )
#    val2_scores.append(- metrics.mean_squared_error(norm_tnd_outputs_te2, lr_tnd.predict(norm_inputs_te2)) )
#plt.plot(alpha_s, train_scores, label='training score')
#plt.plot(alpha_s, val1_scores, label='validation period 1 score')
#plt.plot(alpha_s, val2_scores, label='validation period 1 score')
#plt.legend()
#
#
##-----------------------------------------------------------------------------
## For r2 score, calculate and plot the mean score, and the region +/- one s.d
##-----------------------------------------------------------------------------
#
#r2_mean = lr_tnd_cv.cv_results_['mean_test_r2']
#r2_std = lr_tnd_cv.cv_results_['std_test_r2']
#r2_std_error = r2_std / np.sqrt(n_folds)
#
#plt.figure().set_size_inches(8, 6)
#plt.semilogx(alpha_s, r2_mean, label='mean r2 score from CV', color='blue')
#plt.fill_between(alpha_s, r2_mean + r2_std_error, r2_mean - r2_std_error, alpha=0.1, color='blue')
#plt.ylabel('CV score +/- std error')
#plt.xlabel('alpha')
##plt.axhline(np.max(r2_scores), linestyle='--', color='.5')
#
## Also calculate training and validation scores for same values of alpha without CV 
## and overplot
#
#val1_scores = []
#val2_scores = []
#train_scores = []
#for alpha in alpha_s:
#    lr_tnd.set_params(alpha=alpha)
#    lr_tnd.fit(norm_inputs_tr, norm_tnd_outputs_tr)
#    train_scores.append(lr_tnd.score(norm_inputs_tr, norm_tnd_outputs_tr))
#    val1_scores.append(lr_tnd.score(norm_inputs_te1, norm_tnd_outputs_te1))#
#    val2_scores.append(lr_tnd.score(norm_inputs_te2, norm_tnd_outputs_te2))#
#plt.plot(alpha_s, train_scores, label='training score')
#plt.plot(alpha_s, val1_scores, label='validation period 1 score')
#plt.plot(alpha_s, val2_scores, label='validation period 2 score')
#plt.legend()
#
#
##---------------------------------------------------------------------------------------------
## For Max Error, plot the mean score, and the region +/- one s.d, (huge s.d. as you'd expect)
##---------------------------------------------------------------------------------------------
#
#max_error_mean = lr_tnd_cv.cv_results_['mean_test_max_error']
#max_error_std = lr_tnd_cv.cv_results_['std_test_max_error']
#max_error_std_error = max_error_std / np.sqrt(n_folds)
#
#plt.figure().set_size_inches(8, 6)
#plt.semilogx(alpha_s, max_error_mean, label='mean negative-max-error from CV', color='red')
#plt.fill_between(alpha_s, max_error_mean + max_error_std_error, max_error_mean - max_error_std_error, alpha=0.1, color='red')
#plt.ylabel('CV score +/- std error')
#plt.xlabel('alpha')
#
## Also calculate training and validation scores for same values of alpha without CV 
## and overplot
#
#val1_scores = []
#val2_scores = []
#train_scores = []
#for alpha in alpha_s:
#    lr_tnd.set_params(alpha=alpha)
#    lr_tnd.fit(norm_inputs_tr, norm_tnd_outputs_tr)
#    train_scores.append(- metrics.max_error(norm_tnd_outputs_tr, lr_tnd.predict(norm_inputs_tr)) )
#    val1_scores.append(- metrics.max_error(norm_tnd_outputs_te1, lr_tnd.predict(norm_inputs_te1)) )
#    val2_scores.append(- metrics.max_error(norm_tnd_outputs_te2, lr_tnd.predict(norm_inputs_te2)) )
#plt.plot(alpha_s, train_scores, label='training score')
#plt.plot(alpha_s, val1_scores, label='validation period 1 score')
#plt.plot(alpha_s, val2_scores, label='validation period 2 score')
#plt.legend()
#
#
##---------------------------------------------------------------------------------------------------------------------------------------
## Having Tuned Alpha, run the model: predict next time step value, and assess (using best alpha selected from above based on R2 scores)
##---------------------------------------------------------------------------------------------------------------------------------------
#
#lr_tnd= linear_model.Ridge(alpha=best_alpha, normalize=False)
## eventually might want to aim for a zero intercept, but model seems to struggle a lot with this so leave for now
#
#lr_tnd.fit(norm_inputs_tr, norm_tnd_outputs_tr )  # train to evaluate the value at the next time step...
#
#print('coefs     : ' + str(lr_tnd.coef_))
#print('intercept : ' + str(lr_tnd.intercept_))
#print(' ')
#
## For Validation period 1
#lr_predicted_tnd1 = lr_tnd.predict(norm_inputs_te1)
#tnd1_r2 = r2_score(norm_tnd_outputs_te1, lr_predicted_tnd1)
#tnd1_maxer = metrics.max_error(norm_tnd_outputs_te1, lr_predicted_tnd1)
#tnd1_mse = metrics.mean_squared_error(norm_tnd_outputs_te1, lr_predicted_tnd1)
#
#print('For Validation period 1:')
#print('persistance r2 score ; ', r2_score(norm_tnd_outputs_te1, predict_persistance_tnd1))
#print('persistance max error ; ', metrics.max_error(norm_tnd_outputs_te1, predict_persistance_tnd1))
#print('persistance mean squared error ; ', metrics.mean_squared_error(norm_tnd_outputs_te1, predict_persistance_tnd1))
#print('')
#print('model r2 score ; ', tnd1_r2)
#print('model max error ; ', tnd1_maxer)
#print('model mean squared error ; ', tnd1_mse)
#
## For Validation period 2
#lr_predicted_tnd2 = lr_tnd.predict(norm_inputs_te2)
#tnd2_r2 = r2_score(norm_tnd_outputs_te2, lr_predicted_tnd2)
#tnd2_maxer = metrics.max_error(norm_tnd_outputs_te2, lr_predicted_tnd2)
#tnd2_mse = metrics.mean_squared_error(norm_tnd_outputs_te2, lr_predicted_tnd2)
#
#print('For Validation period 2:')
#print('persistance r2 score ; ', r2_score(norm_tnd_outputs_te2, predict_persistance_tnd2))
#print('persistance max error ; ', metrics.max_error(norm_tnd_outputs_te2, predict_persistance_tnd2))
#print('persistance mean squared error ; ', metrics.mean_squared_error(norm_tnd_outputs_te2, predict_persistance_tnd2))
#print('')
#print('model r2 score ; ', tnd2_r2)
#print('model max error ; ', tnd2_maxer)
#print('model mean squared error ; ', tnd2_mse)
#
#
##------------------------------
## first plot normalised values
##------------------------------
#
#fig, ax = plt.subplots(figsize=(10,8))
#ax.scatter(norm_tnd_outputs_te1, lr_predicted_tnd1, edgecolors=(0, 0, 0))
#ax.plot([norm_tnd_outputs_te1.min(), norm_tnd_outputs_te1.max()], [norm_tnd_outputs_te1.min(), norm_tnd_outputs_te1.max()], 'k--', lw=1)
#ax.set_xlabel('Truth')
#ax.set_ylabel('Predicted')
#ax.set_title('Normalised predicted values against GCM values\nnetwork tuned to predict change in '+my_var+' over next time step\nfor validation period 1')
#ax.text(.05,.9,'Mean Squared Error: {:.4e}'.format(tnd1_mse),transform=ax.transAxes)
#plt.savefig('../../regression_plots/'+exp_name+'_predictedVtruth_tnd1.png')
#
#fig, ax = plt.subplots(figsize=(10,8))
#ax.scatter(norm_tnd_outputs_te2, lr_predicted_tnd2, edgecolors=(0, 0, 0))
#ax.plot([norm_tnd_outputs_te2.min(), norm_tnd_outputs_te2.max()], [norm_tnd_outputs_te2.min(), norm_tnd_outputs_te2.max()], 'k--', lw=1)
#ax.set_xlabel('Truth')
#ax.set_ylabel('Predicted')
#ax.set_title('Normalised predicted values against GCM values\nnetwork tuned to predict change in '+my_var+' over next time step\nfor validation period 2')
#ax.text(.05,.9,'Mean Squared Error: {:.4e}'.format(tnd2_mse),transform=ax.transAxes)
#plt.savefig('../../regression_plots/'+exp_name+'_predictedVtruth_tnd2.png')
#
#
##------------------------------------------------------
## de-normalise predicted values and plot against truth
##------------------------------------------------------
#
#denorm_lr_predicted_tnd1 = lr_predicted_tnd1*tnd_outputs_tr_std+tnd_outputs_tr_mean
#fig, ax = plt.subplots(figsize=(10,8))
#ax.scatter(tnd_outputs_te1, denorm_lr_predicted_tnd1, edgecolors=(0, 0, 0))
#ax.plot([tnd_outputs_te1.min(), tnd_outputs_te1.max()], [tnd_outputs_te1.min(), tnd_outputs_te1.max()], 'k--', lw=1)
#ax.set_xlabel('Truth')
#ax.set_ylabel('Predicted')
#ax.set_title('Predicted values against GCM values\nnetwork tuned to predict change in '+my_var+' over next time step\nfor validation period 1')
#ax.text(.05,.9,'Mean Squared Error: {:.4e}'.format(tnd1_mse),transform=ax.transAxes)
#if my_var=='uVeltave':
#    plt.xlim(-0.001,0.001)
#    plt.ylim(-0.001,0.001)
#plt.savefig('../../regression_plots/'+exp_name+'_predictedVtruth_tnd1.png')
#
#denorm_lr_predicted_tnd2 = lr_predicted_tnd2*tnd_outputs_tr_std+tnd_outputs_tr_mean
#fig, ax = plt.subplots(figsize=(10,8))
#ax.scatter(tnd_outputs_te2, denorm_lr_predicted_tnd2, edgecolors=(0, 0, 0))
#ax.plot([tnd_outputs_te2.min(), tnd_outputs_te2.max()], [tnd_outputs_te2.min(), tnd_outputs_te2.max()], 'k--', lw=1)
#ax.set_xlabel('Truth')
#ax.set_ylabel('Predicted')
#ax.set_title('Predicted values against GCM values\nnetwork tuned to predict change in '+my_var+' over next time step\nfor validation period 2')
#ax.text(.05,.9,'Mean Squared Error: {:.4e}'.format(tnd2_mse),transform=ax.transAxes)
#if my_var=='uVeltave':
#    plt.xlim(-0.001,0.001)
#    plt.ylim(-0.001,0.001)
#plt.savefig('../../regression_plots/'+exp_name+'_predictedVtruth_tnd2.png')
#
#
##--------------------------------------------
## Print all scores in one place, and to file
##--------------------------------------------
#
print('Validation Period 1:')
print('next persistance r2 score ; ', pers_nxt1_r2)
print('next persistance max error ; ', pers_nxt1_maxer)
print('next persistance mean squared error ; ', pers_nxt1_mse)
print('')
print('next lr r2 score ; ', lr_nxt1_r2)
print('next lr max error ; ', lr_nxt1_maxer)
print('next lr mean squared error ; ', lr_nxt1_mse)
print('')
print('next rf r2 score ; ', rf_nxt1_r2)
print('next rf max error ; ', rf_nxt1_maxer)
print('next rf mean squared error ; ', rf_nxt1_mse)
print('')
print('next gbr r2 score ; ', gbr_nxt1_r2)
print('next gbr max error ; ', gbr_nxt1_maxer)
print('next gbr mean squared error ; ', gbr_nxt1_mse)
print('')
print('next mlp r2 score ; ', mlp_nxt1_r2)
print('next mlp max error ; ', mlp_nxt1_maxer)
print('next mlp mean squared error ; ', mlp_nxt1_mse)
print('')
#print('tend persistance r2 score ; ', r2_score(norm_tnd_outputs_te1, predict_persistance_tnd1))
#print('tend persistance max error ; ', metrics.max_error(norm_tnd_outputs_te1, predict_persistance_tnd1))
#print('tend persistance mean squared error ; ', metrics.mean_squared_error(norm_tnd_outputs_te1, predict_persistance_tnd1))
#print('')
#print('tend lr r2 score ; ', tnd1_r2)
#print('tend lr max error ; ', tnd1_maxer)
#print('tend lr mean squared error ; ', tnd1_mse)

print('Validation Period 2:')
print('next persistance r2 score ; ', pers_nxt2_r2)
print('next persistance max error ; ', pers_nxt2_maxer)
print('next persistance mean squared error ; ', pers_nxt2_mse)
print('')
print('next lr r2 score ; ', lr_nxt2_r2)
print('next lr max error ; ', lr_nxt2_maxer)
print('next lr mean squared error ; ', lr_nxt2_mse)
print('')
print('next rf r2 score ; ', rf_nxt2_r2)
print('next rf max error ; ', rf_nxt2_maxer)
print('next rf mean squared error ; ', rf_nxt2_mse)
print('')
print('next gbr r2 score ; ', gbr_nxt2_r2)
print('next gbr max error ; ', gbr_nxt2_maxer)
print('next gbr mean squared error ; ', gbr_nxt2_mse)
print('')
print('next mlp r2 score ; ', mlp_nxt2_r2)
print('next mlp max error ; ', mlp_nxt2_maxer)
print('next mlp mean squared error ; ', mlp_nxt2_mse)
print('')
#print('tend persistance r2 score ; ', r2_score(norm_tnd_outputs_te2, predict_persistance_tnd2))
#print('tend persistance max error ; ', metrics.max_error(norm_tnd_outputs_te2, predict_persistance_tnd2))
#print('tend persistance mean squared error ; ', metrics.mean_squared_error(norm_tnd_outputs_te2, predict_persistance_tnd2))
#print('')
#print('tend lr r2 score ; ', tnd2_r2)
#print('tend lr max error ; ', tnd2_maxer)
#print('tend lr mean squared error ; ', tnd2_mse)
#
file=open('../../regression_plots/'+exp_name+'.txt',"w+")
file.write('shape for inputs and outputs. Tr, test1, test2')
file.write(str(inputs_tr.shape)+'  '+str(nxt_outputs_tr.shape))
file.write(str(inputs_te1.shape)+'  '+str(nxt_outputs_te1.shape))
file.write(str(inputs_te2.shape)+'  '+str(nxt_outputs_te2.shape))
file.write('Training Scores: \n')
file.write('\n')
file.write('next pers rms score %.10f; \n' %  np.sqrt(pers_nxt_train_mse))
file.write('next lr   rms score %.10f; \n' %  np.sqrt(lr_nxt_train_mse))
file.write('next rf   rms score %.10f; \n' %  np.sqrt(rf_nxt_train_mse))
file.write('next gbr  rms score %.10f; \n' %  np.sqrt(gbr_nxt_train_mse))
file.write('next mlp  rms score %.10f; \n' %  np.sqrt(mlp_nxt_train_mse))
file.write('\n')
file.write('--------------------------------------------------------')
file.write('\n')
file.write('Validation Period 1: \n')
file.write('\n')
file.write('next persistance r2   score %.10f; \n' % pers_nxt1_r2)
file.write('next persistance max  error %.10f; \n' % pers_nxt1_maxer)
file.write('next persistance mean squared error %.10f; \n' % pers_nxt1_mse)
file.write('next persistance rms  error %.10f; \n' % np.sqrt(pers_nxt1_mse))
file.write('\n')
file.write('next lr r2   score %.10f; \n' %  lr_nxt1_r2)
file.write('next lr max  error %.10f; \n' %  lr_nxt1_maxer)
file.write('next lr mean squared error %.10f; \n' % lr_nxt1_mse)
file.write('next lr rms error %.10f; \n' % np.sqrt(lr_nxt1_mse))
file.write('\n')
file.write('next rf r2   score %.10f; \n' %  rf_nxt1_r2)
file.write('next rf max  error %.10f; \n' %  rf_nxt1_maxer)
file.write('next rf mean squared error %.10f; \n' % rf_nxt1_mse)
file.write('next rf rms  error %.10f; \n' % np.sqrt(rf_nxt1_mse))
file.write('\n')
file.write('next gbr r2   score %.10f; \n' %  gbr_nxt1_r2)
file.write('next gbr max  error %.10f; \n' %  gbr_nxt1_maxer)
file.write('next gbr mean squared error %.10f; \n' % gbr_nxt1_mse)
file.write('next gbr rms  error %.10f; \n' % np.sqrt(gbr_nxt1_mse))
file.write('\n')
file.write('next mlp r2   score %.10f; \n' %  mlp_nxt1_r2)
file.write('next mlp max  error %.10f; \n' %  mlp_nxt1_maxer)
file.write('next mlp mean squared error %.10f; \n' % mlp_nxt1_mse)
file.write('next mlp rms  error %.10f; \n' % np.sqrt(mlp_nxt1_mse))
file.write('\n')
#file.write('tend persistance r2 score %.10f; \n' % pers_tnd1_r2)
#file.write('tend persistance max error %.10f; \n' % pers_tnd1_maxer)
#file.write('tend persistance mean squared error %.10f; \n' % pers_tnd1_mse)
#file.write('tend persistance rms error %.10f; \n' % np.sqrt(pers_tnd1_mse))
#file.write('\n')
#file.write('tend lr r2 score %.10f; \n' % tnd1_r2)
#file.write('tend lr max error %.10f; \n' % tnd1_maxer)
#file.write('tend lr mean squared error %.10f; \n' % tnd1_mse)
#file.write('tend lr rms error %.10f; \n' % np.sqrt(tnd1_mse))
#file.write('\n')
file.write('--------------------------------------------------------')
file.write('\n')
file.write('Validation Period 2:\n')
file.write('\n')
file.write('next persistance r2   score %.10f; \n' % pers_nxt2_r2)
file.write('next persistance max  error %.10f; \n' % pers_nxt2_maxer)
file.write('next persistance mean squared error %.10f; \n' % pers_nxt2_mse)
file.write('next persistance rms  error %.10f; \n' % np.sqrt(pers_nxt2_mse))
file.write('\n')
file.write('next lr r2   score %.10f; \n' %  lr_nxt2_r2)
file.write('next lr max  error %.10f; \n' %  lr_nxt2_maxer)
file.write('next lr mean squared error %.10f; \n' % lr_nxt2_mse)
file.write('next lr rms  error %.10f; \n' % np.sqrt(lr_nxt2_mse))
file.write('\n')
file.write('next rf r2   score %.10f; \n' %  rf_nxt2_r2)
file.write('next rf max  error %.10f; \n' %  rf_nxt2_maxer)
file.write('next rf mean squared error %.10f; \n' % rf_nxt2_mse)
file.write('next rf rms  error %.10f; \n' % np.sqrt(rf_nxt2_mse))
file.write('\n')
file.write('next gbr r2   score %.10f; \n' %  gbr_nxt2_r2)
file.write('next gbr max  error %.10f; \n' %  gbr_nxt2_maxer)
file.write('next gbr mean squared error %.10f; \n' % gbr_nxt2_mse)
file.write('next gbr rms  error %.10f; \n' % np.sqrt(gbr_nxt2_mse))
file.write('\n')
file.write('next mlp r2   score %.10f; \n' %  mlp_nxt2_r2)
file.write('next mlp max  error %.10f; \n' %  mlp_nxt2_maxer)
file.write('next mlp mean squared error %.10f; \n' % mlp_nxt2_mse)
file.write('next mlp rms  error %.10f; \n' % np.sqrt(mlp_nxt2_mse))
file.write('\n')
#file.write('tend persistance r2 score %.10f; \n' % pers_tnd2_r2)
#file.write('tend persistance max error %.10f; \n' % pers_tnd2_maxer)
#file.write('tend persistance mean squared error %.10f; \n' % pers_tnd2_mse)
#file.write('tend persistance rms error %.10f; \n' % np.sqrt(pers_tnd2_mse))
#file.write('\n')
#file.write('tend lr r2 score %.10f; \n' % tnd2_r2)
#file.write('tend lr max error %.10f; \n' % tnd2_maxer)
#file.write('tend lr mean squared error %.10f; \n' % tnd2_mse)
#file.write('tend lr rms error %.10f; \n' % np.sqrt(tnd2_mse))
file.close() 
