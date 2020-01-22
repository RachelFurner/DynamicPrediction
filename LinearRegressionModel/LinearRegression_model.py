#!/usr/bin/env python
# coding: utf-8

#----------------------------
print('Import neccessary packages')
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
print('Set variables')
#---------------

StepSize = 1 # how many output steps (months!) to predict over

halo_size = 1
halo_list = (range(-halo_size, halo_size+1))
## Calculate x,y position in each feature list to use as 'now' value
## When using 3-d inputs
if halo_size == 1:
   xy = 13
elif halo_size == 2:
   xy == 62
else:
   print('error in halo size....')
## When using 2-d inputs
#xy=2*(halo_size^2+halo_size) # if using a 2-d halo in x and y....if using a 3-d halo need to use above.

# ratio of test to train data to randomly split
data_end_index = 2000 * 12   # look at first 2000 years only - while model is still dynamically active. Ignore rest for now.
split_ratio = .7             # ratio of test data to training data

temp = 'Ttave'

#exp_name='T_3dLatDepVarsINT_halo'+str(halo_size)+'_pred'+str(StepSize)
exp_name='T_3dLatDepVarsINT_halo'+str(halo_size)+'_pred'+str(StepSize)


#------------------
# Read in the data
#------------------

DIR = '/data/hpcdata/users/racfur/MITGCM_OUTPUT/'
exp = '20000yr_Windx1.00_mm_diag/'
filename=DIR+exp+'cat_tave_5000yrs_SelectedVars.nc'
print(filename)
ds   = xr.open_dataset(filename)

da_T=ds['Ttave'].values
da_S=ds['Stave'].values
da_U=ds['uVeltave'].values
da_V=ds['vVeltave'].values
da_Eta=ds['ETAtave'].values
da_lat=ds['Y'].values
da_depth=ds['Z'].values

#------------------------------------------------------------------------------------------
print('Read in data as one dataset, which we randomly split into train and test later')
#------------------------------------------------------------------------------------------
inputs = []
outputs = []

#for z in range(2,38,15):
#    for x in range(2,7,3):
#        for y in range(2,74,20):
#            for time in range(0, data_end_index, 1000):  # look at first 
for z in range(2,38,2):
    for x in range(2,7,2):
        for y in range(2,74,5):
            for time in range(0, data_end_index, 100):  # look at first 2000 years only - ignore region when model is at equilibrium

                input_temp = []
                #[input_temp.append(da_T[time,z,y+y_offset,x+x_offset]) for x_offset in halo_list for y_offset in halo_list]
                [input_temp.append(da_T[time,z+z_offset,y+y_offset,x+x_offset]) for x_offset in halo_list for y_offset in halo_list for z_offset in halo_list]
                [input_temp.append(da_S[time,z+z_offset,y+y_offset,x+x_offset]) for x_offset in halo_list for y_offset in halo_list for z_offset in halo_list]
                [input_temp.append(da_U[time,z+z_offset,y+y_offset,x+x_offset]) for x_offset in halo_list for y_offset in halo_list for z_offset in halo_list]
                [input_temp.append(da_V[time,z+z_offset,y+y_offset,x+x_offset]) for x_offset in halo_list for y_offset in halo_list for z_offset in halo_list]
                [input_temp.append(da_Eta[time,y+y_offset,x+x_offset]) for x_offset in halo_list for y_offset in halo_list]
                input_temp.append(da_lat[y])
                input_temp.append(da_depth[z])

                inputs.append(input_temp)
                outputs.append([da_T[time+StepSize,z,y,x]])
               
inputs=np.asarray(inputs)
outputs=np.asarray(outputs)
print('now save them')
filename = '/data/hpcdata/users/racfur/DynamicPrediction/RegressionOutputs/DATASETS/'+exp_name+'_InputsOutputs.npz'
np.savez(filename, inputs,outputs)
print('saved')
print('open array from file')
inputs, outputs = np.load(filename).values()

#randomise the sample order, and split into test and train data
split=int(split_ratio * inputs.shape[0])

np.random.seed(5)
ordering = np.random.permutation(inputs.shape[0])

inputs_tr = inputs[ordering][:split]
outputs_tr = outputs[ordering][:split]
inputs_te1 = inputs[ordering][split:]
outputs_te1 = outputs[ordering][split:]

print('shape for inputs and outputs: full; tr; te1')
print(inputs.shape, outputs.shape)
print(inputs_tr.shape, outputs_tr.shape)
print(inputs_te1.shape, outputs_te1.shape)


#----------------------------------------------------------------------------------
print('Plot inputs against outputs, to see if variance changes with input values') 
#----------------------------------------------------------------------------------
fig = plt.figure(figsize=(21, 7))

ax1 = fig.add_subplot(131)
ax1.scatter(inputs[:,xy], outputs, edgecolors=(0, 0, 0))
ax1.set_xlabel('Inputs (t)')
ax1.set_ylabel('Outputs (t+1)')
ax1.set_title('Full dataset')

ax2 = fig.add_subplot(132)
ax2.scatter(inputs_tr[:,xy], outputs_tr, edgecolors=(0, 0, 0))
ax2.set_xlabel('Inputs (t)')
ax2.set_ylabel('Outputs (t+1)')
ax2.set_title('Training Data')

ax3 = fig.add_subplot(133)
ax3.scatter(inputs_te1[:,xy], outputs_te1, edgecolors=(0, 0, 0))
ax3.set_xlabel('Inputs (t)')
ax3.set_ylabel('Outputs (t+1)')
ax3.set_title('Validation Data')

plt.suptitle('Inputs against Outputs, i.e. Temp at t, against Temp at t+1', x=0.5, y=0.94, fontsize=15)

plt.savefig('../../RegressionOutputs/PLOTS/'+exp_name+'_InputsvsOutputs', bbox_inches = 'tight', pad_inches = 0.1)

#----------------------------------------------
# Normalise Data (based on training data only)
#----------------------------------------------
print('normalise data')
def normalise_data(train,test1):
    train_mean, train_std = np.mean(train), np.std(train)
    norm_train = (train - train_mean) / train_std
    norm_test1 = (test1 - train_mean) / train_std
    return norm_train, norm_test1, train_mean, train_std

## normalise inputs
inputs_mean = np.zeros(inputs_tr.shape[1])
inputs_std  = np.zeros(inputs_tr.shape[1])
norm_inputs_tr   = np.zeros(inputs_tr.shape)
norm_inputs_te1   = np.zeros(inputs_te1.shape)
for i in range(inputs_tr.shape[1]):  #loop over each feature, normalising individually
    norm_inputs_tr[:, i], norm_inputs_te1[:, i], inputs_mean[i], inputs_std[i] = normalise_data(inputs_tr[:, i], inputs_te1[:,i])

## normalise outputs
norm_outputs_tr, norm_outputs_te1, outputs_mean, outputs_std = normalise_data(outputs_tr[:], outputs_te1[:])

## save mean and std to un-normalise later (in a different script)
norm_file=open('/data/hpcdata/users/racfur/DynamicPrediction/RegressionOutputs/STATS/normalising_parameters_'+exp_name+'.txt','w')
for i in range(inputs_tr.shape[1]):  #loop over each feature, normalising individually
    norm_file.write('inputs_mean['+str(i)+']\n')
    norm_file.write(str(inputs_mean[i])+'\n')
    norm_file.write('inputs_std['+str(i)+']\n')
    norm_file.write(str(inputs_std[i])+'\n')
norm_file.write('outputs_mean\n')
norm_file.write(str(outputs_mean)+'\n')
norm_file.write('outputs_std\n')
norm_file.write(str(outputs_std)+'\n')
norm_file.close()

#---------------------------------------------------------------------
#---------------------------------------------------------------------
# Set up a model to directly predict variable value at next time step
#---------------------------------------------------------------------
#---------------------------------------------------------------------

#---------------------------------------------------------
# First calculate 'persistance' score, to give a baseline
#---------------------------------------------------------
print('calc persistance')

# For training data
predict_persistance_tr = norm_inputs_tr[:,xy]
pers_tr_mse = metrics.mean_squared_error(norm_outputs_tr, predict_persistance_tr)
# For validation data
predict_persistance_te1 = norm_inputs_te1[:,xy]
pers_te1_mse = metrics.mean_squared_error(norm_outputs_te1, predict_persistance_te1)

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
polynomial_features = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)

lrpipe = Pipeline([("polynomial_features", polynomial_features), ("linear_regression", lr_cv)])
#lrpipe = lr_cv # don't include polynomial features

print('fit the model')
lrpipe.fit(norm_inputs_tr, norm_outputs_tr )  # train to evaluate the value at the next time step...
lrpipe.get_params()

print('pickle it')
pkl_filename = '/data/hpcdata/users/racfur/DynamicPrediction/RegressionOutputs/MODELS/pickle_lr_'+exp_name+'.pkl'
with open(pkl_filename, 'wb') as file:
    pickle.dump(lrpipe, file)

print('predict values and calc error')
lr_predicted_tr = lrpipe.predict(norm_inputs_tr)
lrpipe_tr_mse = metrics.mean_squared_error(norm_outputs_tr, lr_predicted_tr)

lr_predicted_te1 = lrpipe.predict(norm_inputs_te1)
lrpipe_te1_mse = metrics.mean_squared_error(norm_outputs_te1, lr_predicted_te1)


#------------------------------------------
# Plot normalised prediction against truth
#------------------------------------------
bottom = min(min(norm_outputs_tr), min(lr_predicted_tr), min(norm_outputs_te1), min(lr_predicted_te1))
top    = max(max(norm_outputs_tr), max(lr_predicted_tr), max(norm_outputs_te1), max(lr_predicted_te1))
bottom = bottom - 0.1*abs(top)
top    = top + 0.1*abs(top)

fig = plt.figure(figsize=(20,9.4))
ax1 = fig.add_subplot(121)
ax1.scatter(norm_outputs_tr, lr_predicted_tr, edgecolors=(0, 0, 0))
ax1.plot([norm_outputs_tr.min(), norm_outputs_tr.max()], [norm_outputs_tr.min(), norm_outputs_tr.max()], 'k--', lw=1)
ax1.set_xlabel('Truth')
ax1.set_ylabel('Predicted')
ax1.set_title('Train')
ax1.text(.05,.95,'Mean Squared Error: {:.4e}'.format(lrpipe_tr_mse),transform=ax1.transAxes)
ax1.set_xlim(bottom, top)
ax1.set_ylim(bottom, top)

ax2 = fig.add_subplot(122)
ax2.scatter(norm_outputs_te1, lr_predicted_te1, edgecolors=(0, 0, 0))
ax2.plot([norm_outputs_te1.min(), norm_outputs_te1.max()], [norm_outputs_te1.min(), norm_outputs_te1.max()], 'k--', lw=1)
ax2.set_xlabel('Truth')
ax2.set_ylabel('Predicted')
ax2.set_title('Test')
ax2.set_xlim(bottom, top)
ax2.set_ylim(bottom, top)
ax2.text(.05,.95,'Mean Squared Error: {:.4e}'.format(lrpipe_te1_mse),transform=ax2.transAxes)

plt.suptitle('Noramlised temperature predicted by Linear Regression model against normalised GCM data (truth)', x=0.5, y=0.94, fontsize=15)

plt.savefig('../../RegressionOutputs/PLOTS/'+exp_name+'_norm_predictedVtruth.png', bbox_inches = 'tight', pad_inches = 0.1)


#------------------------------------------------------
# de-normalise predicted values and plot against truth
#------------------------------------------------------
denorm_lr_predicted_tr = lr_predicted_tr*outputs_std+outputs_mean
denorm_lr_predicted_te1 = lr_predicted_te1*outputs_std+outputs_mean

bottom = min(min(outputs_tr), min(denorm_lr_predicted_tr), min(outputs_te1), min(denorm_lr_predicted_te1))
top    = max(max(outputs_tr), max(denorm_lr_predicted_tr), max(outputs_te1), max(denorm_lr_predicted_te1))
bottom = bottom - 0.1*abs(top)
top    = top + 0.1*abs(top)

fig = plt.figure(figsize=(20,9.4))

ax1 = fig.add_subplot(121)
ax1.scatter(outputs_tr, denorm_lr_predicted_tr, edgecolors=(0, 0, 0))
ax1.plot([outputs_tr.min(), outputs_tr.max()], [outputs_tr.min(), outputs_tr.max()], 'k--', lw=1)
ax1.set_xlabel('Truth')
ax1.set_ylabel('Predicted')
ax1.set_title('Train')
ax1.text(.05,.95,'Mean Squared Error: {:.4e}'.format(lrpipe_tr_mse),transform=ax1.transAxes)
ax1.set_xlim(bottom, top)
ax1.set_ylim(bottom, top)

ax2 = fig.add_subplot(122)
ax2.scatter(outputs_te1, denorm_lr_predicted_te1, edgecolors=(0, 0, 0))
ax2.plot([outputs_te1.min(), outputs_te1.max()], [outputs_te1.min(), outputs_te1.max()], 'k--', lw=1)
ax2.set_xlabel('Truth')
ax2.set_ylabel('Predicted')
ax2.set_title('Test')
ax2.set_xlim(bottom, top)
ax2.set_ylim(bottom, top)
ax2.text(.05,.95,'Mean Squared Error: {:.4e}'.format(lrpipe_te1_mse),transform=ax2.transAxes)

plt.suptitle('Temperature predicted by Linear Regression model against GCM data (truth)', x=0.5, y=0.94, fontsize=15)

plt.savefig('../../RegressionOutputs/PLOTS/'+exp_name+'_predictedVtruth.png', bbox_inches = 'tight', pad_inches = 0.1)


#------------------------------------------------------------------------------------------------------------------
# plot normalised prediction-original value, against truth-original value (i.e. convert to look like tendancy plot....)
#------------------------------------------------------------------------------------------------------------------
lims = max( max(abs(outputs_tr[:,0]-inputs_tr[:,xy])), max(abs(outputs_te1[:,0]-inputs_te1[:,xy])),
            max(abs(denorm_lr_predicted_tr[:,0]-inputs_tr[:,xy])), max(abs(denorm_lr_predicted_te1[:,0]-inputs_te1[:,xy])) )
bottom = -lims - 0.1*lims
top    = lims + 0.1*lims

fig = plt.figure(figsize=(20,9.4))
ax1 = fig.add_subplot(121)
ax1.scatter(outputs_tr[:,0]-inputs_tr[:,xy], denorm_lr_predicted_tr[:,0]-inputs_tr[:,xy], edgecolors=(0, 0, 0))
ax1.set_xlabel('Truth')
ax1.set_ylabel('Predicted')
ax1.set_title('Train')
ax1.text(.05,.95,'Mean Squared Error: {:.4e}'.format(lrpipe_tr_mse),transform=ax1.transAxes)
ax1.set_xlim(bottom, top)
ax1.set_ylim(bottom, top)

ax2 = fig.add_subplot(122)
ax2.scatter(outputs_te1[:,0]-inputs_te1[:,xy], denorm_lr_predicted_te1[:,0]-inputs_te1[:,xy], edgecolors=(0, 0, 0))
ax2.set_xlabel('Truth')
ax2.set_ylabel('Predicted')
ax2.set_title('Test')
ax2.set_xlim(bottom, top)
ax2.set_ylim(bottom, top)
ax2.text(.05,.95,'Mean Squared Error: {:.4e}'.format(lrpipe_te1_mse),transform=ax2.transAxes)

plt.suptitle('Normalised Predicted tendancies against GCM tendancies', x=0.5, y=0.94, fontsize=15)
plt.savefig('/data/hpcdata/users/racfur/DynamicPrediction/RegressionOutputs/PLOTS/'+exp_name+'_norm_predictedVtruth_diffastend.png', bbox_inches = 'tight', pad_inches = 0.1)

#--------------------------------------------
# Print all scores to file
#--------------------------------------------
#
file=open('/data/hpcdata/users/racfur/DynamicPrediction/RegressionOutputs/STATS/'+exp_name+'.txt',"w+")
file.write('shape for inputs and outputs. full; tr; te1 \n')
file.write(str(inputs.shape)+'  '+str(outputs.shape)+'\n')
file.write(str(norm_inputs_tr.shape)+'  '+str(norm_outputs_tr.shape)+'\n')
file.write(str(norm_inputs_te1.shape)+'  '+str(norm_outputs_te1.shape)+'\n')
file.write('\n')
file.write('--------------------------------------------------------')
file.write('\n')
file.write('Training Scores: \n')
file.write('\n')
file.write(' persistance rms score %.10f; \n' % np.sqrt(pers_tr_mse))
file.write('          lr rms score %.10f; \n' % np.sqrt(lrpipe_tr_mse))
file.write('\n')
file.write('--------------------------------------------------------')
file.write('\n')
file.write('Validation Scores: \n')
file.write(' persistance rms score %.10f; \n' % np.sqrt(pers_te1_mse))
file.write('    lr test1 rms error %.10f; \n' % np.sqrt(lrpipe_te1_mse))
file.write('\n')
file.close() 
