#!/usr/bin/env python
# coding: utf-8

#----------------------------
print('Import neccessary packages')
#----------------------------
import ReadTestTrain_MITGCM

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

#------------------
# Read in the data
#------------------

DIR = '/data/hpcdata/users/racfur/MITGCM_OUTPUT/'
exp = '20000yr_Windx1.00_mm_diag/'
MITGCM_filename=DIR+exp+'cat_tave_5000yrs_SelectedVars.nc'
print(MITGCM_filename)

exp_name, xy_pos, norm_inputs_tr, norm_inputs_te, norm_outputs_tr, norm_outputs_te = ReadTestTrain_MITGCM.ReadData_MITGCM(MITGCM_filename, 0.7, dimension=2, lat=False, dep=False, current=False, sal=False, eta=False, poly_degree=1)

#-------------------------------------------------------------------------------------------------------------
print('Plot temp at time t against outputs (temp at time t+1), to see if variance changes with input values') 
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

plt.suptitle('Inputs against Outputs, i.e. Temp at t, against Temp at t+1', x=0.5, y=0.94, fontsize=15)

plt.savefig('../../RegressionOutputs/PLOTS/'+exp_name+'_InputsvsOutputs', bbox_inches = 'tight', pad_inches = 0.1)

#---------------------------------------------------------
# First calculate 'persistance' score, to give a baseline
#---------------------------------------------------------
print('calc persistance')

# For training data
predict_persistance_tr = np.zeros(norm_outputs_tr.shape)
pers_tr_mse = metrics.mean_squared_error(norm_outputs_tr, predict_persistance_tr)
# For validation data
predict_persistance_te = np.zeros(norm_outputs_te.shape)  
pers_te_mse = metrics.mean_squared_error(norm_outputs_te, predict_persistance_te)

##-------------------------------------------------------------------------------------
## Set up a model in scikitlearn to predict deltaT 
## Run ridge regression with interaction between terms, tuning alpha through cross val
##-------------------------------------------------------------------------------------

alpha_s = [0.00, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10, 30]
parameters = {'alpha': alpha_s}
n_folds=3

lr = linear_model.Ridge(fit_intercept=True)

print('set up regressor')
lr_cv = GridSearchCV(lr, param_grid=parameters, cv=n_folds, scoring='neg_mean_squared_error', refit=True)

print('fit the model')
lr.fit(norm_inputs_tr, norm_outputs_tr)
lr.get_params()

print('pickle it')
pkl_filename = '/data/hpcdata/users/racfur/DynamicPrediction/RegressionOutputs/MODELS/pickle_lr_'+exp_name+'.pkl'
with open(pkl_filename, 'wb') as pckl_file:
    pickle.dump(lr, pckl_file)

print('predict values and calc error')
lr_predicted_tr = lr.predict(norm_inputs_tr)
lr_tr_mse = metrics.mean_squared_error(norm_outputs_tr, lr_predicted_tr)

lr_predicted_te = lr.predict(norm_inputs_te)
lr_te_mse = metrics.mean_squared_error(norm_outputs_te, lr_predicted_te)


#-----------------------------------------------------------------
# Set up linear regression model in PyTorch (so we can use GPUs!)
#-----------------------------------------------------------------
#################################
#print('Store data as Dataset') #
#################################
#
#class MITGCM_Dataset(data.Dataset):
#    """Dataset of MITGCM outputs"""
#       
#    def __init__(self, ds_inputs, ds_outputs):
#        """
#        Args:
#           The arrays containing the training data
#        """
#        self.ds_inputs = ds_inputs
#        self.ds_outputs = ds_outputs
#
#    def __getitem__(self, index):
#
#        sample_input = self.ds_inputs[index,:]
#        sample_output = self.ds_outputs[index]
#
#        return (sample_input, sample_output)
#
#    def __len__(self):
#        return self.ds_inputs.shape[0]
#
#
## Instantiate the dataset
#Train_Dataset = MITGCM_Dataset(norm_inputs_tr, norm_outputs_tr)
#Test_Dataset  = MITGCM_Dataset(norm_inputs_te, norm_outputs_te)
#
#print('2')
#os.system('nvidia-smi')
#
#class linearRegression(torch.nn.Module):
#    def __init__(self, inputSize, outputSize):
#        super(linearRegression, self).__init__()
#        self.linear = torch.nn.Linear(inputSize, outputSize)
#
#    def forward(self, x):
#        out = self.linear(x)
#        return out
#
#inputDim = norm_inputs_tr.shape[1]         # takes variable 'x' 
#outputDim = norm_outputs_tr.shape[1]       # takes variable 'y'
#print('inputDim ; '+str(inputDim))
#print('outputDim ; '+str(outputDim))
#learningRate = 0.01 
#epochs = 2000
#
#print('3')
#os.system('nvidia-smi')
#
#model = linearRegression(inputDim, outputDim)
###### For GPU #######
#if torch.cuda.is_available():
#    model.cuda()
#print('4')
#os.system('nvidia-smi')
#
#criterion = torch.nn.MSELoss() 
#
#[w, b] = model.parameters()
#
#def get_param_values():
#   return w.data[0][0], b.data[0]
#
#alpha=0.1
#optimizer = torch.optim.Adam(model.parameters(), lr=learningRate, weight_decay=alpha)
#
#
#print('5')
#os.system('nvidia-smi')
#
#batch_size=64
#
#train_loader = torch.utils.data.DataLoader(Train_Dataset, batch_size=batch_size, num_workers=16)
#val_loader   = torch.utils.data.DataLoader(Test_Dataset,  batch_size=batch_size, num_workers=16)
#
## Train the model:
#for epoch in range(epochs):
#
#    for inputs, outputs in train_loader:
#    
#        # Converting inputs and labels to Variable
#        if torch.cuda.is_available():
#            inputs = Variable(inputs.cuda().float())
#            outputs = Variable(outputs.cuda().float())
#        else:
#            inputs = Variable(inputs.float())
#            outputs = Variable(outputs.float())
#        
#        # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
#        optimizer.zero_grad()
#    
#        # get prediction from the model, given the inputs
#        lr_predicted_tr = model(inputs)
#    
#        # get loss for the predicted output
#        loss = criterion(lr_predicted_tr, outputs)
#        # get gradients w.r.t to parameters
#        loss.backward()
#    
#        # update parameters
#        optimizer.step()
#    
#    print('epoch {}, loss {}'.format(epoch, loss.item()))
#
#
##Test the model:
#with torch.no_grad(): # we don't need gradients in the testing phase
#    if torch.cuda.is_available():
#        lr_predicted_te = model(Variable(torch.from_numpy(norm_inputs_te).cuda().float())).cpu().data.numpy()
#    else:
#        lr_predicted_te = model(Variable(torch.from_numpy(norm_inputs_te).float())).data.numpy()
#
#lr_predicted_tr = lr_predicted_tr.cpu()
#lr_predicted_tr = lr_predicted_tr.detach().numpy()
#
# -------------------------------------
print('Calculate some stats')
lr_tr_mse = metrics.mean_squared_error(norm_outputs_tr, lr_predicted_tr)
lr_te_mse = metrics.mean_squared_error(norm_outputs_te, lr_predicted_te)


#------------------------------------------
# Plot normalised prediction against truth
#------------------------------------------
bottom = min(min(norm_outputs_tr), min(lr_predicted_tr), min(norm_outputs_te), min(lr_predicted_te))
top    = max(max(norm_outputs_tr), max(lr_predicted_tr), max(norm_outputs_te), max(lr_predicted_te))
bottom = bottom - 0.1*abs(top)
top    = top + 0.1*abs(top)

fig = plt.figure(figsize=(20,9.4))
ax1 = fig.add_subplot(121)
ax1.scatter(norm_outputs_tr, lr_predicted_tr, edgecolors=(0, 0, 0))
ax1.plot([bottom, top], [bottom, top], 'k--', lw=1)
ax1.set_xlabel('Truth')
ax1.set_ylabel('Predicted')
ax1.set_title('Train')
ax1.text(.05,.95,'Mean Squared Error: {:.4e}'.format(lr_tr_mse),transform=ax1.transAxes)
ax1.set_xlim(bottom, top)
ax1.set_ylim(bottom, top)

ax2 = fig.add_subplot(122)
ax2.scatter(norm_outputs_te, lr_predicted_te, edgecolors=(0, 0, 0))
ax2.plot([bottom, top], [bottom, top], 'k--', lw=1)
ax2.set_xlabel('Truth')
ax2.set_ylabel('Predicted')
ax2.set_title('Test')
ax2.set_xlim(bottom, top)
ax2.set_ylim(bottom, top)
ax2.text(.05,.95,'Mean Squared Error: {:.4e}'.format(lr_te_mse),transform=ax2.transAxes)

plt.suptitle('Noramlised temperature predicted by Linear Regression model against normalised GCM data (truth)', x=0.5, y=0.94, fontsize=15)

plt.savefig('../../RegressionOutputs/PLOTS/'+exp_name+'_norm_predictedVtruth.png', bbox_inches = 'tight', pad_inches = 0.1)


#------------------------------------------------------
# de-normalise predicted values and plot against truth
#------------------------------------------------------
#Read in mean and std to normalise inputs - should move this outside of the loop!
print('read in info to normalise data')
norm_file=open('/data/hpcdata/users/racfur/DynamicPrediction/RegressionOutputs/STATS/normalising_parameters_'+exp_name+'.txt',"r")
count = len(norm_file.readlines(  ))
#print('count ; '+str(count))
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

denorm_lr_predicted_tr = lr_predicted_tr*output_std+output_mean
denorm_lr_predicted_te = lr_predicted_te*output_std+output_mean
denorm_outputs_tr = norm_outputs_tr*output_std+output_mean
denorm_outputs_te = norm_outputs_te*output_std+output_mean

bottom = min(min(denorm_outputs_tr), min(denorm_lr_predicted_tr), min(denorm_outputs_te), min(denorm_lr_predicted_te))
top    = max(max(denorm_outputs_tr), max(denorm_lr_predicted_tr), max(denorm_outputs_te), max(denorm_lr_predicted_te))
bottom = bottom - 0.1*abs(top)
top    = top + 0.1*abs(top)

fig = plt.figure(figsize=(20,9.4))

ax1 = fig.add_subplot(121)
ax1.scatter(denorm_outputs_tr, denorm_lr_predicted_tr, edgecolors=(0, 0, 0))
ax1.plot([bottom, top], [bottom, top], 'k--', lw=1)
ax1.set_xlabel('Truth')
ax1.set_ylabel('Predicted')
ax1.set_title('Train')
ax1.text(.05,.95,'Mean Squared Error: {:.4e}'.format(lr_tr_mse),transform=ax1.transAxes)
ax1.set_xlim(bottom, top)
ax1.set_ylim(bottom, top)

ax2 = fig.add_subplot(122)
ax2.scatter(denorm_outputs_te, denorm_lr_predicted_te, edgecolors=(0, 0, 0))
ax2.plot([bottom, top], [bottom, top], 'k--', lw=1)
ax2.set_xlabel('Truth')
ax2.set_ylabel('Predicted')
ax2.set_title('Test')
ax2.set_xlim(bottom, top)
ax2.set_ylim(bottom, top)
ax2.text(.05,.95,'Mean Squared Error: {:.4e}'.format(lr_te_mse),transform=ax2.transAxes)

plt.suptitle('Temperature predicted by Linear Regression model against GCM data (truth)', x=0.5, y=0.94, fontsize=15)

plt.savefig('../../RegressionOutputs/PLOTS/'+exp_name+'_predictedVtruth.png', bbox_inches = 'tight', pad_inches = 0.1)

#--------------------------------------------
# Print all scores to file
#--------------------------------------------
stats_file=open('/data/hpcdata/users/racfur/DynamicPrediction/RegressionOutputs/STATS/'+exp_name+'.txt',"w+")
stats_file.write('shape for inputs and outputs: tr; te \n')
stats_file.write(str(norm_inputs_tr.shape)+'  '+str(norm_outputs_tr.shape)+'\n')
stats_file.write(str(norm_inputs_te.shape)+'  '+str(norm_outputs_te.shape)+'\n')

stats_file.write('\n')
stats_file.write('--------------------------------------------------------')
stats_file.write('\n')
stats_file.write('Training Scores: \n')
stats_file.write('\n')
stats_file.write(' persistance rms score %.10f; \n' % np.sqrt(pers_tr_mse))
stats_file.write('          lr rms score %.10f; \n' % np.sqrt(lr_tr_mse))
stats_file.write('\n')
stats_file.write('--------------------------------------------------------')
stats_file.write('\n')
stats_file.write('Validation Scores: \n')
stats_file.write(' persistance rms score %.10f; \n' % np.sqrt(pers_te_mse))
stats_file.write('    lr test1 rms error %.10f; \n' % np.sqrt(lr_te_mse))
stats_file.write('\n')
stats_file.close() 
