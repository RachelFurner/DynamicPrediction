#!/usr/bin/env python
# coding: utf-8

#----------------------------
print('Import neccessary packages')
#----------------------------
import sys
sys.path.append('/data/hpcdata/users/racfur/DynamicPrediction/code_git/Tools/')
import RF_routines as RF

import numpy as np
import os
import xarray as xr
import matplotlib.pyplot as plt
import pickle

from sklearn.preprocessing import PolynomialFeatures
import sklearn.metrics as metrics

from torch.utils import data
import torch.nn as nn
from torch.autograd import Variable
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using device:', device)
print()

torch.cuda.empty_cache()

#-------------------- -----------------
# Manually set variables for this run
#--------------------------------------
run_vars={'dimension':2, 'lat':True, 'dep':True, 'current':True, 'sal':True, 'eta':True, 'poly_degree':1}
model_type = 'nn'

DIR = '/data/hpcdata/users/racfur/MITGCM_OUTPUT/20000yr_Windx1.00_mm_diag/'
MITGCM_filename=DIR+'cat_tave_5000yrs_SelectedVars.nc'

read_data = False

batch_size=32
learningRate = 0.001 
epochs = 200
criterion = torch.nn.MSELoss()
#--------------------------------
# Calculate some other variables 
#--------------------------------
exp_name = RF.create_expname(model_type, run_vars)

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
   norm_inputs_tr, norm_inputs_te, norm_outputs_tr, norm_outputs_te = RF.ReadMITGCM(MITGCM_filename, 0.7, exp_name, run_vars)
else:
   array_file = '/data/hpcdata/users/racfur/DynamicPrediction/DATASETS/'+exp_name+'_InputsOutputs.npz'
   norm_inputs_tr, norm_inputs_te, norm_outputs_tr, norm_outputs_te = np.load(array_file).values()

#----------------------------------------------------------------------------------------------------------------
# Calculate 'persistance' score, to give a baseline - this is just zero everywhere as we're predicting the trend
#----------------------------------------------------------------------------------------------------------------
print('calc persistance')

# For training data
predict_persistance_tr = np.zeros(norm_outputs_tr.shape)
pers_tr_mse = metrics.mean_squared_error(norm_outputs_tr, predict_persistance_tr)
# For validation data
predict_persistance_te = np.zeros(norm_outputs_te.shape)  
pers_te_mse = metrics.mean_squared_error(norm_outputs_te, predict_persistance_te)

#-----------------------------------------------------------------
# Set up regression model in PyTorch (so we can use GPUs!)
#-----------------------------------------------------------------
# Store data as Dataset'

class MITGCM_Dataset(data.Dataset):
    """Dataset of MITGCM outputs"""
       
    def __init__(self, ds_inputs, ds_outputs):
        """
        Args:
           The arrays containing the training data
        """
        self.ds_inputs = ds_inputs
        self.ds_outputs = ds_outputs

    def __getitem__(self, index):

        sample_input = self.ds_inputs[index,:]
        sample_output = self.ds_outputs[index]

        return (sample_input, sample_output)

    def __len__(self):
        return self.ds_inputs.shape[0]


# Instantiate the dataset
Train_Dataset = MITGCM_Dataset(norm_inputs_tr, norm_outputs_tr)
Test_Dataset  = MITGCM_Dataset(norm_inputs_te, norm_outputs_te)

train_loader = torch.utils.data.DataLoader(Train_Dataset, batch_size=batch_size, num_workers=16)
val_loader   = torch.utils.data.DataLoader(Test_Dataset,  batch_size=batch_size, num_workers=16)

inputSize = norm_inputs_tr.shape[1]
outputSize = norm_outputs_tr.shape[1]

h = nn.Sequential( nn.Linear( inputSize, outputSize) )

# parallelise and send to GPU
if torch.cuda.device_count() > 1:
   print("Let's use", torch.cuda.device_count(), "GPUs!")
   h = nn.DataParallel(h)
h.to(device)

optimizer = torch.optim.Adam(h.parameters(), lr=learningRate)

train_loss_batch = []
train_loss_epoch = []
val_loss_epoch = []

# Train the model:
for epoch in range(epochs):
    train_loss_temp = 0.0
    val_loss_temp = 0.0
    for inputs, outputs in train_loader:
    
        # Converting inputs and labels to Variable and send to GPU if available
        if torch.cuda.is_available():
            inputs = Variable(inputs.cuda().float())
            outputs = Variable(outputs.cuda().float())
        else:
            inputs = Variable(inputs.float())
            outputs = Variable(outputs.float())
        
        h.train(True)

        # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
        optimizer.zero_grad()
    
        # get prediction from the model, given the inputs
        nn_predicted_tr = h(inputs)
    
        # get loss for the predicted output
        loss = criterion(nn_predicted_tr, outputs)
        # get gradients w.r.t to parameters
        loss.backward()
    
        # update parameters
        optimizer.step()
        
        train_loss_batch.append(loss.item())
        train_loss_temp += loss.item()
    
    for inputs, outputs in val_loader:
    
        # Converting inputs and labels to Variable and send to GPU if available
        if torch.cuda.is_available():
            inputs = Variable(inputs.cuda().float())
            outputs = Variable(outputs.cuda().float())
        else:
            inputs = Variable(inputs.float())
            outputs = Variable(outputs.float())
        
        h.train(False)

        # get prediction from the model, given the inputs
        nn_predicted_tr = h(inputs)
    
        # get loss for the predicted output
        loss = criterion(nn_predicted_tr, outputs)

        val_loss_temp += loss.item()     

    train_loss_epoch.append(train_loss_temp / norm_inputs_tr.shape[0] )
    val_loss_epoch.append(val_loss_temp / norm_inputs_te.shape[0] )
    print('Epoch {}, Training Loss: {:.8f}'.format(epoch, train_loss_epoch[-1]))
    print('epoch {}, Validation Loss: {:.8f}'.format(epoch, val_loss_epoch[-1]))    


nn_predicted_tr = nn_predicted_tr.cpu()
nn_predicted_tr = nn_predicted_tr.detach().numpy()

# -------------------------------------
print('Calculate some stats')
nn_tr_mse = metrics.mean_squared_error(norm_outputs_tr, nn_predicted_tr)
nn_te_mse = metrics.mean_squared_error(norm_outputs_te, nn_predicted_te)


#------------------------------------------
# Plot normalised prediction against truth
#------------------------------------------
bottom = min(min(norm_outputs_tr), min(nn_predicted_tr), min(norm_outputs_te), min(nn_predicted_te))
top    = max(max(norm_outputs_tr), max(nn_predicted_tr), max(norm_outputs_te), max(nn_predicted_te))
bottom = bottom - 0.1*abs(top)
top    = top + 0.1*abs(top)

fig = plt.figure(figsize=(20,9.4))
ax1 = fig.add_subplot(121)
ax1.scatter(norm_outputs_tr, nn_predicted_tr, edgecolors=(0, 0, 0))
ax1.plot([bottom, top], [bottom, top], 'k--', lw=1)
ax1.set_xlabel('Truth')
ax1.set_ylabel('Predicted')
ax1.set_title('Train')
ax1.text(.05,.95,'Mean Squared Error: {:.4e}'.format(nn_tr_mse),transform=ax1.transAxes)
ax1.set_xlim(bottom, top)
ax1.set_ylim(bottom, top)

ax2 = fig.add_subplot(122)
ax2.scatter(norm_outputs_te, nn_predicted_te, edgecolors=(0, 0, 0))
ax2.plot([bottom, top], [bottom, top], 'k--', lw=1)
ax2.set_xlabel('Truth')
ax2.set_ylabel('Predicted')
ax2.set_title('Test')
ax2.set_xlim(bottom, top)
ax2.set_ylim(bottom, top)
ax2.text(.05,.95,'Mean Squared Error: {:.4e}'.format(nn_te_mse),transform=ax2.transAxes)

plt.suptitle('Noramlised temperature predicted by Linear Regression model against normalised GCM data (truth)', x=0.5, y=0.94, fontsize=15)

plt.savefig('../../NNOutputs/PLOTS/'+exp_name+'_norm_predictedVtruth.png', bbox_inches = 'tight', pad_inches = 0.1)


#------------------------------------------------------
# de-normalise predicted values and plot against truth
#------------------------------------------------------
#Read in mean and std to normalise inputs - should move this outside of the loop!
print('read in info to normalise data')
norm_file=open('/data/hpcdata/users/racfur/DynamicPrediction/NNOutputs/STATS/normalising_parameters_'+exp_name+'.txt',"r")
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

denorm_nn_predicted_tr = nn_predicted_tr*output_std+output_mean
denorm_nn_predicted_te = nn_predicted_te*output_std+output_mean
denorm_outputs_tr = norm_outputs_tr*output_std+output_mean
denorm_outputs_te = norm_outputs_te*output_std+output_mean

bottom = min(min(denorm_outputs_tr), min(denorm_nn_predicted_tr), min(denorm_outputs_te), min(denorm_nn_predicted_te))
top    = max(max(denorm_outputs_tr), max(denorm_nn_predicted_tr), max(denorm_outputs_te), max(denorm_nn_predicted_te))
bottom = bottom - 0.1*abs(top)
top    = top + 0.1*abs(top)

fig = plt.figure(figsize=(20,9.4))

ax1 = fig.add_subplot(121)
ax1.scatter(denorm_outputs_tr, denorm_nn_predicted_tr, edgecolors=(0, 0, 0))
ax1.plot([bottom, top], [bottom, top], 'k--', lw=1)
ax1.set_xlabel('Truth')
ax1.set_ylabel('Predicted')
ax1.set_title('Train')
ax1.text(.05,.95,'Mean Squared Error: {:.4e}'.format(nn_tr_mse),transform=ax1.transAxes)
ax1.set_xlim(bottom, top)
ax1.set_ylim(bottom, top)

ax2 = fig.add_subplot(122)
ax2.scatter(denorm_outputs_te, denorm_nn_predicted_te, edgecolors=(0, 0, 0))
ax2.plot([bottom, top], [bottom, top], 'k--', lw=1)
ax2.set_xlabel('Truth')
ax2.set_ylabel('Predicted')
ax2.set_title('Test')
ax2.set_xlim(bottom, top)
ax2.set_ylim(bottom, top)
ax2.text(.05,.95,'Mean Squared Error: {:.4e}'.format(nn_te_mse),transform=ax2.transAxes)

plt.suptitle('Temperature predicted by Linear Regression model against GCM data (truth)', x=0.5, y=0.94, fontsize=15)

plt.savefig('../../NNOutputs/PLOTS/'+exp_name+'_predictedVtruth.png', bbox_inches = 'tight', pad_inches = 0.1)

#--------------------------------------------
# Print all scores to file
#--------------------------------------------
stats_file=open('/data/hpcdata/users/racfur/DynamicPrediction/NNOutputs/STATS/'+exp_name+'.txt',"w+")
stats_file.write('shape for inputs and outputs: tr; te \n')
stats_file.write(str(norm_inputs_tr.shape)+'  '+str(norm_outputs_tr.shape)+'\n')
stats_file.write(str(norm_inputs_te.shape)+'  '+str(norm_outputs_te.shape)+'\n')

stats_file.write('\n')
stats_file.write('--------------------------------------------------------')
stats_file.write('\n')
stats_file.write('Training Scores: \n')
stats_file.write('\n')
stats_file.write(' persistance rms score %.10f; \n' % np.sqrt(pers_tr_mse))
stats_file.write('          nn rms score %.10f; \n' % np.sqrt(nn_tr_mse))
stats_file.write('\n')
stats_file.write('--------------------------------------------------------')
stats_file.write('\n')
stats_file.write('Validation Scores: \n')
stats_file.write(' persistance rms score %.10f; \n' % np.sqrt(pers_te_mse))
stats_file.write('    nn test1 rms error %.10f; \n' % np.sqrt(nn_te_mse))
stats_file.write('\n')
stats_file.close() 
