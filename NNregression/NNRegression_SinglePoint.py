#!/usr/bin/env python
# coding: utf-8

#----------------------------
print('Import neccessary packages')
#----------------------------
import sys
sys.path.append('/data/hpcdata/users/racfur/DynamicPrediction/code_git/')
from Tools import CreateExpName as cn
from Tools import ReadRoutines as rr

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
run_vars={'dimension':3, 'lat':True, 'dep':True, 'current':True, 'sal':True, 'eta':True, 'poly_degree':2}
model_type = 'nn'

DIR = '/data/hpcdata/users/racfur/MITGCM_OUTPUT/20000yr_Windx1.00_mm_diag/'
MITGCM_filename=DIR+'cat_tave_5000yrs_SelectedVars.nc'

read_data = False

batch_size=64
learningRate = 0.001 
epochs = 200
criterion = torch.nn.MSELoss()
#--------------------------------
# Calculate some other variables 
#--------------------------------
exp_name = cn.create_expname(model_type, run_vars)

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
   norm_inputs_tr, norm_inputs_te, norm_outputs_tr, norm_outputs_te = rr.ReadMITGCM(MITGCM_filename, 0.7, exp_name, run_vars)
else:
   inputsoutputs_file = '/data/hpcdata/users/racfur/DynamicPrediction/INPUT_OUTPUT_ARRAYS/'+exp_name+'_InputsOutputs.npz'
   norm_inputs_tr, norm_inputs_te, norm_outputs_tr, norm_outputs_te = np.load(inputsoutputs_file).values()

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

inputFeatures = norm_inputs_tr.shape[1]
outputFeatures = norm_outputs_tr.shape[1]

h_linear = nn.Sequential( nn.Linear( inputFeatures, outputFeatures) )

# parallelise and send to GPU
#if torch.cuda.device_count() > 1:
#   print("Let's use", torch.cuda.device_count(), "GPUs!")
#   h = nn.DataParallel(h)
h_linear.to(device)

optimizer = torch.optim.Adam( h_linear.parameters(), lr=learningRate )

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
        
        h_linear.train(True)

        # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
        optimizer.zero_grad()
    
        # get prediction from the model, given the inputs
        predicted = h_linear(inputs)
    
        # get loss for the predicted output
        loss = criterion(predicted, outputs)
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
        
        h_linear.train(False)

        # get prediction from the model, given the inputs
        predicted = h_linear(inputs)
    
        # get loss for the predicted output
        loss = criterion(predicted, outputs)

        val_loss_temp += loss.item()     

    train_loss_epoch.append(train_loss_temp / norm_inputs_tr.shape[0] )
    val_loss_epoch.append(val_loss_temp / norm_inputs_te.shape[0] )
    print('Epoch {}, Training Loss: {:.8f}'.format(epoch, train_loss_epoch[-1]))
    print('epoch {}, Validation Loss: {:.8f}'.format(epoch, val_loss_epoch[-1]))    


if torch.cuda.is_available():
    tr_inputs = Variable(torch.from_numpy(norm_inputs_tr).cuda().float())
    te_inputs = Variable(torch.from_numpy(norm_inputs_te).cuda().float())
else:
    tr_inputs = Variable(torch.from_numpy(norm_inputs_tr).float())
    te_inputs = Variable(torch.from_numpy(norm_inputs_te).float())
nn_predicted_tr = h_linear(tr_inputs)
nn_predicted_te = h_linear(te_inputs)

nn_predicted_tr = nn_predicted_tr.cpu()
nn_predicted_te = nn_predicted_te.cpu()
nn_predicted_tr = nn_predicted_tr.detach().numpy()
nn_predicted_te = nn_predicted_te.detach().numpy()

print('pickle it')
pkl_filename = '/data/hpcdata/users/racfur/DynamicPrediction/NNOutputs/MODELS/pickle_'+model_type+'_'+exp_name+'.pkl'
with open(pkl_filename, 'wb') as pckl_file:
    pickle.dump(h_linear, pckl_file)


#------------------
# Assess the model
#------------------

am.get_stats(model_type, exp_name, norm_outputs_tr, norm_outputs_te, nn_predicted_tr, nn_predicted_te)
am.plot_results(model_type, exp_name, norm_outputs_tr, norm_outputs_te, nn_predicted_tr, nn_predicted_te)

