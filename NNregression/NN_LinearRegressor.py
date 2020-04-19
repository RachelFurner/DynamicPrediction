#!/usr/bin/env python
# coding: utf-8

#----------------------------
print('Import neccessary packages')
#----------------------------
import sys
sys.path.append('/data/hpcdata/users/racfur/DynamicPrediction/code_git/')
from Tools import CreateDataName as cn
from Tools import ReadRoutines as rr
from Tools import AssessModel as am
from Tools import Model_Plotting as rfplt

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
run_vars={'dimension':3, 'lat':True , 'lon':True , 'dep':True , 'current':True , 'bolus_vel':True , 'sal':True , 'eta':True ,'density':True, 'poly_degree':2}
time_step='1mnth'

exp_prefix = 'NNasLinearRegressor_'

model_type = 'nn'

mit_dir = '/data/hpcdata/users/racfur/MITGCM_OUTPUT/20000yr_Windx1.00_mm_diag/'
MITGCM_filename=mit_dir+'cat_tave_2000yrs_SelectedVars_masked.nc'

learningRate = 0.001 
epochs = 100

#--------------------------------
# Calculate some other variables 
#--------------------------------
data_name = cn.create_dataname(run_vars)
data_name = time_step+'_'+data_name
exp_name = exp_prefix+data_name

#--------------------------------------------------------------
# Read in the data from saved array
#--------------------------------------------------------------
inputsoutputs_file = '/data/hpcdata/users/racfur/DynamicPrediction/INPUT_OUTPUT_ARRAYS/SinglePoint_'+data_name+'_InputsOutputs.npz'
print(inputsoutputs_file)
norm_inputs_tr, norm_inputs_val, norm_inputs_te, norm_outputs_tr, norm_outputs_val, norm_outputs_te = np.load(inputsoutputs_file).values()

del norm_inputs_te
del norm_outputs_te

#norm_inputs_tr  = norm_inputs_tr[:100,:]
#norm_inputs_val = norm_inputs_val[:100,:]
#norm_outputs_tr  = norm_outputs_tr[:100,:]
#norm_outputs_val = norm_outputs_val[:100,:]

#-----------------------------------------------------------------
# Set up regression model in PyTorch (so we can use GPUs!)
#-----------------------------------------------------------------
class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out

#-----------------------
# Store data as Dataset
#-----------------------

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
Val_Dataset  = MITGCM_Dataset(norm_inputs_val, norm_outputs_val)

train_loader = torch.utils.data.DataLoader(Train_Dataset, batch_size=64)
val_loader   = torch.utils.data.DataLoader(Val_Dataset,  batch_size=64)



# Call the model
inputDim = norm_inputs_tr.shape[1] 
outputDim = norm_outputs_tr.shape[1]

model = linearRegression(inputDim, outputDim)
# send to GPU if available
if torch.cuda.is_available():
    model.cuda()

criterion = torch.nn.MSELoss() 
optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)

#-----------------
# Train the model 
#-----------------
train_loss_epoch = []
train_loss_batch = []
#val_loss_epoch = []

for epoch in range(epochs):
    train_loss_epoch_temp = 0.0
    
    # optimise batch at a time, using train_loader
    for batch_inputs, batch_outputs in train_loader:
       # Converting inputs and labels to Variable
       if torch.cuda.is_available():
           batch_inputs = Variable(batch_inputs.cuda().float())
           batch_labels = Variable(batch_outputs.cuda().float())
       else:
           batch_inputs = Variable(batch_inputs.float())
           batch_labels = Variable(batch_outputs.float())
   
       # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
       optimizer.zero_grad()
   
       # get predictions from the model, given the inputs
       predictions = model(batch_inputs)
   
       # get loss for the predicted output
       loss = criterion(predictions, batch_labels)
       # get gradients w.r.t to parameters
       loss.backward()
   
       # update parameters
       optimizer.step()

       train_loss_batch.append(loss.item()/batch_inputs.shape[0])
       train_loss_epoch_temp += loss.item()
   
    epoch_loss = train_loss_epoch_temp / norm_inputs_tr.shape[0]
    print('epoch {}, loss {}'.format(epoch, epoch_loss))
    train_loss_epoch.append(epoch_loss)

# Plot training loss 
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.plot(train_loss_epoch)
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Training Loss')
ax1.set_yscale('log')
ax2 = fig.add_subplot(212)
ax2.plot(train_loss_batch)
ax2.set_xlabel('batch')
ax2.set_ylabel('Training Loss')
ax2.set_yscale('log')
plt.savefig('../../'+model_type+'_Outputs/PLOTS/'+exp_name+'_TrainingLossPerEpoch.png', bbox_inches = 'tight', pad_inches = 0.1)

print('pickle model')
pkl_filename = '../../'+model_type+'_Outputs/MODELS/pickle_'+exp_name+'.pkl'
with open(pkl_filename, 'wb') as pckl_file:
    pickle.dump(model, pckl_file)


#------------------------
# Predict with the model
#------------------------
with torch.no_grad(): # we don't need gradients in the testing phase
    if torch.cuda.is_available():
        norm_predicted_tr = model(Variable(torch.from_numpy(norm_inputs_tr).cuda().float())).cpu().data.numpy()
        norm_predicted_val = model(Variable(torch.from_numpy(norm_inputs_val).cuda().float())).cpu().data.numpy()
    else:
        norm_predicted_tr = model(Variable(torch.from_numpy(norm_inputs_tr))).data.np()
        norm_predicted_val = model(Variable(torch.from_numpy(norm_inputs_val))).data.np()


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
denorm_predicted_tr = denormalise_data(norm_predicted_tr, output_mean, output_std)
denorm_predicted_val = denormalise_data(norm_predicted_val, output_mean, output_std)
denorm_outputs_tr = denormalise_data(norm_outputs_tr, output_mean, output_std)
denorm_outputs_val = denormalise_data(norm_outputs_val, output_mean, output_std)

#------------------
# Assess the model
#------------------
# Calculate 'persistance' score - persistence prediction is just zero everywhere as we're predicting the trend
predict_persistance_tr = np.zeros(denorm_outputs_tr.shape)
predict_persistance_val = np.zeros(denorm_outputs_val.shape)

print('get stats')
am.get_stats(model_type, exp_name, name1='Training', truth1=denorm_outputs_tr, exp1=denorm_predicted_tr, pers1=predict_persistance_tr,
                                name2='Validation',  truth2=denorm_outputs_val, exp2=denorm_predicted_val, pers2=predict_persistance_val, name='TrainVal')

print('plot results')
am.plot_results(model_type, exp_name, denorm_outputs_tr, denorm_predicted_tr, name='training')
am.plot_results(model_type, exp_name, denorm_outputs_val, denorm_predicted_val, name='val')

#-------------------------------------------------------------------
# plot histograms:
#-------------------------------------------------
fig = rfplt.Plot_Histogram(denorm_predicted_tr, 100) 
plt.savefig('../../'+model_type+'_Outputs/PLOTS/'+exp_name+'_histogram_train_predictions', bbox_inches = 'tight', pad_inches = 0.1)

fig = rfplt.Plot_Histogram(denorm_predicted_val, 100)
plt.savefig('../../'+model_type+'_Outputs/PLOTS/'+exp_name+'_histogram_val_predictions', bbox_inches = 'tight', pad_inches = 0.1)

fig = rfplt.Plot_Histogram(denorm_predicted_tr-denorm_outputs_tr, 100)
plt.savefig('../../'+model_type+'_Outputs/PLOTS/'+exp_name+'_histogram_train_errors', bbox_inches = 'tight', pad_inches = 0.1)


