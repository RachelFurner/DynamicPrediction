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

import matplotlib
matplotlib.use('agg')
print('matplotlib backend is:')
print(matplotlib.get_backend())
print("os.environ['DISPLAY']:")
print(os.environ['DISPLAY'])

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using device:', device)
print()

torch.cuda.empty_cache()

#-------------------- -----------------
# Manually set variables for this run
#--------------------------------------
run_vars = {'dimension':3, 'lat':True , 'lon':True , 'dep':True , 'current':True ,
            'bolus_vel':True , 'sal':True , 'eta':True ,'density':True , 'poly_degree':2}
time_step = '24hrs'

learningRate = 0.001 
epochs = 100
batch_size = 512

model_prefix = 'NNasLinearRegressor_'

model_type = 'nn'

train_end_ratio = 0.7
val_end_ratio = 0.9

train_model = True 
#--------------------------------
# Calculate some other variables 
#--------------------------------
data_name = cn.create_dataname(run_vars)
data_name = time_step+'_'+data_name
model_name = model_prefix+data_name+'_lr'+str(learningRate)

pkl_filename = '../../'+model_type+'_Outputs/MODELS/pickle_'+model_name+'.pkl'
#--------------------------------------------------------------
# Read in the data from saved arrays
#--------------------------------------------------------------
inputs_tr_file   = '/data/hpcdata/users/racfur/DynamicPrediction/INPUT_OUTPUT_ARRAYS/SinglePoint_'+data_name+'_InputsTr.npy'
inputs_val_file  = '/data/hpcdata/users/racfur/DynamicPrediction/INPUT_OUTPUT_ARRAYS/SinglePoint_'+data_name+'_InputsVal.npy'
outputs_tr_file  = '/data/hpcdata/users/racfur/DynamicPrediction/INPUT_OUTPUT_ARRAYS/SinglePoint_'+data_name+'_OutputsTr.npy'
outputs_val_file = '/data/hpcdata/users/racfur/DynamicPrediction/INPUT_OUTPUT_ARRAYS/SinglePoint_'+data_name+'_OutputsVal.npy'
norm_inputs_tr   = np.load(inputs_tr_file, mmap_mode='r')
norm_inputs_val  = np.load(inputs_val_file, mmap_mode='r')
norm_outputs_tr  = np.load(outputs_tr_file, mmap_mode='r')
norm_outputs_val = np.load(outputs_val_file, mmap_mode='r')

#norm_inputs_tr   = norm_inputs_tr[:10,:]
#norm_inputs_val  = norm_inputs_val[:10,:]
#norm_outputs_tr  = norm_outputs_tr[:10,:]
#norm_outputs_val = norm_outputs_val[:10,:]

print(norm_inputs_tr.shape)
print(norm_outputs_tr.shape)

#-----------------------------------------------------------------
# Set up regression model in PyTorch (so we can use GPUs!)
#-----------------------------------------------------------------
class linearRegression(torch.nn.Sequential):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out
   
if train_model:
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
           sample_input = np.array(self.ds_inputs[index,:])
           sample_output = np.array(self.ds_outputs[index])
  
           return (sample_input, sample_output)
   
       def __len__(self):
           return self.ds_inputs.shape[0]
   
   
   # Instantiate the dataset
   Train_Dataset = MITGCM_Dataset(norm_inputs_tr, norm_outputs_tr)
   Val_Dataset  = MITGCM_Dataset(norm_inputs_val, norm_outputs_val)
   
   train_loader = torch.utils.data.DataLoader(Train_Dataset, batch_size=batch_size)
   val_loader   = torch.utils.data.DataLoader(Val_Dataset,  batch_size=batch_size)
   
   # Call the model
   inputDim = norm_inputs_tr.shape[1] 
   outputDim = norm_outputs_tr.shape[1]
   
   print('inputDim:')
   print(inputDim)
   print('outputDim:')
   print(outputDim)

   model = linearRegression(inputDim, outputDim)
   # send to GPU if available
   if torch.cuda.is_available():
       model.cuda()
   
   criterion = torch.nn.MSELoss(reduction = 'sum') 
   optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)
   
   #-----------------
   # Train the model 
   #-----------------
   train_loss_epoch = []
   train_loss_batch = []
   val_loss_epoch = []
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
      
          model.train(True)
      
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
      
       # Calc val loss batch at a time, using val_loader
       val_loss_epoch_temp = 0.0
       for batch_inputs, batch_outputs in val_loader:
          # Converting inputs and labels to Variable
          if torch.cuda.is_available():
              batch_inputs = Variable(batch_inputs.cuda().float())
              batch_labels = Variable(batch_outputs.cuda().float())
          else:
              batch_inputs = Variable(batch_inputs.float())
              batch_labels = Variable(batch_outputs.float())
      
          model.train(False)
      
          # get predictions from the model, given the inputs
          predictions = model(batch_inputs)
      
          # get loss for the predicted output
          loss = criterion(predictions, batch_labels)
                    
          val_loss_epoch_temp += loss.item()
      
       train_epoch_loss = train_loss_epoch_temp / norm_inputs_tr.shape[0]
       print('epoch {}, loss {}'.format(epoch, train_epoch_loss))
       train_loss_epoch.append(train_epoch_loss)
       val_loss_epoch.append(val_loss_epoch_temp / norm_inputs_val.shape[0])

   del batch_inputs
   del batch_outputs
   torch.cuda.empty_cache()
   
   # Plot training loss 
   fig = plt.figure(figsize=(12,8))
   ax1 = fig.add_subplot(311)
   ax1.plot(train_loss_epoch, color='blue')
   ax1.plot(val_loss_epoch, color='orange')
   ax1.set_xlabel('Epochs')
   ax1.set_ylabel('Loss')
   ax1.set_yscale('log')
   ax1.annotate('Final Epoch Loss: '+str(np.round(train_loss_epoch[-1],6)), (0.75, 0.94), xycoords='figure fraction')
   ax1.annotate('Final Val Loss: '+str(np.round(val_loss_epoch[-1],6)), (0.772, 0.91), xycoords='figure fraction')
   plt.legend(['Training Loss', 'Validation Loss'])
   ax2 = fig.add_subplot(312)
   ax2.plot(train_loss_epoch, color='blue')
   ax2.set_xlabel('Epochs')
   ax2.set_ylabel('Loss')
   ax2.set_yscale('log')
   ax2.annotate('Final Epoch Loss: '+str(np.round(train_loss_epoch[-1],6)), (0.75, 0.615), xycoords='figure fraction')
   ax3 = fig.add_subplot(313)
   ax3.plot(val_loss_epoch, color='orange')
   ax3.set_xlabel('Epochs')
   ax3.set_ylabel('Loss')
   ax3.set_yscale('log')
   ax3.annotate('Final Val Loss: '+str(np.round(val_loss_epoch[-1],6)), (0.772, 0.29), xycoords='figure fraction')
   plt.subplots_adjust(hspace = 0.35, left=0.05, right=0.95, bottom=0.07, top=0.95)
   plt.savefig('../../'+model_type+'_Outputs/PLOTS/'+model_name+'_TrainValLossPerEpoch.png', bbox_inches = 'tight', pad_inches = 0.1)
   
   print('pickle model')
   with open(pkl_filename, 'wb') as pckl_file:
       pickle.dump(model, pckl_file)
else:
   print('opening '+pkl_filename)
   with open(pkl_filename, 'rb') as file:
      model = pickle.load(file)
   # send to GPU if available
   if torch.cuda.is_available():
      model.cuda()

#-----------------------------------------------------
# Create full set of predictions and assess the model
#-----------------------------------------------------

norm_predicted_tr = np.zeros((norm_outputs_tr.shape))
norm_predicted_val = np.zeros((norm_outputs_val.shape))
# Loop through to predict for dataset, as memory limits mean we can't do this all at once.
# Maybe some clever way of doing this with the dataloader?
chunk_size = batch_size
no_chunks  = int(norm_inputs_tr.shape[0]/chunk_size)
for i in range(no_chunks):
   if torch.cuda.is_available():
      norm_tr_inputs_temp  = Variable(torch.from_numpy(norm_inputs_tr[i*chunk_size:(i+1)*chunk_size]).cuda().float())
      norm_val_inputs_temp = Variable(torch.from_numpy(norm_inputs_val[i*chunk_size:(i+1)*chunk_size]).cuda().float())
   else:
      norm_tr_inputs_temp  = Variable(torch.from_numpy(norm_inputs_tr[i*chunk_size:(i+1)*chunk_size]).float())
      norm_val_inputs_temp = Variable(torch.from_numpy(norm_inputs_val[i*chunk_size:(i+1)*chunk_size]).float())

   norm_predicted_tr[i*chunk_size:(i+1)*chunk_size]  = model(norm_tr_inputs_temp).cpu().detach().numpy()
   norm_predicted_val[i*chunk_size:(i+1)*chunk_size] = model(norm_val_inputs_temp).cpu().detach().numpy()
   del norm_tr_inputs_temp
   del norm_val_inputs_temp
   torch.cuda.empty_cache()

#------------------------------------------
# De-normalise the outputs and predictions
#------------------------------------------
mean_std_file = '/data/hpcdata/users/racfur/DynamicPrediction/INPUT_OUTPUT_ARRAYS/SinglePoint_'+data_name+'_MeanStd.npz'
input_mean, input_std, output_mean, output_std = np.load(mean_std_file).values()

def denormalise_data(norm_data,mean,std):
    denorm_data = norm_data * std + mean
    return denorm_data

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
am.get_stats(model_type, model_name, name1='Training', truth1=denorm_outputs_tr, exp1=denorm_predicted_tr, pers1=predict_persistance_tr,
                                name2='Validation',  truth2=denorm_outputs_val, exp2=denorm_predicted_val, pers2=predict_persistance_val, name='TrainVal')

print('plot results')
am.plot_results(model_type, model_name, denorm_outputs_tr, denorm_predicted_tr, name='training')
am.plot_results(model_type, model_name, denorm_outputs_val, denorm_predicted_val, name='val')

#-------------------------------------------------------------------
# plot histograms:
#-------------------------------------------------
fig = rfplt.Plot_Histogram(denorm_predicted_tr, 100) 
plt.savefig('../../'+model_type+'_Outputs/PLOTS/'+model_name+'_histogram_train_predictions.png', bbox_inches = 'tight', pad_inches = 0.1)

fig = rfplt.Plot_Histogram(denorm_predicted_val, 100)
plt.savefig('../../'+model_type+'_Outputs/PLOTS/'+model_name+'_histogram_val_predictions.png', bbox_inches = 'tight', pad_inches = 0.1)

fig = rfplt.Plot_Histogram(denorm_predicted_tr-denorm_outputs_tr, 100)
plt.savefig('../../'+model_type+'_Outputs/PLOTS/'+model_name+'_histogram_train_errors.png', bbox_inches = 'tight', pad_inches = 0.1)

fig = rfplt.Plot_Histogram(denorm_predicted_val-denorm_outputs_val, 100)
plt.savefig('../../'+model_type+'_Outputs/PLOTS/'+model_name+'_histogram_val_errors.png', bbox_inches = 'tight', pad_inches = 0.1)
