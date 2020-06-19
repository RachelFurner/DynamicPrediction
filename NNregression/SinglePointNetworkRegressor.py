#!/usr/bin/env python
# coding: utf-8

#----------------------------
print('Import neccessary packages')
#----------------------------
from comet_ml import Experiment

import sys
sys.path.append('/data/hpcdata/users/racfur/DynamicPrediction/code_git/')
from Tools import CreateDataName as cn
from Tools import ReadRoutines as rr
from Tools import AssessModel as am
from Tools import Model_Plotting as rfplt

import numpy as np
import os
import xarray as xr
import matplotlib ; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle

from sklearn.preprocessing import PolynomialFeatures
import sklearn.metrics as metrics

from torch.utils import data
import torch.nn as nn
from torch.autograd import Variable
import torch

torch.cuda.empty_cache()

import matplotlib
matplotlib.use('agg')
print('matplotlib backend is:')
print(matplotlib.get_backend())
print("os.environ['DISPLAY']:")
print(os.environ['DISPLAY'])

#-------------------- -----------------
# Manually set variables for this run
#--------------------------------------
run_vars = {'dimension':3, 'lat':True , 'lon':True , 'dep':True , 'current':True ,
            'bolus_vel':True , 'sal':True , 'eta':True , 'density':True , 'poly_degree':2}
time_step='24hrs'

hyper_params = {
    "batch_size": 512,
    "num_epochs": 100, 
    "learning_rate": 0.001,
    "criterion": torch.nn.MSELoss(),
    "no_layers": 1,     # no of *hidden* layers
    "no_nodes": 10000   # Note 26106 features....
}

model_prefix = str(hyper_params["no_layers"])+'hiddenlayer_'

model_type = 'nn'

#--------------------------------
# Calculate some other variables 
#--------------------------------
data_name = cn.create_dataname(run_vars)
data_name = time_step+'_'+data_name
model_name = model_prefix+data_name+'_lr'+str(hyper_params['learning_rate'])

pkl_filename = '../../'+model_type+'_Outputs/MODELS/pickle_'+model_name+'.pkl'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using device:', device)

##---------------------
## Set up Comet stuff:
##---------------------
#experiment = Experiment(project_name="learnmitgcm2deg_SinglePointPrediction")
#experiment.set_name(model_name)
#experiment.log_parameters(hyper_params)
#experiment.log_others(run_vars)

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

#norm_inputs_tr   = norm_inputs_tr[:50,:]
#norm_inputs_val  = norm_inputs_val[:50,:] 
#norm_outputs_tr  = norm_outputs_tr[:50] 
#norm_outputs_val = norm_outputs_val[:50]  

#-------------------------
# Set up regression model
#-------------------------
class NetworkRegression(torch.nn.Sequential):
    def __init__(self, inputSize, outputSize, hyper_params):
        super(NetworkRegression, self).__init__()
        if hyper_params["no_layers"] == 1:
            self.linear = nn.Sequential( nn.Linear(inputSize, hyper_params["no_nodes"]), nn.Tanh(),
                                         nn.Linear(hyper_params["no_nodes"], outputSize) )
        elif hyper_params["no_layers"] == 2:
            self.linear = nn.Sequential( nn.Linear(inputSize, hyper_params["no_nodes"]), nn.Tanh(),
                                         nn.Linear(hyper_params["no_nodes"], hyper_params["no_nodes"]), nn.Tanh(), 
                                         nn.Linear(hyper_params["no_nodes"], outputSize) )
        elif hyper_params["no_layers"] == 3:
            self.linear = nn.Sequential( nn.Linear(inputSize, hyper_params["no_nodes"]), nn.Tanh(),
                                         nn.Linear(hyper_params["no_nodes"], hyper_params["no_nodes"]), nn.Tanh(),
                                         nn.Linear(hyper_params["no_nodes"], hyper_params["no_nodes"]), nn.Tanh(),
                                         nn.Linear(hyper_params["no_nodes"], outputSize) )
        elif hyper_params["no_layers"] == 4:
            self.linear = nn.Sequential( nn.Linear(inputSize, hyper_params["no_nodes"]), nn.Tanh(),
                                         nn.Linear(hyper_params["no_nodes"], hyper_params["no_nodes"]), nn.Tanh(),
                                         nn.Linear(hyper_params["no_nodes"], hyper_params["no_nodes"]), nn.Tanh(),
                                         nn.Linear(hyper_params["no_nodes"], hyper_params["no_nodes"]), nn.Tanh(),
                                         nn.Linear(hyper_params["no_nodes"], outputSize) )
    def forward(self, x):
        out = self.linear(x)
        return out

#------------------------
# Store data as Dataset'
#------------------------

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


#------------------------
# Instantiate the dataset
#------------------------
Train_Dataset = MITGCM_Dataset(norm_inputs_tr, norm_outputs_tr)
Val_Dataset  = MITGCM_Dataset(norm_inputs_val, norm_outputs_val)

train_loader = torch.utils.data.DataLoader(Train_Dataset, batch_size=hyper_params["batch_size"])
val_loader   = torch.utils.data.DataLoader(Val_Dataset,  batch_size=hyper_params["batch_size"])

#--------------
# Set up model
#--------------
inputDim = norm_inputs_tr.shape[1]
outputDim = norm_outputs_tr.shape[1]

model = NetworkRegression(inputDim, outputDim, hyper_params)
model.to(device)

optimizer = torch.optim.Adam( model.parameters(), lr=hyper_params["learning_rate"] )

train_loss_batch = []
train_loss_epoch = []
val_loss_epoch = []

# Train the model:
#with experiment.train():
if True:
    for epoch in range(hyper_params["num_epochs"]):
        train_loss_epoch_temp = 0.0
        val_loss_epoch_temp = 0.0
        for batch_inputs, batch_outputs in train_loader:
        
            # Converting inputs and labels to Variable and send to GPU if available
            if torch.cuda.is_available():
                batch_inputs = Variable(batch_inputs.cuda().float())
                batch_outputs = Variable(batch_outputs.cuda().float())
            else:
                batch_inputs = Variable(batch_inputs.float())
                batch_outputs = Variable(batch_outputs.float())
            
            model.train(True)

            # Clear gradient buffers because we don't want any gradient from previous batch to carry forward
            optimizer.zero_grad()
        
            # get prediction from the model, given the inputs
            predictions = model(batch_inputs)
        
            # get loss for the predicted output
            loss = hyper_params["criterion"](predictions, batch_outputs)
            # get gradients w.r.t to parameters
            loss.backward()
        
            # update parameters
            optimizer.step()
            
            train_loss_batch.append(loss.item()/batch_inputs.shape[0])
            train_loss_epoch_temp += loss.item()

        for batch_inputs, batch_outputs in val_loader:
        
            # Converting inputs and labels to Variable and send to GPU if available
            if torch.cuda.is_available():
                batch_inputs = Variable(batch_inputs.cuda().float())
                batch_outputs = Variable(batch_outputs.cuda().float())
            else:
                batch_inputs = Variable(batch_inputs.float())
                batch_outputs = Variable(batch_outputs.float())
            
            model.train(False)
    
            # get prediction from the model, given the inputs
            predicted = model(batch_inputs)
        
            # get loss for the predicted output
            loss = hyper_params["criterion"](predicted, batch_outputs)
    
            val_loss_epoch_temp += loss.item()
    
        epoch_loss = train_loss_epoch_temp / norm_inputs_tr.shape[0]
        print('epoch {}, loss {}'.format(epoch, epoch_loss))
        train_loss_epoch.append(epoch_loss )
        val_loss_epoch.append(val_loss_epoch_temp / norm_inputs_val.shape[0] )

        ## Log to Comet.ml
        #experiment.log_metric('Training_Loss', train_loss_epoch[-1], epoch = epoch)
        #experiment.log_metric('Validation_Loss', val_loss_epoch[-1], epoch = epoch)

    del batch_inputs, batch_outputs, predicted 
    torch.cuda.empty_cache()
 
print('pickle model')
with open(pkl_filename, 'wb') as pckl_file:
    pickle.dump(model, pckl_file)

# Plot training and validation loss 
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(train_loss_epoch)
ax1.plot(val_loss_epoch)
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.set_yscale('log')
ax1.legend(['Training Loss', 'Validation Loss'])
plt.savefig('../../'+model_type+'_Outputs/PLOTS/'+model_name+'/'+model_name+'_TrainingValLossPerEpoch.png', bbox_inches = 'tight', pad_inches = 0.1)
plt.close()

#-----------------------------------------------------
# Create full set of predictions and assess the model
#-----------------------------------------------------

norm_predicted_tr = np.zeros((norm_outputs_tr.shape))
norm_predicted_val = np.zeros((norm_outputs_val.shape))
# Loop through to predict for dataset, as memory limits mean we can't do this all at once.
# Maybe some clever way of doing this with the dataloader?
chunk_size = hyper_params["batch_size"]
no_chunks = int(norm_inputs_tr.shape[0]/chunk_size)
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
train_mse, val_mse = am.get_stats(model_type, model_name, name1='Training', truth1=denorm_outputs_tr, exp1=denorm_predicted_tr, pers1=predict_persistance_tr,
                                name2='Validation',  truth2=denorm_outputs_val, exp2=denorm_predicted_val, pers2=predict_persistance_val, name='TrainVal')

#with experiment.test():
#    experiment.log_metric("train_mse", train_mse)
#    experiment.log_metric("val_mse", val_mse)

print('plot results')
am.plot_results(model_type, model_name, denorm_outputs_tr, denorm_predicted_tr, name='training')
am.plot_results(model_type, model_name, denorm_outputs_val, denorm_predicted_val, name='val')

#-------------------------------------------------------------------
# plot histograms:
#-------------------------------------------------
fig = rfplt.Plot_Histogram(denorm_predicted_tr, 100) 
plt.savefig('../../'+model_type+'_Outputs/PLOTS/'+model_name+'/'+model_name+'_histogram_train_predictions.png', bbox_inches = 'tight', pad_inches = 0.1)
plt.close()

fig = rfplt.Plot_Histogram(denorm_predicted_val, 100)
plt.savefig('../../'+model_type+'_Outputs/PLOTS/'+model_name+'/'+model_name+'_histogram_val_predictions.png', bbox_inches = 'tight', pad_inches = 0.1)
plt.close()

fig = rfplt.Plot_Histogram(denorm_predicted_tr-denorm_outputs_tr, 100)
plt.savefig('../../'+model_type+'_Outputs/PLOTS/'+model_name+'/'+model_name+'_histogram_train_errors.png', bbox_inches = 'tight', pad_inches = 0.1)
plt.close()

fig = rfplt.Plot_Histogram(denorm_predicted_val-denorm_outputs_val, 100)
plt.savefig('../../'+model_type+'_Outputs/PLOTS/'+model_name+'_histogram_val_errors.png', bbox_inches = 'tight', pad_inches = 0.1)
plt.close()

