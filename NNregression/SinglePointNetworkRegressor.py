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
    "batch_size": 4092,
    "num_epochs": 1, 
    "learning_rate": 0.001,
    "criterion": torch.nn.MSELoss(),
    "no_layers": 0,     # no of *hidden* layers
    "no_nodes": 10000   # Note 26106 features....
}

model_prefix = str(hyper_params["no_layers"])+'hiddenlayer_lr'+str(hyper_params["learning_rate"])+'_batchsize'+str(hyper_params["batch_size"])

model_type = 'nn'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using device:', device)

Datadir = '/data/hpcdata/users/racfur/MITGCM_OUTPUT/100yr_Windx1.00_FrequentOutput/'
MITGCM_filename=Datadir+'cat_tave_50yr_SelectedVars_masked_withBolus.nc'
density_file = Datadir+'DensityData.npy'

data_end_index = 7200   # look at first 20yrs only
trainval_split_ratio = 0.7
valtest_split_ratio = 0.9

seed_value = 12321

#--------------------------------
# Calculate some other variables 
#--------------------------------
data_name = cn.create_dataname(run_vars)
data_name = data_name+'_'+time_step
model_name = model_prefix+data_name

pkl_filename = '../../'+model_type+'_Outputs/MODELS/pickle_'+model_name+'.pkl'

plot_dir = '../../'+model_type+'_Outputs/PLOTS/'+model_name
if not os.path.isdir(plot_dir):
   os.system("mkdir %s" % (plot_dir))

trainval_split = int(data_end_index*trainval_split_ratio) # point at which to switch from training data to validation data
valtest_split = int(data_end_index*valtest_split_ratio)   # point at which to switch from validation data to test data

torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
np.random.seed(seed_value)
os.environ['PYTHONHASHSEED']=str(seed_value)

#---------------------
# Set up Comet stuff:
#---------------------
#experiment = Experiment('VMfytsbw7TF2jYbEMShwqC3TD', project_name="learnmitgcm2deg_SinglePointPrediction")
#experiment.set_name(model_name)
#experiment.log_parameters(hyper_params)
#experiment.log_others(run_vars)

#---------------------
# Load MITGCM dataset
#---------------------
mitgcm_ds = xr.open_dataset(MITGCM_filename)
density = np.load( density_file, mmap_mode='r' )
#------------------------
# Store data as Dataset'
#------------------------
print('set up Dataset')

class MITGCM_Dataset(data.Dataset):
    """Dataset of MITGCM outputs"""
       
    def __init__(self, mitgcm_dataset, density):
        """
        Args:
           mitgcm_dataset - The mitgcm dataset read in through xarray
        """
        self.da_T=mitgcm_dataset['Ttave'].values
        self.da_S=mitgcm_dataset['Stave'].values
        self.da_U_tmp=mitgcm_dataset['uVeltave'].values
        self.da_V_tmp=mitgcm_dataset['vVeltave'].values
        self.da_Kwx=mitgcm_dataset['Kwx'].values
        self.da_Kwy=mitgcm_dataset['Kwy'].values
        self.da_Kwz=mitgcm_dataset['Kwz'].values
        self.da_Eta=mitgcm_dataset['ETAtave'].values
        self.da_lat=mitgcm_dataset['Y'].values
        self.da_lon=mitgcm_dataset['X'].values
        self.da_depth=mitgcm_dataset['Z'].values
        # Calc U and V by averaging surrounding points, to get on same grid as other variables
        self.da_U = (self.da_U_tmp[:,:,:,:-1]+self.da_U_tmp[:,:,:,1:])/2.
        self.da_V = (self.da_V_tmp[:,:,:-1,:]+self.da_V_tmp[:,:,1:,:])/2.
        self.density = density

        # Pad the data at each x-side, to deal with through flow
        # Add East most data to column on West side, and west most data to additional column on East side
        self.da_T    = np.concatenate( (self.da_T[:,:,:,-1:], self.da_T[:,:,:,:], self.da_T[:,:,:,:1]), axis=3)
        self.da_S    = np.concatenate( (self.da_S[:,:,:,-1:], self.da_S[:,:,:,:], self.da_S[:,:,:,:1]), axis=3)
        self.da_U    = np.concatenate( (self.da_U[:,:,:,-1:], self.da_U[:,:,:,:], self.da_U[:,:,:,:1]), axis=3)
        self.da_V    = np.concatenate( (self.da_V[:,:,:,-1:], self.da_V[:,:,:,:], self.da_V[:,:,:,:1]), axis=3)
        self.da_Kwx  = np.concatenate( (self.da_Kwx[:,:,:,-1:], self.da_Kwx[:,:,:,:], self.da_Kwx[:,:,:,:1]), axis=3)
        self.da_Kwy  = np.concatenate( (self.da_Kwy[:,:,:,-1:], self.da_Kwy[:,:,:,:], self.da_Kwy[:,:,:,:1]), axis=3)
        self.da_Kwz  = np.concatenate( (self.da_Kwz[:,:,:,-1:], self.da_Kwz[:,:,:,:], self.da_Kwz[:,:,:,:1]), axis=3)
        self.da_Eta  = np.concatenate( (self.da_Eta[:,:,-1:], self.da_Eta[:,:,:], self.da_Eta[:,:,:1]), axis=2)
        self.da_lon  = np.concatenate( (self.da_lon[-1:], self.da_lon[:], self.da_lon[:1]), axis=0)
        self.density = np.concatenate( (self.density[:,:,:,-1:], self.density[:,:,:,:], self.density[:,:,:,:1]), axis=3) 
     
       	x_size = self.da_T.shape[3]
        y_size = self.da_T.shape[2]
        z_size = self.da_T.shape[1]
        t_size = self.da_T.shape[0]

        # Create array of the locations (temporal and spatial) of all input samples
        input_times = np.arange(200,t_size,2000)  # Take every 200th point in time, ignoring the first point
        
        # Need to Split domain into three regions to work around land etc

        # Region 1: main part of domain, ignoring one point above/below land/domain edge at north and south borders, and
        # ignoring one point down entire West boundary, and two points down entire East boundary (i.e. acting as though 
        # land split carries on all the way to the bottom of the domain)
        # Note x points refer to locations in padded grid
        input_x = np.arange(2, x_size-3)
        input_y = np.arange(1, y_size-3)
        input_z = np.arange(1, z_size-1)
        self.sample_locations = np.array(np.meshgrid(input_times, input_z, input_y, input_x)).T.reshape(-1,4)

        # Region 2: West side, Southern edge, above the depth where the land split carries on. One cell strip where throughflow enters.
        # Note x points refer to locations in padded grid
        input_x = np.arange(1, 2)
        input_y = np.arange(1, 15)
        input_z = np.arange(1, 31)
        tmp = np.array(np.meshgrid(input_times, input_z, input_y, input_x)).T.reshape(-1,4)
        self.sample_locations = np.concatenate( (self.sample_locations, tmp), axis=0 )

        # Region 3: East side, Southern edge, above the depth where the land split carries on. Two column strip where throughflow enters.
        # Note x points refer to locations in padded grid
        input_x = np.arange(x_size-3, x_size-1)
        input_y = np.arange(1, 15)
        input_z = np.arange(1, 31)
        tmp = np.array(np.meshgrid(input_times, input_z, input_y, input_x)).T.reshape(-1,4)
        self.sample_locations = np.concatenate( (self.sample_locations, tmp), axis=0 )

        print('self.sample_locations.shape')
        print(self.sample_locations.shape)

        # Load in means and std for normalising
        mean_std_file = '/data/hpcdata/users/racfur/DynamicPrediction/INPUT_OUTPUT_ARRAYS/SinglePoint_'+data_name+'_MeanStd.npz'
        self.input_mean_array, self.input_std_array, self.output_mean, self.output_std = np.load(mean_std_file).values()
           

    def __getitem__(self, index):
        t = int(self.sample_locations[index,0])
        z = int(self.sample_locations[index,1])
        y = int(self.sample_locations[index,2])
        x = int(self.sample_locations[index,3])
        if run_vars['dimension'] == 2:
           sample_input = rr.GetInputs( run_vars, self.da_T[t,z,y-1:y+2,x-1:x+2], self.da_S[t,z,y-1:y+2,x-1:x+2], 
                                        self.da_U[t,z,y-1:y+2,x-1:x+2], self.da_V[t,z,y-1:y+2,x-1:x+2],
                                        self.da_Kwx[t,z,y-1:y+2,x-1:x+2], self.da_Kwy[t,z,y-1:y+2,x-1:x+2], 
                                        self.da_Kwz[t,z,y-1:y+2,x-1:x+2], self.density[t,z,y-1:y+2,x-1:x+2],
                                        self.da_Eta[t,y-1:y+2,x-1:x+2], self.da_lat[y], self.da_lon[x], self.da_depth[z] )
        elif run_vars['dimension'] == 3:
           sample_input = rr.GetInputs( run_vars, self.da_T[t,z-1:z+2,y-1:y+2,x-1:x+2], self.da_S[t,z-1:z+2,y-1:y+2,x-1:x+2], 
                                        self.da_U[t,z-1:z+2,y-1:y+2,x-1:x+2], self.da_V[t,z-1:z+2,y-1:y+2,x-1:x+2],
                                        self.da_Kwx[t,z-1:z+2,y-1:y+2,x-1:x+2], self.da_Kwy[t,z-1:z+2,y-1:y+2,x-1:x+2], 
                                        self.da_Kwz[t,z-1:z+2,y-1:y+2,x-1:x+2], self.density[t,z-1:z+2,y-1:y+2,x-1:x+2],
                                        self.da_Eta[t,y-1:y+2,x-1:x+2], self.da_lat[y], self.da_lon[x], self.da_depth[z] )
        sample_input = sample_input.reshape(-1)

        sample_output = (  self.da_T[ t+1, z, y, x ] 
                         - self.da_T[ t  , z, y, x ] )
        sample_output = np.array(sample_output)
        sample_output = sample_output.reshape(1)

        # Normalise the inputs and outputs
        sample_input = np.divide( np.subtract(sample_input, self.input_mean_array), self.input_std_array)
        sample_output = np.divide( np.subtract(sample_output, self.output_mean), self.output_std)

        return (sample_input, sample_output)

    def __len__(self):
        return self.sample_locations.shape[0]


#------------------------
# Instantiate the dataset
#------------------------
#Train_Dataset = MITGCM_Dataset(norm_inputs_tr, norm_outputs_tr)
#Val_Dataset  = MITGCM_Dataset(norm_inputs_val, norm_outputs_val)
Train_Dataset = MITGCM_Dataset( mitgcm_ds.isel(T=slice(0, trainval_split)), density[0:trainval_split,:,:,:] )
Val_Dataset   = MITGCM_Dataset( mitgcm_ds.isel(T=slice(trainval_split, valtest_split)), density[trainval_split:valtest_split,:,:,:] )

# Get input and output dimensions:
get_shape_loader = torch.utils.data.DataLoader(Train_Dataset, batch_size=1)
batch_inputs, batch_outputs = next(iter(get_shape_loader))
print('batch_inputs.shape')
print(batch_inputs.shape)
print('batch_outputs.shape')
print(batch_outputs.shape)
inputDim = batch_inputs.shape[-1]
outputDim = batch_outputs.shape[-1]

train_loader = torch.utils.data.DataLoader(Train_Dataset, batch_size=hyper_params["batch_size"], shuffle=False)
val_loader   = torch.utils.data.DataLoader(Val_Dataset,  batch_size=hyper_params["batch_size"], shuffle=False)

#-------------------------
# Set up regression model
#-------------------------
print('set up model')
class NetworkRegression(torch.nn.Sequential):
    def __init__(self, inputSize, outputSize, hyper_params):
        super(NetworkRegression, self).__init__()
        if hyper_params["no_layers"] == 0:
            self.linear = torch.nn.Linear(inputSize, outputSize)
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


model = NetworkRegression(inputDim, outputDim, hyper_params)
model.to(device)

optimizer = torch.optim.Adam( model.parameters(), lr=hyper_params["learning_rate"] )


#-----------------
# Train the Model
#-----------------
print('Train the model')
train_loss_batch = []
train_loss_epoch = []
val_loss_epoch = []

# Train the model:
#with experiment.train():
if True:
    for epoch in range(hyper_params["num_epochs"]):
        print('epoch :'+str(epoch))
        train_loss_epoch_temp = 0.0
        for batch_inputs, batch_outputs in train_loader:
            # Converting inputs and labels to Variable and send to GPU if available
            if torch.cuda.is_available():
                batch_inputs = Variable(batch_inputs.cuda().float())
                batch_outputs = Variable(batch_outputs.cuda().float())
            else:
                batch_inputs = Variable(batch_inputs.float())
                batch_outputs = Variable(batch_outputs.float())
     
            print('batch_inputs.shape')
            print(batch_inputs.shape)
            print('batch_outputs.shape')
            print(batch_outputs.shape)
           
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

        val_loss_epoch_temp = 0.0
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
            predictions = model(batch_inputs)
            
            # get loss for the predicted output
            loss = hyper_params["criterion"](predictions, batch_outputs)
            #
            val_loss_epoch_temp += loss.item()
    
        epoch_loss = train_loss_epoch_temp / norm_inputs_tr.shape[0]
        print('epoch {}, loss {}'.format(epoch, epoch_loss))
        train_loss_epoch.append(epoch_loss )
        val_loss_epoch.append(val_loss_epoch_temp / norm_inputs_val.shape[0] )

        # Log to Comet.ml
        #experiment.log_metric('Training_Loss', train_loss_epoch[-1], epoch = epoch)
        #experiment.log_metric('Validation_Loss', val_loss_epoch[-1], epoch = epoch)

    del batch_inputs, batch_outputs, predictions 
    torch.cuda.empty_cache()

# Plot training and validation loss 
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(311)
ax1.plot(train_loss_epoch, color='blue')
ax1.plot(val_loss_epoch, color='orange')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.set_yscale('log')
ax1.annotate('Final Training Loss: '+str(np.round(train_loss_epoch[-1],6)), (0.749, 0.94), xycoords='figure fraction')
ax1.annotate('Final Validation Loss: '+str(np.round(val_loss_epoch[-1],6)), (0.749, 0.91), xycoords='figure fraction')
plt.legend(['Training Loss', 'Validation Loss'])
ax2 = fig.add_subplot(312)
ax2.plot(train_loss_epoch, color='blue')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Loss')
ax2.set_yscale('log')
ax2.annotate('Final Training Loss: '+str(np.round(train_loss_epoch[-1],6)), (0.749, 0.615), xycoords='figure fraction')
ax3 = fig.add_subplot(313)
ax3.plot(val_loss_epoch, color='orange')
ax3.set_xlabel('Epochs')
ax3.set_ylabel('Loss')
ax3.set_yscale('log')
ax3.annotate('Final Validation Loss: '+str(np.round(val_loss_epoch[-1],6)), (0.749, 0.29), xycoords='figure fraction')
plt.subplots_adjust(hspace = 0.35, left=0.05, right=0.95, bottom=0.07, top=0.95)
plt.savefig(plot_dir+'/'+model_name+'_TrainValLossPerEpoch.png', bbox_inches = 'tight', pad_inches = 0.1)
 
print('pickle model')
with open(pkl_filename, 'wb') as pckl_file:
    pickle.dump(model, pckl_file)

#-----------------------------------------------------
# Create full set of predictions and assess the model
#-----------------------------------------------------

norm_predicted_tr = np.zeros((norm_outputs_tr.shape))
norm_predicted_val = np.zeros((norm_outputs_val.shape))
# Loop through to predict for dataset, as memory limits mean we can't do this all at once.
# Maybe some clever way of doing this with the dataloader?
chunk_size = hyper_params["batch_size"]
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
plt.savefig(plot_dir+'/'+model_name+'_histogram_train_predictions.png', bbox_inches = 'tight', pad_inches = 0.1)
plt.close()

fig = rfplt.Plot_Histogram(denorm_predicted_val, 100)
plt.savefig(plot_dir+'/'+model_name+'_histogram_val_predictions.png', bbox_inches = 'tight', pad_inches = 0.1)
plt.close()

fig = rfplt.Plot_Histogram(denorm_predicted_tr-denorm_outputs_tr, 100)
plt.savefig(plot_dir+'/'+model_name+'_histogram_train_errors.png', bbox_inches = 'tight', pad_inches = 0.1)
plt.close()

fig = rfplt.Plot_Histogram(denorm_predicted_val-denorm_outputs_val, 100)
plt.savefig(plot_dir+'/'+model_name+'_histogram_val_errors.png', bbox_inches = 'tight', pad_inches = 0.1)
plt.close()

