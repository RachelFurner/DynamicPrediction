#!/usr/bin/env python
# coding: utf-8

#----------------------------
print('Import neccessary packages')
#----------------------------

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
#print('1')
#os.system('nvidia-smi')

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
exp_name='skLearn_3dLatINT'


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
#            for time in range(0, data_end_index, 1000):  #  
for z in range(2,38,2):
    for x in range(2,7,2):
        for y in range(2,74,5):
            for time in range(0, data_end_index, 100):  # look at first 2000 years only - ignore region when model is at equilibrium
                input_temp = []
                #[input_temp.append(da_T[time,z,y+y_offset,x+x_offset]) for x_offset in halo_list for y_offset in halo_list]
                [input_temp.append(da_T[time,z+z_offset,y+y_offset,x+x_offset]) for x_offset in halo_list for y_offset in halo_list for z_offset in halo_list]
                #[input_temp.append(da_S[time,z+z_offset,y+y_offset,x+x_offset]) for x_offset in halo_list for y_offset in halo_list for z_offset in halo_list]
                #[input_temp.append(da_U[time,z+z_offset,y+y_offset,x+x_offset]) for x_offset in halo_list for y_offset in halo_list for z_offset in halo_list]
                #[input_temp.append(da_V[time,z+z_offset,y+y_offset,x+x_offset]) for x_offset in halo_list for y_offset in halo_list for z_offset in halo_list]
                #[input_temp.append(da_Eta[time,y+y_offset,x+x_offset]) for x_offset in halo_list for y_offset in halo_list]
                input_temp.append(da_lat[y])
                #input_temp.append(da_depth[z])

                inputs.append(input_temp)
                outputs.append([da_T[time+StepSize,z,y,x]-da_T[time,z,y,x]])
ds = None
da_T = None
da_S = None
da_U = None
da_V = None
da_Eta = None
da_lat = None
da_depth = None
 
inputs=np.asarray(inputs)
outputs=np.asarray(outputs)
print('now save them')
array_file = '/data/hpcdata/users/racfur/DynamicPrediction/RegressionOutputs/DATASETS/'+exp_name+'_InputsOutputs.npz'
np.savez(array_file, inputs,outputs)
print('saved')
print('open array from file')
inputs, outputs = np.load(array_file).values()
print('inputs.shape, outputs.shape')
print(inputs.shape, outputs.shape)

## Add polynomial combinations of the features
print('do poly features')
polynomial_features = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False) # bias included at linear regressor stage, not needed in input data
inputs = polynomial_features.fit_transform(inputs)
print('inputs.shape, outputs.shape')
print(inputs.shape, outputs.shape)


#randomise the sample order, and split into test and train data
split=int(split_ratio * inputs.shape[0])

np.random.seed(5)
ordering = np.random.permutation(inputs.shape[0])

inputs_tr = inputs[ordering][:split]
outputs_tr = outputs[ordering][:split]
inputs_te = inputs[ordering][split:]
outputs_te = outputs[ordering][split:]

print('shape for inputs and outputs: full; tr; te')
print(inputs.shape, outputs.shape)
print(inputs_tr.shape, outputs_tr.shape)
print(inputs_te.shape, outputs_te.shape)


#-------------------------------------------------------------------------------------------------------------
print('Plot temp at time t against outputs (temp at time t+1), to see if variance changes with input values') 
#-------------------------------------------------------------------------------------------------------------
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
ax3.scatter(inputs_te[:,xy], outputs_te, edgecolors=(0, 0, 0))
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
norm_inputs_te   = np.zeros(inputs_te.shape)
for i in range(inputs_tr.shape[1]):  #loop over each feature, normalising individually
    norm_inputs_tr[:, i], norm_inputs_te[:, i], inputs_mean[i], inputs_std[i] = normalise_data(inputs_tr[:, i], inputs_te[:,i])

## normalise outputs
norm_outputs_tr, norm_outputs_te, outputs_mean, outputs_std = normalise_data(outputs_tr[:], outputs_te[:])

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

stats_file=open('/data/hpcdata/users/racfur/DynamicPrediction/RegressionOutputs/STATS/'+exp_name+'.txt',"w+")
stats_file.write('shape for inputs and outputs. full; tr; te \n')
stats_file.write(str(inputs.shape)+'  '+str(outputs.shape)+'\n')
stats_file.write(str(norm_inputs_tr.shape)+'  '+str(norm_outputs_tr.shape)+'\n')
stats_file.write(str(norm_inputs_te.shape)+'  '+str(norm_outputs_te.shape)+'\n')

inputs = None
outputs = None

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
denorm_lr_predicted_tr = lr_predicted_tr*outputs_std+outputs_mean
denorm_lr_predicted_te = lr_predicted_te*outputs_std+outputs_mean

bottom = min(min(outputs_tr), min(denorm_lr_predicted_tr), min(outputs_te), min(denorm_lr_predicted_te))
top    = max(max(outputs_tr), max(denorm_lr_predicted_tr), max(outputs_te), max(denorm_lr_predicted_te))
bottom = bottom - 0.1*abs(top)
top    = top + 0.1*abs(top)

fig = plt.figure(figsize=(20,9.4))

ax1 = fig.add_subplot(121)
ax1.scatter(outputs_tr, denorm_lr_predicted_tr, edgecolors=(0, 0, 0))
ax1.plot([bottom, top], [bottom, top], 'k--', lw=1)
ax1.set_xlabel('Truth')
ax1.set_ylabel('Predicted')
ax1.set_title('Train')
ax1.text(.05,.95,'Mean Squared Error: {:.4e}'.format(lr_tr_mse),transform=ax1.transAxes)
ax1.set_xlim(bottom, top)
ax1.set_ylim(bottom, top)

ax2 = fig.add_subplot(122)
ax2.scatter(outputs_te, denorm_lr_predicted_te, edgecolors=(0, 0, 0))
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
