#!/usr/bin/env python
# coding: utf-8

# # Script to predict dynamic time stepping of an ocean model
# 
# Written by Rachel Furner, April 2019.
# 
# Collaborative work with colleagues at BAS, and the ATI

 
#--------------------------------------------------------------------------------------------------------------------------
# Import relevant packages etc
import xarray as xr
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from tensorflow.keras.layers import MaxPooling3D, Conv3D, UpSampling3D, Cropping3D
from tensorflow.keras.callbacks import EarlyStopping
import pickle

print('imports done')

#--------------------------------------------------------------------------------------------------------------------------
# Define variables for this experiment
StepSize = 10 #number of timesteps forward which we want to predict
SkipOver = 5000

# Define early_stopping_monitor
early_stopping_monitor = EarlyStopping(patience=2)

#--------------------------------------------------------------------------------------------------------------------------
# Read in data files as Xarrays
DIR = '/data/hpcdata/users/racfur/'
exp_list = ['4500yr_Windx0.50']
file_list =[]
for exp in exp_list:
    for filename in os.listdir(os.path.join(DIR,exp)):
        #if filename.__contains__('cat'):
        file_list.append(os.path.join(DIR,exp,filename))

print(file_list)
#--------------------------------------------------------------------------------------------------------------------------
# Plot two surface fields, StepSize apart, and their difference
ds   = xr.open_dataset(file_list[0])
fig = plt.figure()
ax1 = plt.subplot(131)
ds.Temp.isel(T=0,Z=0).plot(cmap='OrRd')

ax2 = plt.subplot(132)
ds.Temp.isel(T=StepSize,Z=0).plot(cmap='OrRd')

ax3 = plt.subplot(133)
(ds.Temp.isel(T=StepSize,Z=0)-ds.Temp.isel(T=0,Z=0)).plot(cmap='RdBu' )

fig.savefig('difference.png')


#--------------------------------------------------------------------------------------------------------------------------
# Define Input output pairs

# Eventual aim is that inputs are full model field at t-stepsize and t, outputs are full model fields at t+StepSize.
# For now, input is just Temp at just at time step t, and output is Temp at t+stepsize.
# In future could involve lots of input variables.
 
# StepSize can be changed easily (defined further up) - plan to test different values and see how well things work.
 
# We take input,output pairs with 't' spaced by 'SkipOver' steps apart - ideally 'SkipOver' large enough to ensure some
# independance between training samples. Balance between low values giving us lots of training samples, but also a
# desire for independant training samples

training_data=[]
for file in file_list:
  ds   = xr.open_dataset(file)
  for time in range(StepSize, len(ds.T.data)-StepSize, SkipOver):
    for x in range(3,8,3):
      for y in range(5,73,5):
        for z in range(5,37,5):
          training_data.append([ds.variables['Temp'][time,z,y  ,x  ].values,
				ds.variables['Temp'][time,z,y-1,x  ].values, 
				ds.variables['Temp'][time,z,y-1,x-1].values, 
				ds.variables['Temp'][time,z,y  ,x-1].values, 
				ds.variables['Temp'][time,z,y+1,x-1].values, 
				ds.variables['Temp'][time,z,y+1,x  ].values, 
				ds.variables['Temp'][time,z,y+1,x+1].values, 
				ds.variables['Temp'][time,z,y  ,x+1].values, 
				ds.variables['Temp'][time,z,y-1,x+1].values, 
				ds.variables['Temp'][time+StepSize,z,y,x].values])

#shuffle dataset
random.shuffle(training_data)
print(len(training_data))
print(len(training_data[0]))

#--------------------------------------------------------------------------------------------------------------------------
# Put this into X and Y arrays, ready for model to read
 
X=[]
Y=[]

for feat1,feat2,feat3,feat4,feat5,feat6,feat7,feat8,feat9,label in training_data:
        X.append([feat1,feat2,feat3,feat4,feat5,feat6,feat7,feat8,feat9])
        Y.append(label)

# convert to arrays, as model wont accept a list, if using multiple features we may need to transpose the array so features are in the last dimension - check this.
#X=np.array(X).transpose(0, 2, 3, 4, 1)
X=np.array(X)
Y=np.array(Y)

print(X.shape)
print(Y.shape)


#--------------------------------------------------------------------------------------------------------------------------
# Normalise data
#def normalise_data(X):
#    X=tf.keras/utils.normalize(X, axis=1)
#    return X
#

#--------------------------------------------------------------------------------------------------------------------------
# TEST, A single Layer NN, with the identity (i.e. linear) as the activation function to test basic set up (i.e. a persistance forecast)
if True:
   print(' ')
   print(' ')
   print('------------------------------------------------------------------')
   print(' ')
   print('Running single layer NN with identity function, i.e. perstistance forecast')  

   model_linear = Sequential()
   model_linear.add(Activation('linear'))
   
   model_linear.compile(loss='mean_squared_error',
                        optimizer='adam',
                        metrics=['accuracy'])

   model_linear_training=model_linear.fit(X, Y, epochs=20, validation_split=0.3, callbacks=[early_stopping_monitor], verbose=True)

#--------------------------------------------------------------------------------------------------------------------------
# NN with multiple dense layers
if True:
   print(' ')
   print(' ')
   print('------------------------------------------------------------------')
   print(' ')
   print('Running multi-layer NN with 3 dense layers and relu activation function')  

   model_dense3 = Sequential()

   # define first hidden layer
   model_dense3.add(Dense(100))
   model_dense3.add(Activation('relu'))
   #add second hidden layer 
   model_dense3.add(Dense(100))
   model_dense3.add(Activation('relu'))
   #add third hidden layer
   model_dense3.add(Dense(100))
   model_dense3.add(Activation('relu'))
   #ouput layer, should have linear activation function suitable for a regression problem
   model_dense3.add(Dense(1))
   model_dense3.add(Activation('linear'))

   model_dense3.compile(loss='mean_squared_error',
                        optimizer='adam',
                        metrics=['accuracy'])

   model_dense3_training=model_dense3.fit(X, Y, epochs=20, validation_split=0.3,callbacks=[early_stopping_monitor], verbose=True)


#--------------------------------------------------------------------------------------------------------------------------
# NN with multiple dense layers
if True:
   print(' ')
   print(' ')
   print('------------------------------------------------------------------')
   print(' ')
   print('Running multi-layer NN with 5 dense layers and relu activation function')  

   model_dense5 = Sequential()

   # define first hidden layer
   model_dense5.add(Dense(100))
   model_dense5.add(Activation('relu'))
   #add second hidden layer 
   model_dense5.add(Dense(100))
   model_dense5.add(Activation('relu'))
   #add third hidden layer
   model_dense5.add(Dense(100))
   model_dense5.add(Activation('relu'))
   #add fourth hidden layer
   model_dense5.add(Dense(100))
   model_dense5.add(Activation('relu'))
   #add fifth hidden layer
   model_dense5.add(Dense(100))
   model_dense5.add(Activation('relu'))
   #ouput layer, should have linear activation function suitable for a regression problem
   model_dense5.add(Dense(1))
   model_dense5.add(Activation('linear'))

   model_dense5.compile(loss='mean_squared_error',
                        optimizer='adam',
                        metrics=['accuracy'])

   model_dense5_training=model_dense5.fit(X, Y, epochs=20, validation_split=0.3,callbacks=[early_stopping_monitor], verbose=True)



#--------------------------------------------------------------------------------------------------------------------------
# Plot the validation results against epochs for each model
print(' ')
print(' ')
print('------------------------------------------------------------------')
print(' ')
print('Plotting validation output against epohcs for various networks')  

# Create the plot
fig=plt.figure()
plt.plot(model_linear_training.history['val_loss'], 'r', label='linear')
plt.plot(model_dense3_training.history['val_loss'], 'b', label='dense3')
plt.plot(model_dense5_training.history['val_loss'], 'g', label='dense5')
plt.xlabel('Epochs')
plt.ylabel('Validation score')
plt.legend()
plt.savefig('validation_against_epochs.png')
