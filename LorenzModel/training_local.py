import numpy as np
np.random.seed(42)

import math
import keras
from keras.models import Sequential
from keras import layers
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import rmsprop, SGD, Adagrad, Adadelta
import json
import matplotlib.pyplot as plt

#from eccodes import *


#### Original D&B code
#file = open('trainingdata.dat', 'r') 
#
#n_run_str  = file.readline()
#n_dummy =  eval(n_run_str)
#n_run = n_dummy[0]
#n_steps =  n_dummy[1]
#
#n_run = 2000000
#
#print n_run, n_steps 
#
#xin_train = np.zeros((n_run*8))
#yin_train = np.zeros((n_run*8))
#
#x_train = np.zeros((8*n_run,4))
#y_train = np.zeros((8*n_run,1))
#
#
## Read in each single value of x variable and its later value
#data_list_i = []
#data_list_o = []
#for i in range(n_run):
#    a_str = file.readline()
#    a = eval('[' + a_str + ']')
#    data_list_i.append(a)
#    a_str = file.readline()
#    a = eval('[' + a_str + ']')
#    data_list_o.append(a)
#
#xin_train = np.array(data_list_i)
#yin_train = np.array(data_list_o)
#yin_train[:] = yin_train[:] - xin_train[:]
#
#del(data_list_i)
#del(data_list_o)
#
##convert shape to nrun by 8, so now each row is a time step, or 8 x values...
#xin_train = xin_train.reshape((n_run, 8))
#yin_train = yin_train.reshape((n_run, 8))
#
#
## Read into the final usable array of inputs and an array of outputs, so each row is a training pair
#n_count = -1
#for i in range(n_run):
#    for j in range(8):
#        n_count = n_count+1
#        n1=(j-2)%8
#        x_train[n_count,0] = xin_train[i,n1]  
#        n2=(j-1)%8
#        x_train[n_count,1] = xin_train[i,n2]        
#        x_train[n_count,2] = xin_train[i,j]   
#        n3=(j+1)%8
#        x_train[n_count,3] = xin_train[i,n3]
#        y_train[n_count,0] = yin_train[i,j]    
#
#del(xin_train)
#del(yin_train)
#
#
##### My code.. based on above
file = open('Lorenz_training_set.txt', 'r') 

#n_run = 2000000
n_run = 20728 # this is the number of MTU's worth of data, mutilpy by 8 (no of x variables) to get number of training samples

print(n_run)

xin_train = np.zeros((n_run,8))
yin_train = np.zeros((n_run,8))

x_train = np.zeros((8*n_run,4))
y_train = np.zeros((8*n_run,1))


# Read in each single value of x variable and its later value
data_list_i = []
data_list_i2 = []
data_list_o = []
for i in range(n_run):
    a_str = file.readline()
    data_list_i2.append(a_str.split()) # dont need this data for original D&B implementation
    a_str = file.readline()
    data_list_i.append(a_str.split())
    a_str = file.readline()
    data_list_o.append(a_str.split())

xin_train = np.array(data_list_i).astype(np.float)
yin_train = np.array(data_list_o).astype(np.float)
print(xin_train.shape)
print(yin_train.shape)
yin_train[:,:] = yin_train[:,:] - xin_train[:,:]  # The y values are the difference between the now and later step, i.e. delta x

del(data_list_i)
del(data_list_i2)
del(data_list_o)


# Read into the final usable array of inputs and an array of outputs, so each row is a training pair
n_count = -1
for i in range(n_run):
    for j in range(8):
        n_count = n_count+1
        n1=(j-2)%8
        x_train[n_count,0] = xin_train[i,n1]  
        n2=(j-1)%8
        x_train[n_count,1] = xin_train[i,n2]        
        x_train[n_count,2] = xin_train[i,j]   
        n3=(j+1)%8
        x_train[n_count,3] = xin_train[i,n3]
        y_train[n_count,0] = yin_train[i,j]    

del(xin_train)
del(yin_train)

max_train = 30.0
min_train = -20.0

x_train = 2.0*(x_train-min_train)/(max_train-min_train)-1.0

file.close() 

model1 = Sequential()
model1.add(Dense(4, input_dim=4, activation='tanh'))
model1.add(Dense(20, activation='tanh'))
model1.add(Dense(20, activation='tanh'))
model1.add(Dense(1, activation='tanh'))

model1.compile(loss='mean_absolute_error', optimizer='SGD', metrics=['mae'])

# Fit the model
model1.fit(x_train, y_train, epochs=200,batch_size=128,validation_split=0.2)
model1.save_weights("./weights")


