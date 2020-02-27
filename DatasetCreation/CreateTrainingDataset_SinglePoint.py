#!/usr/bin/env python
# coding: utf-8

#----------------------------
# Import neccessary packages
#----------------------------
import sys
sys.path.append('/data/hpcdata/users/racfur/DynamicPrediction/code_git/')
from Tools import CreateDataName as cn
from Tools import ReadRoutines as rr
from Tools import AssessModel as am

import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn import linear_model
import sklearn.metrics as metrics
from sklearn.model_selection import GridSearchCV
import torch as torch

torch.cuda.empty_cache()

import struct
print(struct.calcsize("P") * 8)
#----------------------------
# Set variables for this run
#----------------------------
run_vars={'dimension':3, 'lat':True , 'lon':False, 'dep':True , 'current':True , 'sal':True , 'eta':True , 'poly_degree':2}

DIR = '/data/hpcdata/users/racfur/MITGCM_OUTPUT/20000yr_Windx1.00_mm_diag/'
MITGCM_filename=DIR+'cat_tave_2000yrs_SelectedVars_masked.nc'

#---------------------------
# calculate other variables 
#---------------------------
data_name = cn.create_dataname(run_vars)

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
norm_inputs_tr, norm_inputs_te, norm_outputs_tr, norm_outputs_te = rr.ReadMITGCM(MITGCM_filename, 0.7, data_name, run_vars)

#---------------------------------------------------------------------------------------------------------------
# Plot inputs (temp at time t) against outputs (temp at time t+1), to see if variance changes with input values
#---------------------------------------------------------------------------------------------------------------
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

plt.savefig('/data/hpcdata/users/racfur/DynamicPrediction/INPUT_OUTPUT_ARRAYS/'+data_name+'_InputsvsOutputs', bbox_inches = 'tight', pad_inches = 0.1)
