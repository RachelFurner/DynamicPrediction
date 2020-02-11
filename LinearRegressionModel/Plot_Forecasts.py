#!/usr/bin/env python
# coding: utf-8

print('import packages')
import sys
sys.path.append('/data/hpcdata/users/racfur/DynamicPrediction/code_git/')
from Tools import CreateExpName as cn
from Tools import Model_Plotting as rfplt

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from skimage.util import view_as_windows
import os
import xarray as xr
import pickle

#----------------------------
# Set variables for this run
#----------------------------

for_len_yrs = 100    # forecast length in years

x_points=[5,5,5]
y_points=[5,10,15]
z_points=[3,3,3]

start = 200
for_len = int(for_len_yrs * 12)

#-----------------------------------------------
# Read in netcdf file for 'truth' and get shape
#-----------------------------------------------
print('reading in ds')
DIR  = '/data/hpcdata/users/racfur/MITGCM_OUTPUT/20000yr_Windx1.00_mm_diag/'
if for_len_yrs < 500:
   data_filename=DIR+'cat_tave_500yrs_SelectedVars.nc'
else:
   data_filename=DIR+'cat_tave_5000yrs_SelectedVars.nc'
ds = xr.open_dataset(data_filename)
da_T=ds['Ttave'][start:start+for_len_yrs*12]   

z_size = da_T.shape[1]
y_size = da_T.shape[2]
x_size = da_T.shape[3]

print('da_T.shape')
print(da_T.shape)

#----------------------
# Read in predictions
#----------------------

pred_dir = '/data/hpcdata/users/racfur/DynamicPrediction/RegressionOutputs/ITERATED_PREDICTION_ARRAYS/'

predictions1 = np.zeros((for_len+1, z_size, y_size, x_size))
pred1_filename = pred_dir+ 'lr_3dLatDepUVSalEtaPolyDeg2_predictions.npy'
predictions1 = np.load(pred1_filename)

predictions2 = np.zeros((for_len+1, z_size, y_size, x_size))
pred2_filename = pred_dir+ 'lr_2dLatDepUVSalEtaPolyDeg2_predictions.npy'
predictions2 = np.load(pred2_filename)

predictions3 = np.zeros((for_len+1, z_size, y_size, x_size))
pred3_filename = pred_dir+ 'lr_3dDepUVSalEtaPolyDeg2_predictions.npy'
predictions3 = np.load(pred3_filename)

predictions4 = np.zeros((for_len+1, z_size, y_size, x_size))
pred4_filename = pred_dir+ 'lr_3dLatUVSalEtaPolyDeg2_predictions.npy'
predictions4 = np.load(pred4_filename)

predictions5 = np.zeros((for_len+1, z_size, y_size, x_size))
pred5_filename = pred_dir+ 'lr_3dLatDepSalEtaPolyDeg2_predictions.npy'
predictions5 = np.load(pred5_filename)

predictions6 = np.zeros((for_len+1, z_size, y_size, x_size))
pred6_filename = pred_dir+ 'lr_3dLatDepUVEtaPolyDeg2_predictions.npy'
predictions6 = np.load(pred6_filename)

predictions7 = np.zeros((for_len+1, z_size, y_size, x_size))
pred7_filename = pred_dir+ 'lr_3dLatDepUVSalPolyDeg2_predictions.npy'
predictions7 = np.load(pred7_filename)

#---------------------------
# Set persistence forecasts
#---------------------------
print('Set perstistence forecast')
persistence = np.zeros((for_len_yrs*12,len(x_points)))  # assume 1 month forecast period - doesn't matter if this is actually too long
for point in range(len(x_points)):
   persistence[:,point] = da_T[0,z_points[point],y_points[point],x_points[point]]

#-------------------------
# Plot spatial plots....
#-------------------------
fig, ax, im = rfplt.plot_depth_fld(da_T[0,:,:,:], 5, title=None, min_value=None, max_value=None)
plt.show()

#-------------------------
# Plot spatial plots....
#-------------------------
fig = rfplt.plot_depth_fld_diff(da_T[0,:,:,:], da_T[5,:,:,:], 5, title=None)
plt.show()

#---------------------------------------------------------------------------
# Plot the predictions against da_T and persistence for a variety of points
#---------------------------------------------------------------------------
for point in range(len(x_points)):
   x = x_points[point]
   y = y_points[point]
   z = z_points[point]

   fig = plt.figure(figsize=(15 ,8))

   ax1=plt.subplot(311)
   plt_len=int(10 * 12)
   ax1.plot(da_T[:plt_len,z,y,x])
   bot, top = ax1.get_ylim()
   ax1.plot(persistence[:plt_len,point])
   ax1.plot(predictions1[:plt_len,z,y,x])
   ax1.plot(predictions2[:plt_len,z,y,x])
   ax1.plot(predictions3[:plt_len,z,y,x])
   ax1.plot(predictions4[:plt_len,z,y,x])
   ax1.plot(predictions5[:plt_len,z,y,x])
   ax1.plot(predictions6[:plt_len,z,y,x])
   ax1.plot(predictions7[:plt_len,z,y,x])
   #ax1.set_ylim(9, 9.4)
   ax1.set_ylim(bot-0.1*abs(bot), top+0.1*abs(top))
   ax1.set_ylabel('Temperature')
   ax1.set_xlabel('No of months')
   ax1.set_title('10 years')
   ax1.legend(['da_T (truth)', 'persistence', 'full_lr_model', '2d', 'no_Lat', 'no_depth', 'no_currents', 'no_sal', 'no_eta']) 

   ax2=plt.subplot(312)
   plt_len=int(100 * 12)
   ax2.plot(da_T[:plt_len,z,y,x])
   bot, top = ax2.get_ylim()
   ax2.plot(persistence[:plt_len,point])
   ax2.plot(predictions1[:plt_len,z,y,x])
   ax2.plot(predictions2[:plt_len,z,y,x])
   ax2.plot(predictions3[:plt_len,z,y,x])
   ax2.plot(predictions4[:plt_len,z,y,x])
   ax2.plot(predictions5[:plt_len,z,y,x])
   ax2.plot(predictions6[:plt_len,z,y,x])
   ax2.plot(predictions7[:plt_len,z,y,x])
   ax2.set_ylabel('Temperature')
   ax2.set_xlabel('No of forecast steps')
   ax2.set_ylim(9, 9.4)
   ax2.set_title('100 years')
   ax2.legend(['da_T (truth)', 'persistence', 'full_lr_model', '2d', 'no_Lat', 'no_depth', 'no_currents', 'no_sal', 'no_eta']) 

   ax3=plt.subplot(313)
   plt_len=for_len
   ax3.plot(da_T[:plt_len,z,y,x])
   bot, top = ax3.get_ylim()
   ax3.plot(persistence[:plt_len,point])
   ax3.plot(predictions1[:plt_len,z,y,x])
   ax3.plot(predictions2[:plt_len,z,y,x])
   ax3.plot(predictions3[:plt_len,z,y,x])
   ax3.plot(predictions4[:plt_len,z,y,x])
   ax3.plot(predictions5[:plt_len,z,y,x])
   ax3.plot(predictions6[:plt_len,z,y,x])
   ax3.plot(predictions7[:plt_len,z,y,x])
   ax3.set_ylabel('Temperature')
   ax3.set_xlabel('No of months')
   ax3.set_ylim(9, 9.4)
   ax3.set_title(str(for_len_yrs)+' years')
   ax3.legend(['da_T (truth)', 'persistence', 'full_lr_model', '2d', 'no_Lat', 'no_depth', 'no_currents', 'no_sal', 'no_eta']) 

   #plt.subplots_adjust(hspace=0.4, top=0.9, bottom=0.12, left=0.08, right=0.85)
   plt.subplots_adjust(hspace=0.4)
   #plt.suptitle('Timeseries at point '+str(x)+','+str(y)+','+str(z)+' from the simulator (da_T),\n and as predicted by the regression model, and persistence', fontsize=15, x=0.47, y=0.98)

   plt.savefig('/data/hpcdata/users/racfur/DynamicPrediction/RegressionOutputs/PLOTS/multimodel_'+str(x)+'.'+str(y)+'.'+str(z)+'TruthvsPredict_Timeseries.png', bbox_inches = 'tight', pad_inches = 0.1)
   plt.close()

