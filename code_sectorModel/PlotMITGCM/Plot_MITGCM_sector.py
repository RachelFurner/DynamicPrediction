#!/usr/bin/env python
# coding: utf-8
# 
# Script written by Rachel Furner
# Plots instantaneous fields from MITgcm dataset, for 
# various cross sections

print('import packages')
import sys
sys.path.append('../Tools')
import Model_Plotting as SecPlt
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from netCDF4 import Dataset

#---------------
# Set variables
#---------------
plt.rcParams.update({'font.size': 10})
plt.rc('font', family='sans serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')

time = 2500
point = [ 2, 8, 6]

datadir  = '/data/hpcdata/users/racfur/MITgcm/verification/MundaySectorConfig_2degree/runs/100yrs/mnc_test_0002/'
data_filename=datadir + 'cat_tave.nc'

rootdir = '../../../MITGCM_Analysis_Sector/'

level = point[0]
y_coord = point[1]
x_coord = point[2]

#-----------------------------------------------
# Read in netcdf file and get shape
#-----------------------------------------------
print('reading in ds')
ds = xr.open_dataset(data_filename)
da_T_unmasked = ds['Ttave'][time:time+2,:,:,:]
da_Mask = ds['Mask']
# Apply the Mask
da_T = np.where(da_Mask==1, da_T_unmasked, np.nan)
da_X = ds['X']
da_Y = ds['Y']
da_Z = ds['Z']

z_size = da_T.shape[1]
y_size = da_T.shape[2]
x_size = da_T.shape[3]

print('da_T.shape')
print(da_T.shape)

#--------------------------------------------------
# Set up bits to plot diffs between time t and t+1
#--------------------------------------------------
lat_arange = [0, 8.36, 15.5, 21.67, 27.25, 32.46, 37.5, 42.54, 47.75, 53.32, 59.5, 66.64, 75.5]
lon_arange = [0, 4.5, 9.5]
depth_arange = [0, 7, 15, 21, 28, 37, da_Z.values.shape[0]-1]

lat_label = 'Latitude ('+u'\xb0'+' N)'
lon_label = 'Longitude ('+u'\xb0'+' E)'
depth_label = 'Depth (m)'

cbar_label = 'Temperature ('+u'\xb0'+'C)'
cbar_diff_label = 'Temperature Change ('+u'\xb0'+'C)'

min_value = min( np.nanmin(da_T[0:,:,x_coord]), np.nanmin(da_T[1:,:,x_coord]),
                 np.nanmin(da_T[0,level,:,:]), np.nanmin(da_T[1,level,:,:]),
                 np.nanmin(da_T[0,:,y_coord,:]), np.nanmin(da_T[0,:,y_coord,:]) )

max_value = max( np.nanmax(da_T[0:,:,x_coord]), np.amax(da_T[1:,:,x_coord]),
                 np.nanmax(da_T[0,level,:,:]), np.amax(da_T[1,level,:,:]),
                 np.nanmax(da_T[0,:,y_coord,:]), np.amax(da_T[0,:,y_coord,:]) )

#diff_min_value = -max( abs(np.nanmin(da_T[0,level,:,:]-da_T[1,level,:,:])), abs(np.nanmax(da_T[0,level,:,:]-da_T[1,level,:,:])),
#                       abs(np.nanmin(da_T[0:,:,x_coord]-da_T[1:,:,x_coord])), abs(np.nanmax(da_T[0:,:,x_coord]-da_T[1:,:,x_coord])),
#                       abs(np.nanmin(da_T[0,:,y_coord,:]-da_T[1,:,y_coord,:])), abs(np.nanmax(da_T[0,:,y_coord,:]-da_T[1,:,y_coord,:])) )
#
#diff_max_value =  max( abs(np.nanmin(da_T[0,level,:,:]-da_T[1,level,:,:])), abs(np.nanmax(da_T[0,level,:,:]-da_T[1,level,:,:])),
#                       abs(np.nanmin(da_T[0:,:,x_coord]-da_T[1:,:,x_coord])), abs(np.nanmax(da_T[0:,:,x_coord]-da_T[1:,:,x_coord])),
#                       abs(np.nanmin(da_T[0,:,y_coord,:]-da_T[1,:,y_coord,:])), abs(np.nanmax(da_T[0,:,y_coord,:]-da_T[1,:,y_coord,:])) )

diff_min_value = -.001
diff_max_value = .001

#-------------------
# Plot depth fields
#-------------------

### Fig 1a ###
fig = plt.figure( figsize=(2.5, 4.5), dpi=300 )
ax1 = fig.add_subplot(1, 1, 1)
im1 = ax1.pcolormesh(da_T[0,level,:,:], vmin=min_value, vmax=max_value, edgecolors='face', snap=True)
ax1.set_xlabel(lon_label)
ax1.set_ylabel(lat_label)
ax1.set_xticks(lon_arange)
ax1.set_xticklabels(np.round(da_X.values[np.array(lon_arange).astype(int)], decimals=-1).astype(int)) 
ax1.set_yticks(lat_arange)
ax1.set_yticklabels(np.round(da_Y.values[np.array(lat_arange).astype(int)], decimals=-1).astype(int)) 
plt.text(-0.1, 0.86, '(a)', transform=fig.transFigure)
plt.savefig(rootdir+'PLOTS/fig01a.eps', format='eps', bbox_inches = 'tight', pad_inches = 0.1)

### Fig 1b ###
fig = plt.figure( figsize=(2.5, 4.5), dpi=300 )
ax2 = fig.add_subplot(1, 1, 1)
im2 = ax2.pcolormesh(da_T[0,level,:,:]-da_T[1,level,:,:], vmin=diff_min_value, vmax=diff_max_value, 
                     cmap='bwr', edgecolors='face', snap=True)
ax2.set_xlabel(lon_label)
ax2.set_ylabel(lat_label)
ax2.set_xticks(lon_arange)
ax2.set_xticklabels(np.round(da_X.values[np.array(lon_arange).astype(int)], decimals=-1).astype(int)) 
ax2.set_yticks(lat_arange)
ax2.set_yticklabels(np.round(da_Y.values[np.array(lat_arange).astype(int)], decimals=-1).astype(int)) 
plt.text(-0.1, 0.86, '(b)', transform=fig.transFigure)
plt.savefig(rootdir+'PLOTS/fig01b.eps', format='eps', bbox_inches = 'tight', pad_inches = 0.1)

#-----------------------------
# Plot x-const cross sections
#-----------------------------

### Fig 1c ###
fig = plt.figure( figsize=(3.6, 2.0), dpi=300 )
ax5 = fig.add_subplot(1, 1, 1)
im5 = ax5.pcolormesh(da_T[0,:,:,x_coord], vmin=min_value, vmax=max_value, edgecolors='face', snap=True)
ax5.invert_yaxis()
ax5.set_xlabel(lat_label)
ax5.set_ylabel(depth_label)
ax5.set_xticks(lat_arange)
ax5.set_xticklabels(np.round(da_Y.values[np.array(lat_arange).astype(int)], decimals=-1).astype(int)) 
ax5.set_yticks(depth_arange)
ax5.set_yticklabels(da_Z.values[np.array(depth_arange)].astype(int))
plt.text(-0.055, 0.86, '(c)', transform=fig.transFigure)
plt.savefig(rootdir+'PLOTS/fig01c.eps', format='eps', bbox_inches = 'tight', pad_inches = 0.1)

### Fig 1d ###
fig = plt.figure( figsize=(3.6, 2.0), dpi=300 )
ax6 = fig.add_subplot(1, 1, 1)
im6 = ax6.pcolormesh(da_T[0,:,:,x_coord]-da_T[1,:,:,x_coord], vmin=diff_min_value, vmax=diff_max_value,
                     cmap='bwr', edgecolors='face', snap=True)
ax6.invert_yaxis()
ax6.set_xlabel(lat_label)
ax6.set_ylabel(depth_label)
ax6.set_xticks(lat_arange)
ax6.set_xticklabels(np.round(da_Y.values[np.array(lat_arange).astype(int)], decimals=-1).astype(int)) 
ax6.set_yticks(depth_arange)
ax6.set_yticklabels(da_Z.values[np.array(depth_arange)].astype(int))
plt.text(-0.055, 0.86, '(d)', transform=fig.transFigure)
plt.savefig(rootdir+'PLOTS/fig01d.eps', format='eps', bbox_inches = 'tight', pad_inches = 0.1)
 
#-----------------------------
# Plot y-const cross-sections
#-----------------------------

### fig 1e ###
fig = plt.figure( figsize=(3.6, 2.0), dpi=300 )
ax3 = fig.add_subplot(1, 1, 1)
im3 = ax3.pcolormesh(da_T[0,:,y_coord,:], vmin=min_value, vmax=max_value, edgecolors='face', snap=True)
ax3.invert_yaxis()
ax3.set_xlabel(lon_label)
ax3.set_ylabel(depth_label)
ax3.set_xticks(lon_arange)
ax3.set_xticklabels(np.round(da_X.values[np.array(lon_arange).astype(int)], decimals=0 ).astype(int)) 
ax3.set_yticks(depth_arange)
ax3.set_yticklabels(da_Z.values[np.array(depth_arange)].astype(int))
plt.text(-0.055, 0.86, '(e)', transform=fig.transFigure)
plt.savefig(rootdir+'PLOTS/fig01e.eps', format='eps', bbox_inches = 'tight', pad_inches = 0.1)

### fig 1f ###
fig = plt.figure( figsize=(3.6, 2.0), dpi=300 )
ax4 = fig.add_subplot(1, 1, 1)
im4 = ax4.pcolormesh(da_T[0,:,y_coord,:]-da_T[1,:,y_coord,:], vmin=diff_min_value, vmax=diff_max_value,
                     cmap ='bwr', edgecolors='face', snap=True)
ax4.invert_yaxis()
ax4.set_xlabel(lon_label)
ax4.set_ylabel(depth_label)
ax4.set_xticks(lon_arange)
ax4.set_xticklabels(np.round(da_X.values[np.array(lon_arange).astype(int)], decimals=0 ).astype(int)) 
ax4.set_yticks(depth_arange)
ax4.set_yticklabels(da_Z.values[np.array(depth_arange)].astype(int))
plt.text(-0.055, 0.86, '(f)', transform=fig.transFigure)
plt.savefig(rootdir+'PLOTS/fig01f.eps', format='eps', bbox_inches = 'tight', pad_inches = 0.1)

#-----------------
# Plot Colourbars 
#-----------------

fig = plt.figure( figsize=(3, .2), dpi=300 )
cb1axes = fig.add_axes([0.05, 0.05, 0.9, 0.9 ]) 
cb1=plt.colorbar(im1, ax=ax1, orientation='horizontal', cax=cb1axes)
cb1.set_label(cbar_label)    
plt.savefig(rootdir+'PLOTS/fig01_cb.eps', format='eps', bbox_inches = 'tight', pad_inches = 0.1)

fig = plt.figure( figsize=(3, .2), dpi=300 )
cb2axes = fig.add_axes([0.05, 0.05, 0.9, 0.9 ]) 
cb2=plt.colorbar(im2, ax=ax2, orientation='horizontal', cax=cb2axes, extend='both')
cb2.set_label(cbar_diff_label)    
cb2.formatter.set_powerlimits((-2, 2))
cb2.update_ticks()
plt.savefig(rootdir+'PLOTS/fig01_cbdiff.eps', format='eps', bbox_inches = 'tight', pad_inches = 0.1)


