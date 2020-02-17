#!/usr/bin/env python
# coding: utf-8

# Script to contain modules to  plots cross sections, time series etc from netcdf files 
# of the sector configuration (created through MITGCM or NN/LR methods)

# All modules expect the field for plotting to be passed (rather than a filename etc)

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from matplotlib.cm import get_cmap
from scipy.stats import skew

#################################
# Plotting spatial depth fields #
#################################

def plot_depth_ax(ax, field, level, min_value=None, max_value=None, cmap=None):

    # Assumes field is (z, y, x)
    if not min_value:
       min_value = np.amin(field[level,:,:])  # Lowest value
    # Need to work this bit out....
    #if var[0] == 'S':
    #    min_value = 33.5
    if not max_value:
       max_value = np.amax(field[level,:,:])   # Highest value

    im = ax.pcolormesh(field[level,:,:], vmin=min_value, vmax=max_value, cmap=cmap)
    ax.set_xlabel('x (\'longitude\')')
    ax.set_ylabel('y (\'latitude\')')
    
    return(ax, im)

def plot_depth_fld(field, field_name, level, title=None, min_value=None, max_value=None):
    
    # Create a figure
    fig = plt.figure(figsize=(5,9))
    ax = plt.subplot(111)
    ax, im = plot_depth_ax(ax, field, level, min_value, max_value)

    ax.set_title(str(field_name)+' at z='+str(level))

    # Add a color bar
    cb=plt.colorbar(im, ax=(ax), shrink=0.9, anchor=(0.5, 1.3))

    if title:
       plt.suptitle(title, fontsize=14)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.01,hspace=0.01)

    return(fig, ax, im)

def plot_depth_fld_diff(field1, field1_name, field2, field2_name, level, title=None):
    
    flds_min_value = min( np.amin(field1[level,:,:]), np.amin(field2[level,:,:]) )
    flds_max_value = max( np.amax(field1[level,:,:]), np.amax(field2[level,:,:]) )

    diff_min_value = -max( abs(np.amin(field1[level,:,:]-field2[level,:,:])), abs(np.amax(field1[level,:,:]-field2[level,:,:])) )
    diff_max_value =  max( abs(np.amin(field1[level,:,:]-field2[level,:,:])), abs(np.amax(field1[level,:,:]-field2[level,:,:])) )
 
    fig = plt.figure(figsize=(15,9))
    ax1 = plt.subplot(131)
    ax2 = plt.subplot(132)
    ax3 = plt.subplot(133)
    ax1, im1 = plot_depth_ax(ax1, field1, level, flds_min_value, flds_max_value)
    ax2, im2 = plot_depth_ax(ax2, field2, level, flds_min_value, flds_max_value)
    ax3, im3 = plot_depth_ax(ax3, field1-field2, level, diff_min_value, diff_max_value, cmap='bwr')

    ax1.set_title(str(field1_name)+' at z='+str(level))
    ax2.set_title(str(field2_name)+' at z='+str(level))
    ax3.set_title('The Difference')

    cb1axes = fig.add_axes([0.05, 0.035, 0.60, 0.03]) 
    cb3axes = fig.add_axes([0.73, 0.035, 0.24, 0.03]) 
    cb1=plt.colorbar(im1, ax=(ax1,ax2), cax=cb1axes, orientation='horizontal')
    cb3=plt.colorbar(im3, ax=ax3, cax=cb3axes, orientation='horizontal')

    if title:
       plt.suptitle(title, fontsize=14)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2, hspace = 0.01, bottom=0.15)

    return(fig)

##########################################################################################
## Plot cross sections for y=const at a few discrete time points - start, middle and end #
##########################################################################################

def plot_yconst_crss_sec_ax(ax, field, y, min_value=None, max_value=None, cmap=None):

    # Assumes field is (z, y, x)

    if not min_value:
       min_value = np.amin(field[:,y,:])  # Lowest value
    # Need to work this bit out....
    #if var[0] == 'S':
    #    min_value = 33.5
    if not max_value:
       max_value = np.amax(field[:,y,:])   # Highest value

    im = ax.pcolormesh(field[:,y,:], vmin=min_value, vmax=max_value, cmap=cmap)
    ax.invert_yaxis()
    ax.set_xlabel('x (\'longitude\')')
    ax.set_ylabel('z (\'depth\')')
    
    return(ax, im)

def plot_yconst_crss_sec(field, field_name, y, title=None, min_value=None, max_value=None):
    
    # Create a figure
    fig = plt.figure(figsize=(9,5))
    ax = plt.subplot(111)
    ax, im = plot_yconst_crss_sec_ax(ax, field, y, min_value, max_value)

    ax.set_title(str(field_name)+' at y='+str(y))

    # Add a color bar
    cb=plt.colorbar(im, ax=(ax), shrink=0.9)
    
    if title:
       plt.suptitle(title, fontsize=14)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.01,hspace=0.01)

    return(fig, ax, im)

def plot_yconst_crss_sec_diff(field1, field1_name, field2, field2_name, y, title=None):
    
    flds_min_value = min( np.amin(field1[:,y,:]), np.amin(field2[:,y,:]) )
    flds_max_value = max( np.amax(field1[:,y,:]), np.amax(field2[:,y,:]) )

    diff_min_value = -max( abs(np.amin(field1[:,y,:]-field2[:,y,:])), abs(np.amax(field1[:,y,:]-field2[:,y,:])) )
    diff_max_value =  max( abs(np.amin(field1[:,y,:]-field2[:,y,:])), abs(np.amax(field1[:,y,:]-field2[:,y,:])) )
 
    fig = plt.figure(figsize=(9,15))
    ax1 = plt.subplot(311)
    ax2 = plt.subplot(312)
    ax3 = plt.subplot(313)
    ax1, im1 = plot_yconst_crss_sec_ax(ax1, field1, y, flds_min_value, flds_max_value)
    ax2, im2 = plot_yconst_crss_sec_ax(ax2, field2, y, flds_min_value, flds_max_value)
    ax3, im3 = plot_yconst_crss_sec_ax(ax3, field1-field2, y, diff_min_value, diff_max_value, cmap='bwr')

    ax1.set_title(str(field1_name)+' at y='+str(y))
    ax2.set_title(str(field2_name)+' at y='+str(y))
    ax3.set_title('The Difference')

    cb1axes = fig.add_axes([0.92, 0.42, 0.03, 0.52]) 
    cb3axes = fig.add_axes([0.92, 0.08, 0.03, 0.20]) 
    cb1=plt.colorbar(im1, ax=(ax1,ax2), orientation='vertical', cax=cb1axes)
    cb3=plt.colorbar(im3, ax=ax3, orientation='vertical', cax=cb3axes)
 
    if title:
       plt.suptitle(title, fontsize=14)

    plt.tight_layout()
    plt.subplots_adjust(hspace = 0.2, right=0.9, bottom=0.07, top=0.95)

    return(fig)

#######################################################
# Plot cross section at x=const (i.e. North to South) #   
#######################################################

def plot_xconst_crss_sec_ax(ax, field, x, min_value=None, max_value=None, cmap=None):

    # Assumes field is (z, y, x)

    if not min_value:
       min_value = np.amin(field[:,:,x])  # Lowest value
    # Need to work this bit out....
    #if var[0] == 'S':
    #    min_value = 33.5
    if not max_value:
       max_value = np.amax(field[:,:,x])   # Highest value

    im = ax.pcolormesh(field[:,:,x], vmin=min_value, vmax=max_value, cmap=cmap)
    ax.invert_yaxis()
    ax.invert_xaxis()
    ax.set_xlabel('y (\'latitude\')')
    ax.set_ylabel('z (\'depth\')')
    
    return(ax, im)

def plot_xconst_crss_sec(field, field_name, x, title=None, min_value=None, max_value=None):
    
    # Create a figure
    fig = plt.figure(figsize=(9,5))
    ax = plt.subplot(111)
    ax, im = plot_xconst_crss_sec_ax(ax, field, x, min_value, max_value)

    ax.set_title(str(field_name)+' at x='+str(x))

    # Add a color bar
    cb=plt.colorbar(im, ax=(ax), shrink=0.9)
    
    if title:
       plt.suptitle(title, fontsize=14)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.01,hspace=0.01)

    return(fig, ax, im)

def plot_xconst_crss_sec_diff(field1, field1_name, field2, field2_name, x, title=None):
    
    flds_min_value = min( np.amin(field1[:,:,x]), np.amin(field2[:,:,x]) )
    flds_max_value = max( np.amax(field1[:,:,x]), np.amax(field2[:,:,x]) )

    diff_min_value = -max( abs(np.amin(field1[:,:,x]-field2[:,:,x])), abs(np.amax(field1[:,:,x]-field2[:,:,x])) )
    diff_max_value =  max( abs(np.amin(field1[:,:,x]-field2[:,:,x])), abs(np.amax(field1[:,:,x]-field2[:,:,x])) )
 
    fig = plt.figure(figsize=(9,15))
    ax1 = plt.subplot(311)
    ax2 = plt.subplot(312)
    ax3 = plt.subplot(313)
    ax1, im1 = plot_xconst_crss_sec_ax(ax1, field1, x, flds_min_value, flds_max_value)
    ax2, im2 = plot_xconst_crss_sec_ax(ax2, field2, x, flds_min_value, flds_max_value)
    ax3, im3 = plot_xconst_crss_sec_ax(ax3, field1-field2, x, diff_min_value, diff_max_value, cmap='bwr')

    ax1.set_title(str(field1_name)+' at x='+str(x))
    ax2.set_title(str(field2_name)+' at x='+str(x))
    ax3.set_title('The Difference')

    cb1axes = fig.add_axes([0.92, 0.42, 0.03, 0.52]) 
    cb3axes = fig.add_axes([0.92, 0.08, 0.03, 0.20]) 
    cb1=plt.colorbar(im1, ax=(ax1,ax2), orientation='vertical', cax=cb1axes)
    cb3=plt.colorbar(im3, ax=ax3, orientation='vertical', cax=cb3axes)
 
    if title:
       plt.suptitle(title, fontsize=14)

    plt.tight_layout()
    plt.subplots_adjust(hspace = 0.2, right=0.9, bottom=0.07, top=0.95)

    return(fig)

#######################################
# Plot time series at specific points #
#######################################

def plt_timeseries_ax(ax, point, length, datasets, ylim=None):

   my_legend=[]
   for name, dataset in datasets.items():

      if len(point) == 0:
          ii = np.argwhere(np.isnan(dataset[:]))
          if ii.shape[0]==0:
              end=length
          else: 
              end=min(np.amin(ii),length)
          ax.plot(dataset[:end])

      elif len(point) == 1:
          ii = np.argwhere(np.isnan(dataset[:, point[0]]))
          if ii.shape[0]==0:
              end=length
          else: 
              end=min(np.amin(ii),length)
          ax.plot(dataset[:end, point[0]])

      elif len(point) == 2:
          ii = np.argwhere(np.isnan(dataset[:, point[0], point[1]]))
          if ii.shape[0]==0:
              end=length
          else: 
              end=min(np.amin(ii),length)
          ax.plot(dataset[:end, point[0], point[1]])

      elif len(point) == 3:
          ii = np.argwhere(np.isnan(dataset[:, point[0], point[1], point[2]]))
          if ii.shape[0]==0:
              end=length
          else: 
              end=min(np.amin(ii),length)
          ax.plot(dataset[:end, point[0], point[1], point[2]])

      my_legend.append(name)
   ax.legend(my_legend)
   ax.set_ylabel('Temperature')
   ax.set_xlabel('No of months')
   ax.set_title(str(int(length/12))+' years')
   if ylim:
      ax.set_ylim(ylim)
 
   return(ax)

def plt_timeseries(point, length, datasets, ylim=None):
   
   fig = plt.figure(figsize=(15 ,3))

   ax=plt.subplot(111)
   ax=plt_timeseries_ax(ax, point, length, datasets, ylim=ylim)

   plt.tight_layout()
   plt.subplots_adjust(hspace = 0.5, left=0.05, right=0.95, bottom=0.15, top=0.90)

   return(fig)

def plt_3_timeseries(point, length1, length2, length3, datasets, ylim=None):
   
   fig = plt.figure(figsize=(15 ,11))

   ax=plt.subplot(311)
   bottom = min(datasets['da_T (truth)'][:length1, point[0], point[1], point[2]])-1
   top    = max(datasets['da_T (truth)'][:length1, point[0], point[1], point[2]])+1
   ylim=[bottom, top]
   ax=plt_timeseries_ax(ax, point, length1, datasets, ylim=ylim)

   ax=plt.subplot(312)
   bottom = min(datasets['da_T (truth)'][:length2, point[0], point[1], point[2]])-1
   top    = max(datasets['da_T (truth)'][:length2, point[0], point[1], point[2]])+1
   ylim=[bottom, top]
   ax=plt_timeseries_ax(ax, point, length2, datasets, ylim=ylim)

   ax=plt.subplot(313)
   bottom = min(datasets['da_T (truth)'][:length3, point[0], point[1], point[2]])-1
   top    = max(datasets['da_T (truth)'][:length3, point[0], point[1], point[2]])+1
   ylim=[bottom, top]
   ax=plt_timeseries_ax(ax, point, length3, datasets, ylim=ylim)

   plt.tight_layout()
   plt.subplots_adjust(hspace = 0.5, left=0.05, right=0.95, bottom=0.07, top=0.95)

   return(fig)

###############################
# Plot histogram of y_hat - y #
###############################
def Plot_Histogram(data, no_bins):

    fig = plt.figure(figsize=(10, 8))

    if len(data.shape) > 1:
       data = data.reshape(-1)

    plt.hist(data, bins = no_bins)
    plt.yscale('log')
    plt.annotate('skew = '+str(skew(data)), (0.15,0.85), xycoords='figure fraction')
    return(fig)
