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

plt.rcParams.update({'font.size': 14})

#################################
# Plotting spatial depth fields #
#################################

def plot_depth_ax(ax, field, x_labels, y_labels, depth_labels, min_value=None, max_value=None, cmap=None):

    # Assumes field is (z, y, x)
    if not min_value:
       min_value = np.nanmin(field[:,:])  # Lowest value
    if not max_value:
       max_value = np.nanmax(field[:,:])   # Highest value

    im = ax.pcolormesh(field[:,:], vmin=min_value, vmax=max_value, cmap=cmap)
    ax.set_xlabel('x position (km)')
    ax.set_ylabel('y position (km)')
   
    # Give axis ticks in lat/lon/depth
    x_arange = [0,40,80,120,159,199,239]  # np.arange(0,x_labels.size,40)
    ax.set_xticks(x_arange)
    ax.set_xticklabels(np.round(x_labels[np.array(x_arange).astype(int)]/1000, decimals=0).astype(int)) 
    y_arange = [0,25,50,75,95]    #np.arange(0,y_labels.size,25) 
    ax.set_yticks(y_arange)
    ax.set_yticklabels(np.round(y_labels[np.array(y_arange).astype(int)]/1000, decimals=0).astype(int)) 
 
    return(ax, im)

def plot_depth_fld(field, field_name, level, x_labels, y_labels, depth_labels, 
                   title=None, min_value=None, max_value=None, diff=False, cmap=None): 
    
    # Create a figure
    fig = plt.figure(figsize=(8,4))  #(24,10))
    ax = plt.subplot(111)
    if diff:
       if min_value==None:
          min_value = - max( abs(np.nanmin(field[:,:])), abs(np.nanmax(field[:,:])) )
       if max_value==None:
          max_value =   max( abs(np.nanmin(field[:,:])), abs(np.nanmax(field[:,:])) )
       if cmap==None:
          cmap = 'bwr'
    else:
       if cmap==None:
          cmap = 'viridis'
    ax, im = plot_depth_ax(ax, field, x_labels, y_labels, depth_labels, min_value, max_value, cmap)

    ax.set_title(str(field_name)+' at '+str(int(depth_labels[level]))+'m depth')

    # Add a color bar
    cb=plt.colorbar(im, ax=(ax), shrink=0.9, anchor=(0.5, 1.3))

    if title:
       plt.suptitle(title, fontsize=14)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.01,hspace=0.01)

    return(fig, ax, im)

def plot_depth_fld_diff(field1, field1_name, field2, field2_name, level, x_labels, y_labels, depth_labels, 
                        title=None, flds_min_value=None, flds_max_value=None, diff_min_value=None, diff_max_value=None):

    if flds_min_value == None: 
       flds_min_value = min( np.nanmin(field1[:,:]), np.nanmin(field2[:,:]) )
    if flds_max_value == None: 
       flds_max_value = max( np.nanmax(field1[:,:]), np.amax(field2[:,:]) )

    if diff_min_value == None: 
       diff_min_value = -max( abs(np.nanmin(field1[:,:]-field2[:,:])), abs(np.nanmax(field1[:,:]-field2[:,:])) )
    if diff_max_value == None: 
       diff_max_value =  max( abs(np.nanmin(field1[:,:]-field2[:,:])), abs(np.nanmax(field1[:,:]-field2[:,:])) )
 
    fig = plt.figure(figsize=(8,15))
    ax1 = plt.subplot(311)
    ax2 = plt.subplot(312)
    ax3 = plt.subplot(313)
    ax1, im1 = plot_depth_ax(ax1, field1, x_labels, y_labels, depth_labels, flds_min_value, flds_max_value, cmap='viridis')
    ax2, im2 = plot_depth_ax(ax2, field2, x_labels, y_labels, depth_labels, flds_min_value, flds_max_value, cmap='viridis')
    ax3, im3 = plot_depth_ax(ax3, field1-field2, x_labels, y_labels, depth_labels, diff_min_value, diff_max_value, cmap='bwr')

    ax1.set_title(str(field1_name)+' at '+str(int(depth_labels[level]))+'m depth')
    ax2.set_title(str(field2_name)+' at '+str(int(depth_labels[level]))+'m depth')
    ax3.set_title('The Difference')

    cb1axes = fig.add_axes([0.92, 0.42, 0.03, 0.52]) 
    cb3axes = fig.add_axes([0.92, 0.08, 0.03, 0.20]) 
    cb1=plt.colorbar(im1, ax=(ax1,ax2), orientation='vertical', cax=cb1axes)
    cb3=plt.colorbar(im3, ax=ax3, orientation='vertical', cax=cb3axes)
 
    if title:
       plt.suptitle(title, fontsize=14)

    plt.tight_layout()
    plt.subplots_adjust(hspace = 0.3, right=0.9, bottom=0.07, top=0.95)
    #plt.subplots_adjust(wspace=0.01, hspace = 0.4, bottom=0.15)

    return(fig)

##########################################################################################
## Plot cross sections for y=const at a few discrete time points - start, middle and end #
##########################################################################################

def plot_yconst_crss_sec_ax(ax, field, y, x_labels, y_labels, depth_labels, min_value=None, max_value=None, cmap=None):

    # Assumes field is (z, y, x)

    if not min_value:
       min_value = np.nanmin(field[:,y,:])  # Lowest value
    if not max_value:
       max_value = np.nanmax(field[:,y,:])   # Highest value

    im = ax.pcolormesh(field[:,y,:], vmin=min_value, vmax=max_value, cmap=cmap)
    ax.invert_yaxis()
    ax.set_xlabel('x position (km)')
    ax.set_ylabel('depth level')
   
    # Give axis ticks in lat/lon/depth
    x_arange = [0,40,80,120,159,199,239]  # np.arange(0,x_labels.size,40)
    ax.set_xticks(x_arange)
    ax.set_xticklabels(np.round(x_labels[np.array(x_arange).astype(int)]/1000, decimals=0 ).astype(int)) 
    depth_arange = [0, 4, 9, 14, 19, 24, 29, 34] 
    ax.set_yticks(depth_arange)
    ax.set_yticklabels(depth_labels[np.array(depth_arange)].astype(int)) 
 
    return(ax, im)

def plot_yconst_crss_sec(field, field_name, y, x_labels, y_labels, depth_labels, title=None, min_value=None, max_value=None, diff=False, cmap=None):
    
    # Create a figure
    fig = plt.figure(figsize=(9,4))
    ax = plt.subplot(111)
    if diff:
       min_value = - max( abs(np.nanmin(field[:,y,:])), abs(np.nanmax(field[:,y,:])) )
       max_value =   max( abs(np.nanmin(field[:,y,:])), abs(np.nanmax(field[:,y,:])) )
       if cmap==None:
          cmap = 'bwr'
    else:
       if cmap==None:
          cmap = 'viridis'
    ax, im = plot_yconst_crss_sec_ax(ax, field, y, x_labels, y_labels, depth_labels, min_value, max_value, cmap)

    ax.set_title(str(field_name)+' at '+str(int(y_labels[y]/1000))+'km in y direction')

    # Add a color bar
    cb=plt.colorbar(im, ax=(ax), shrink=0.9)
    
    if title:
       plt.suptitle(title, fontsize=14)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.01,hspace=0.01)

    return(fig, ax, im)

def plot_yconst_crss_sec_diff(field1, field1_name, field2, field2_name, y, x_labels, y_labels, depth_labels, title=None):
    
    flds_min_value = min( np.nanmin(field1[:,y,:]), np.nanmin(field2[:,y,:]) )
    flds_max_value = max( np.nanmax(field1[:,y,:]), np.amax(field2[:,y,:]) )

    diff_min_value = -max( abs(np.nanmin(field1[:,y,:]-field2[:,y,:])), abs(np.nanmax(field1[:,y,:]-field2[:,y,:])) )
    diff_max_value =  max( abs(np.nanmin(field1[:,y,:]-field2[:,y,:])), abs(np.nanmax(field1[:,y,:]-field2[:,y,:])) )
 
    fig = plt.figure(figsize=(9,14))
    ax1 = plt.subplot(311)
    ax2 = plt.subplot(312)
    ax3 = plt.subplot(313)
    ax1, im1 = plot_yconst_crss_sec_ax(ax1, field1, y, x_labels, y_labels, depth_labels, flds_min_value, flds_max_value)
    ax2, im2 = plot_yconst_crss_sec_ax(ax2, field2, y, x_labels, y_labels, depth_labels, flds_min_value, flds_max_value)
    ax3, im3 = plot_yconst_crss_sec_ax(ax3, field1-field2, y, x_labels, y_labels, depth_labels, diff_min_value, diff_max_value, cmap='bwr')

    ax1.set_title(str(field1_name)+' at '+str(int(y_labels[y]/1000))+'km in y direction')
    ax2.set_title(str(field2_name)+' at '+str(int(y_labels[y]/1000))+'km in y direction') 
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

def plot_xconst_crss_sec_ax(ax, field, x, x_labels, y_labels, depth_labels, min_value=None, max_value=None, cmap=None):

    # Assumes field is (z, y, x)

    if not min_value:
       min_value = np.nanmin(field[:,:,x])  # Lowest value
    # Need to work this bit out....
    #if var[0] == 'S':
    #    min_value = 33.5
    if not max_value:
       max_value = np.nanmax(field[:,:,x])   # Highest value

    im = ax.pcolormesh(field[:,:,x], vmin=min_value, vmax=max_value, cmap=cmap)
    ax.invert_yaxis()
    ax.invert_xaxis()
    ax.set_xlabel('y position (km)')
    ax.set_ylabel('depth levels')

    ## Give axis ticks in lat/lon/depth
    y_arange = [0,25,50,75,95]    #np.arange(0,y_labels.size,25) 
    ax.set_xticks(y_arange)
    ax.set_xticklabels(np.round(y_labels[np.array(y_arange).astype(int)]/1000, decimals=0).astype(int)) 
    depth_arange = [0, 4, 9, 14, 19, 24, 29, 34] 
    ax.set_yticks(depth_arange)
    ax.set_yticklabels(depth_labels[np.array(depth_arange)].astype(int)) 
    
    return(ax, im)

def plot_xconst_crss_sec(field, field_name, x, x_labels, y_labels, depth_labels, title=None, min_value=None, max_value=None, diff=False, cmap=None):
    
    # Create a figure
    fig = plt.figure(figsize=(9,4))
    ax = plt.subplot(111)
    if diff:
       if min_value == None:
          min_value = - max( abs(np.nanmin(field[:,:,x])), abs(np.nanmax(field[:,:,x])) )
          max_value =   max( abs(np.nanmin(field[:,:,x])), abs(np.nanmax(field[:,:,x])) )
       if cmap==None:
          cmap = 'bwr'
    else:
       if cmap==None:
          cmap = 'viridis'
    ax, im = plot_xconst_crss_sec_ax(ax, field, x, x_labels, y_labels, depth_labels, min_value, max_value, cmap)

    ax.set_title(str(field_name)+' at '+str(int(x_labels[x]/1000))+'km in x direction')

    # Add a color bar
    cb=plt.colorbar(im, ax=(ax), shrink=0.9)
    
    if title:
       plt.suptitle(title, fontsize=14)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.01,hspace=0.01)

    return(fig, ax, im)

def plot_xconst_crss_sec_diff(field1, field1_name, field2, field2_name, x, x_labels, y_labels, depth_labels, title=None):
    
    flds_min_value = min( np.nanmin(field1[:,:,x]), np.nanmin(field2[:,:,x]) )
    flds_max_value = max( np.nanmax(field1[:,:,x]), np.amax(field2[:,:,x]) )

    diff_min_value = -max( abs(np.nanmin(field1[:,:,x]-field2[:,:,x])), abs(np.nanmax(field1[:,:,x]-field2[:,:,x])) )
    diff_max_value =  max( abs(np.nanmin(field1[:,:,x]-field2[:,:,x])), abs(np.nanmax(field1[:,:,x]-field2[:,:,x])) )
 
    fig = plt.figure(figsize=(9,14))
    ax1 = plt.subplot(311)
    ax2 = plt.subplot(312)
    ax3 = plt.subplot(313)
    ax1, im1 = plot_xconst_crss_sec_ax(ax1, field1, x, x_labels, y_labels, depth_labels, flds_min_value, flds_max_value)
    ax2, im2 = plot_xconst_crss_sec_ax(ax2, field2, x, x_labels, y_labels, depth_labels, flds_min_value, flds_max_value)
    ax3, im3 = plot_xconst_crss_sec_ax(ax3, field1-field2, x, x_labels, y_labels, depth_labels, diff_min_value, diff_max_value, cmap='bwr')

    ax1.set_title(str(field1_name)+' at '+str(int(x_labels[x]/1000))+'km in x direction')
    ax2.set_title(str(field2_name)+' at '+str(int(x_labels[x]/1000))+'km in x direction') 
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
              end=min(np.nanmin(ii),length)
          ax.plot(dataset[:end])

      elif len(point) == 1:
          ii = np.argwhere(np.isnan(dataset[:, point[0]]))
          if ii.shape[0]==0:
              end=length
          else: 
              end=min(np.nanmin(ii),length)
          ax.plot(dataset[:end, point[0]])

      elif len(point) == 2:
          ii = np.argwhere(np.isnan(dataset[:, point[0], point[1]]))
          if ii.shape[0]==0:
              end=length
          else: 
              end=min(np.nanmin(ii),length)
          ax.plot(dataset[:end, point[0], point[1]])

      elif len(point) == 3:
          ii = np.argwhere(np.isnan(dataset[:, point[0], point[1], point[2]]))
          if ii.shape[0]==0:
              end=length
          else: 
              end=min(np.nanmin(ii),length)
          ax.plot(dataset[:end, point[0], point[1], point[2]])

      my_legend.append(name)
   ax.legend(my_legend)
   ax.set_ylabel('Temperature')
   ax.set_xlabel('No of days')
   ax.set_title(str(int(length/30))+' months')
   if ylim:
      ax.set_ylim(ylim)
 
   return(ax)

def plt_timeseries(point, length, datasets, ylim=None):
   
   fig = plt.figure(figsize=(15 ,3))
   ax=plt.subplot(111)
   if ylim == None:
      if len(point) == 2:
         bottom = min(next(iter(datasets.values()))[:length, point[0], point[1]])-1
         top    = max(next(iter(datasets.values()))[:length, point[0], point[1]])+1
      if len(point) == 3:
         bottom = min(next(iter(datasets.values()))[:length, point[0], point[1], point[2]])-1
         top    = max(next(iter(datasets.values()))[:length, point[0], point[1], point[2]])+1
      ylim=[bottom, top]
   ax=plt_timeseries_ax(ax, point, length, datasets, ylim=ylim)

   plt.tight_layout()
   plt.subplots_adjust(hspace = 0.5, left=0.05, right=0.95, bottom=0.15, top=0.90)

   return(fig)

def plt_2_timeseries(point, length1, length2, datasets, ylim=None):
   
   fig = plt.figure(figsize=(15 ,7))

   ax=plt.subplot(211)
   if ylim == None:
      bottom = min(datasets['True Temp'][:length1, point[0], point[1], point[2]])-1
      top    = max(datasets['True Temp'][:length1, point[0], point[1], point[2]])+1
      ylim=[bottom, top]
   ax=plt_timeseries_ax(ax, point, length1, datasets, ylim=ylim)

   ax=plt.subplot(212)
   if ylim == None:
      bottom = min(datasets['True Temp'][:length2, point[0], point[1], point[2]])-1
      top    = max(datasets['True Temp'][:length2, point[0], point[1], point[2]])+1
      ylim=[bottom, top]
   ax=plt_timeseries_ax(ax, point, length2, datasets, ylim=ylim)

   plt.tight_layout()
   plt.subplots_adjust(hspace = 0.5, left=0.05, right=0.95, bottom=0.07, top=0.95)

   return(fig)

def plt_3_timeseries(point, length1, length2, length3, datasets, ylim=None):
   
   fig = plt.figure(figsize=(15 ,11))

   ax=plt.subplot(311)
   if ylim == None:
      bottom = min(datasets['True Temp'][:length1, point[0], point[1], point[2]])-1
      top    = max(datasets['True Temp'][:length1, point[0], point[1], point[2]])+1
      ylim=[bottom, top]
   ax=plt_timeseries_ax(ax, point, length1, datasets, ylim=ylim)

   ax=plt.subplot(312)
   if ylim == None:
      bottom = min(datasets['True Temp'][:length2, point[0], point[1], point[2]])-1
      top    = max(datasets['True Temp'][:length2, point[0], point[1], point[2]])+1
      ylim=[bottom, top]
   ax=plt_timeseries_ax(ax, point, length2, datasets, ylim=ylim)

   ax=plt.subplot(313)
   if ylim == None:
      bottom = min(datasets['True Temp'][:length3, point[0], point[1], point[2]])-1
      top    = max(datasets['True Temp'][:length3, point[0], point[1], point[2]])+1
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
    plt.annotate('skew = '+str(np.round(skew(data),5)), (0.1,0.9), xycoords='figure fraction')
    return(fig)
