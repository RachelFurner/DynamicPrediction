#!/usr/bin/env python
# coding: utf-8

# Script written by Rachel Furner
# Contains functions to  plots cross sections, time series etc from netcdf files 
# of the sector configuration (created from either MITGCM or LR methods)
# All functions expect the field for plotting to be passed (rather than a filename etc)

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew

#------------------------
# Set plotting variables
#------------------------

plt.rcParams.update({'font.size': 10})
plt.rc('font', family='sans serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')


#-------------------------------
# Plotting spatial depth fields 
#-------------------------------

def plot_depth_ax(fig, ax, field, level, lon_labels, lat_labels, depth_labels, text=None, min_value=None, max_value=None, cmap=None):
    # Assumes field is (z, y, x)

    if not min_value:
       min_value = np.nanmin(field[level,:,:])  # Lowest value
    if not max_value:
       max_value = np.nanmax(field[level,:,:])   # Highest value

    im = ax.pcolormesh(field[level,:,:], vmin=min_value, vmax=max_value, cmap=cmap, edgecolors='face', snap=True)
    ax.set_xlabel('Longitude ('+u'\xb0'+' E)')
    ax.set_ylabel('Latitude ('+u'\xb0'+' N)')
   
    # Give axis ticks in lat/lon/depth
    lon_arange = [0, 4.5, 9.5]
    ax.set_xticks(lon_arange)
    ax.set_xticklabels(np.round(lon_labels[np.array(lon_arange).astype(int)], decimals=-1).astype(int)) 
    lat_arange = [0, 8.36, 15.5, 21.67, 27.25, 32.46, 37.5, 42.54, 47.75, 53.32, 59.5, 66.64, 75.5]
    ax.set_yticks(lat_arange)
    ax.set_yticklabels(np.round(lat_labels[np.array(lat_arange).astype(int)], decimals=-1).astype(int)) 

    if text:
       plt.text(0.01, 0.94, text, transform=fig.transFigure)    
 
    return(ax, im)

def plot_depth_fld(field, field_name, level, lon_labels, lat_labels, depth_labels, text=None, title=None, min_value=None, max_value=None,
                   diff=False, cmap=None, cbar_label=None, Sci=None): 
    
    # Create a figure
    fig = plt.figure(figsize=(3.40,4.15), dpi=300 )
    ax = plt.subplot(111)
    if diff:
       if min_value==None:
          min_value = - max( abs(np.nanmin(field[level,:,:])), abs(np.nanmax(field[level,:,:])) )
          max_value =   max( abs(np.nanmin(field[level,:,:])), abs(np.nanmax(field[level,:,:])) )
       if cmap==None:
          cmap = 'bwr'
    else:
       if cmap==None:
          cmap = 'viridis'
    ax, im = plot_depth_ax(fig, ax, field, level, lon_labels, lat_labels, depth_labels, text, min_value, max_value, cmap)

    # Add a color bar
    cb=plt.colorbar(im, ax=(ax), shrink=0.8, pad=0.10, extend='max') #, anchor=(0.5, 1.3))
    if cbar_label:
       cb.set_label(cbar_label)    
    if Sci:
       cb.formatter.set_powerlimits((-2, 2))
       cb.ax.yaxis.set_offset_position('right')
       cb.update_ticks()

    if title:
       plt.suptitle(title, fontsize=14)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.5,hspace=0.01)

    return(fig, ax, im)


def plot_depth_fld_diff(fig, ax1, ax2, field1, field1_name, field2, field2_name, level, lon_labels, lat_labels, depth_labels, title=None,
                        cbar_label=None, cbar_diff_label=None, Sci=None):
    
    flds_min_value = min( np.nanmin(field1[level,:,:]), np.nanmin(field2[level,:,:]) )
    flds_max_value = max( np.nanmax(field1[level,:,:]), np.amax(field2[level,:,:]) )

    diff_min_value = -max( abs(np.nanmin(field1[level,:,:]-field2[level,:,:])), abs(np.nanmax(field1[level,:,:]-field2[level,:,:])) )
    diff_max_value =  max( abs(np.nanmin(field1[level,:,:]-field2[level,:,:])), abs(np.nanmax(field1[level,:,:]-field2[level,:,:])) )
 
    ax1, im1 = plot_depth_ax(fig, ax1, field1, level, lon_labels, lat_labels, depth_labels, flds_min_value, flds_max_value)
    ax2, im2 = plot_depth_ax(fig, ax2, field1-field2, level, lon_labels, lat_labels, depth_labels, diff_min_value, diff_max_value, cmap='bwr')

    cb1axes = fig.add_axes([0.05, 0.045, 0.35, 0.03]) 
    cb1=plt.colorbar(im1, ax=ax1, cax=cb1axes, orientation='horizontal')
    if cbar_label:
       cb1.set_label(cbar_label)    
    if Sci:
       cb1.formatter.set_powerlimits((-2, 2))

    cb2axes = fig.add_axes([0.55, 0.045, 0.35, 0.03]) 
    cb2=plt.colorbar(im2, ax=ax2, cax=cb2axes, orientation='horizontal')
    if cbar_diff_label:
       cb2.set_label(cbar_diff_label)    
    if Sci:
       cb2.formatter.set_powerlimits((-2, 2))
       cb2.ax.yaxis.set_offset_position('left')
       cb2.update_ticks()

    if title:
       plt.suptitle(title, fontsize=14)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.4, hspace = 0.01, bottom=0.15)

    return(ax1, ax2)


#-----------------------------------------------------
# Plot cross section at x=const (i.e. North to South)
#-----------------------------------------------------

def plot_xconst_crss_sec_ax(fig, ax, field, x, lon_labels, lat_labels, depth_labels, text=None, min_value=None, max_value=None, cmap=None):
    # Assumes field is (z, y, x)

    if not min_value:
       min_value = np.nanmin(field[:,:,x])  # Lowest value
    if not max_value:
       max_value = np.nanmax(field[:,:,x])   # Highest value

    im = ax.pcolormesh(field[:,:,x], vmin=min_value, vmax=max_value, cmap=cmap, edgecolors='face', snap=True)
    ax.invert_yaxis()
    ax.set_xlabel('Latitude ('+u'\xb0'+' N)')
    ax.set_ylabel('Depth (m)')

    # Give axis ticks in lat/lon/depth
    lat_arange = [0, 8.36, 15.5, 21.67, 27.25, 32.46, 37.5, 42.54, 47.75, 53.32, 59.5, 66.64, 75.5]
    ax.set_xticks(lat_arange)
    ax.set_xticklabels(np.round(lat_labels[np.array(lat_arange).astype(int)], decimals=-1).astype(int)) 
    depth_arange = [0, 7, 15, 21, 28, 37, depth_labels.shape[0]-1]
    ax.set_yticks(depth_arange)
    ax.set_yticklabels(depth_labels[np.array(depth_arange)].astype(int))

    if text:
       plt.text(0.015, 0.90, text, transform=fig.transFigure)    
 
    return(ax, im)

def plot_xconst_crss_sec(field, field_name, x, lon_labels, lat_labels, depth_labels, text=None, title=None, min_value=None, max_value=None,
                         diff=False, cmap=None, cbar_label=None, Sci=None):

    # Create a figure
    fig = plt.figure( figsize=(4.45 ,2.3), dpi=300 )
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
    ax, im = plot_xconst_crss_sec_ax(fig, ax, field, x, lon_labels, lat_labels, depth_labels, text, min_value, max_value, cmap)

    # Add a color bar
    cb=plt.colorbar(im, ax=(ax), shrink=0.8)
    if cbar_label:
       cb.set_label(cbar_label)    
    if Sci: 
       cb.formatter.set_powerlimits((-2, 2))
       cb.ax.yaxis.set_offset_position('left')
       cb.update_ticks()

    if title:
       plt.suptitle(title, fontsize=14)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.01,hspace=0.01)

    return(fig, ax, im)

def plot_xconst_crss_sec_diff(fig, ax1, ax2, field1, field1_name, field2, field2_name, x, lon_labels, lat_labels, depth_labels, title=None,
                              cbar_label=None, cbar_diff_label=None, Sci=None):
    
    flds_min_value = min( np.nanmin(field1[:,:,x]), np.nanmin(field2[:,:,x]) )
    flds_max_value = max( np.nanmax(field1[:,:,x]), np.amax(field2[:,:,x]) )

    diff_min_value = -max( abs(np.nanmin(field1[:,:,x]-field2[:,:,x])), abs(np.nanmax(field1[:,:,x]-field2[:,:,x])) )
    diff_max_value =  max( abs(np.nanmin(field1[:,:,x]-field2[:,:,x])), abs(np.nanmax(field1[:,:,x]-field2[:,:,x])) )
 
    fig = plt.figure(figsize=(9,11), dpi=300 )
    ax1, im1 = plot_xconst_crss_sec_ax(fig, ax1, field1, x, lon_labels, lat_labels, depth_labels, flds_min_value, flds_max_value)
    ax2, im2 = plot_xconst_crss_sec_ax(fig, ax2, field1-field2, x, lon_labels, lat_labels, depth_labels, diff_min_value, diff_max_value, cmap='bwr')

    cb1axes = fig.add_axes([0.92, 0.58, 0.03, 0.35]) 
    cb1=plt.colorbar(im1, ax=ax1, orientation='vertical', cax=cb1axes)
    if cbar_label:
       cb1.set_label(cbar_label)    
    if Sci:
       cb1.formatter.set_powerlimits((-2, 2))
       cb1.ax.yaxis.set_offset_position('left')
       cb1.update_ticks()

    cb2axes = fig.add_axes([0.92, 0.08, 0.03, 0.35]) 
    cb2=plt.colorbar(im2, ax=ax2, orientation='vertical', cax=cb2axes)
    if cbar_diff_label:
       cb2.set_label(cbar_diff_label)    
    if Sci:
       cb2.formatter.set_powerlimits((-2, 2))
       cb2.ax.yaxis.set_offset_position('left')
       cb2.update_ticks()
 
    if title:
       plt.suptitle(title, fontsize=14)

    plt.tight_layout()
    plt.subplots_adjust(hspace = 0.2, right=0.9, bottom=0.07, top=0.95)

    return(ax1, ax2)


#---------------------------------------------------------------------------------------
# Plot cross sections for y=const at a few discrete time points - start, middle and end
#---------------------------------------------------------------------------------------

def plot_yconst_crss_sec_ax(fig, ax, field, y, lon_labels, lat_labels, depth_labels, text=None, min_value=None, max_value=None, cmap=None):
    # Assumes field is (z, y, x)

    if not min_value:
       min_value = np.nanmin(field[:,y,:])  # Lowest value
    if not max_value:
       max_value = np.nanmax(field[:,y,:])   # Highest value

    im = ax.pcolormesh(field[:,y,:], vmin=min_value, vmax=max_value, cmap=cmap, edgecolors='face', snap=True)
    ax.invert_yaxis()
    ax.set_xlabel('Longitude ('+u'\xb0'+' E)')
    ax.set_ylabel('Depth (m)')
   
    # Give axis ticks in lat/lon/depth
    lon_arange = [0, 2, 4.5, 7, 9.5]
    ax.set_xticks(lon_arange)
    ax.set_xticklabels(np.round(lon_labels[np.array(lon_arange).astype(int)], decimals=0 ).astype(int)) 
    depth_arange = [0, 7, 15, 21, 28, 37, depth_labels.shape[0]-1]
    ax.set_yticks(depth_arange)
    ax.set_yticklabels(depth_labels[np.array(depth_arange)].astype(int))
   
    if text:
       plt.text(0.015, 0.90, text, transform=fig.transFigure)    
 
    return(ax, im)

def plot_yconst_crss_sec(field, field_name, y, lon_labels, lat_labels, depth_labels, text=None, title=None, min_value=None, max_value=None,
                         diff=False, cmap=None, cbar_label=None, Sci=None):
 
    # Create a figure
    fig = plt.figure( figsize=(4.45 ,2.3), dpi=300 )
    ax = plt.subplot(111)
    if diff:
       min_value = - max( abs(np.nanmin(field[:,y,:])), abs(np.nanmax(field[:,y,:])) )
       max_value =   max( abs(np.nanmin(field[:,y,:])), abs(np.nanmax(field[:,y,:])) )
       if cmap==None:
          cmap = 'bwr'
    else:
       if cmap==None:
          cmap = 'viridis'
    ax, im = plot_yconst_crss_sec_ax(fig, ax, field, y, lon_labels, lat_labels, depth_labels, text, min_value, max_value, cmap)

    # Add a color bar
    cb=plt.colorbar(im, ax=(ax), shrink=0.8)
    if cbar_label:
       cb.set_label(cbar_label)    
    if Sci:
       cb.formatter.set_powerlimits((-2, 2))
       cb.ax.yaxis.set_offset_position('left')
       cb.update_ticks()
    
    if title:
       plt.suptitle(title, fontsize=14)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.01,hspace=0.01)

    return(fig, ax, im)

def plot_yconst_crss_sec_diff(fig, ax1, ax2, field1, field1_name, field2, field2_name, y, lon_labels, lat_labels, depth_labels, title=None,
                              cbar_label=None, cbar_diff_label=None, Sci=None):
    
    flds_min_value = min( np.nanmin(field1[:,y,:]), np.nanmin(field2[:,y,:]) )
    flds_max_value = max( np.nanmax(field1[:,y,:]), np.amax(field2[:,y,:]) )

    diff_min_value = -max( abs(np.nanmin(field1[:,y,:]-field2[:,y,:])), abs(np.nanmax(field1[:,y,:]-field2[:,y,:])) )
    diff_max_value =  max( abs(np.nanmin(field1[:,y,:]-field2[:,y,:])), abs(np.nanmax(field1[:,y,:]-field2[:,y,:])) )
 
    ax1, im1 = plot_yconst_crss_sec_ax(fig, ax1, field1, y, lon_labels, lat_labels, depth_labels, flds_min_value, flds_max_value)
    ax2, im2 = plot_yconst_crss_sec_ax(fig, ax2, field1-field2, y, lon_labels, lat_labels, depth_labels, diff_min_value, diff_max_value, cmap='bwr')

    cb1axes = fig.add_axes([0.92, 0.58, 0.03, 0.35]) 
    cb1=plt.colorbar(im1, ax=ax1, orientation='vertical', cax=cb1axes)
    if cbar_label:
       cb1.set_label(cbar_label)    
    if Sci:
       cb1.formatter.set_powerlimits((-2, 2))
       cb1.ax.yaxis.set_offset_position('left')
       cb1.update_ticks()

    cb2axes = fig.add_axes([0.92, 0.08, 0.03, 0.35]) 
    cb2=plt.colorbar(im2, ax=ax2, orientation='vertical', cax=cb2axes)
    if cbar_diff_label:
       cb2.set_label(cbar_diff_label)    
    if Sci:
       cb2.formatter.set_powerlimits((-2, 2))
       cb2.ax.yaxis.set_offset_position('left')
       cb2.update_ticks()
 
    if title:
       plt.suptitle(title, fontsize=14)

    plt.tight_layout()
    plt.subplots_adjust(hspace = 0.2, right=0.9, bottom=0.07, top=0.95)

    return(ax1, ax2)


#-------------------------------------
# Plot time series at specific points
#-------------------------------------
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
   if ylim:
      ax.set_ylim(ylim)
 
   return(ax)

def plt_timeseries(point, length, datasets, ylim=None):
   
   fig = plt.figure(figsize=(15 ,3), dpi=300 )
   ax=plt.subplot(111)
   if ylim == None:
      bottom = min(datasets['MITGCM'][:length, point[0], point[1], point[2]])-1
      top    = max(datasets['MITGCM'][:length, point[0], point[1], point[2]])+1
      ylim=[bottom, top]
   ax=plt_timeseries_ax(ax, point, length, datasets, ylim=ylim)

   plt.tight_layout()
   plt.subplots_adjust(hspace = 0.5, left=0.05, right=0.95, bottom=0.15, top=0.90)

   return(fig)

def plt_2_timeseries(point, length1, length2, datasets, ylim=None):
   
   fig = plt.figure(figsize=(15 ,7))

   ax=plt.subplot(211)
   if ylim == None:
      bottom = min(datasets['MITGCM'][:length1, point[0], point[1], point[2]])-1
      top    = max(datasets['MITGCM'][:length1, point[0], point[1], point[2]])+1
      ylim=[bottom, top]
   ax=plt_timeseries_ax(ax, point, length1, datasets, ylim=ylim)

   ax=plt.subplot(212)
   if ylim == None:
      bottom = min(datasets['MITGCM'][:length2, point[0], point[1], point[2]])-1
      top    = max(datasets['MITGCM'][:length2, point[0], point[1], point[2]])+1
      ylim=[bottom, top]
   ax=plt_timeseries_ax(ax, point, length2, datasets, ylim=ylim)

   plt.tight_layout()
   plt.subplots_adjust(hspace = 0.5, left=0.05, right=0.95, bottom=0.07, top=0.95)

   return(fig)

def plt_3_timeseries(point, length1, length2, length3, datasets, ylim=None):
   
   fig = plt.figure(figsize=(15 ,11))

   ax=plt.subplot(311)
   if ylim == None:
      bottom = min(datasets['MITGCM'][:length1, point[0], point[1], point[2]])-1
      top    = max(datasets['MITGCM'][:length1, point[0], point[1], point[2]])+1
      ylim=[bottom, top]
   ax=plt_timeseries_ax(ax, point, length1, datasets, ylim=ylim)

   ax=plt.subplot(312)
   if ylim == None:
      bottom = min(datasets['MITGCM'][:length2, point[0], point[1], point[2]])-1
      top    = max(datasets['MITGCM'][:length2, point[0], point[1], point[2]])+1
      ylim=[bottom, top]
   ax=plt_timeseries_ax(ax, point, length2, datasets, ylim=ylim)

   ax=plt.subplot(313)
   if ylim == None:
      bottom = min(datasets['MITGCM'][:length3, point[0], point[1], point[2]])-1
      top    = max(datasets['MITGCM'][:length3, point[0], point[1], point[2]])+1
      ylim=[bottom, top]
   ax=plt_timeseries_ax(ax, point, length3, datasets, ylim=ylim)

   plt.tight_layout()
   plt.subplots_adjust(hspace = 0.5, left=0.05, right=0.95, bottom=0.07, top=0.95)

   return(fig)


#-----------------------------
# Plot histogram of y_hat - y
#-----------------------------
def Plot_Histogram(data, no_bins):

    fig = plt.figure(figsize=(10, 8))

    if len(data.shape) > 1:
       data = data.reshape(-1)

    plt.hist(data, bins = no_bins)
    plt.yscale('log')
    plt.annotate('skew = '+str(np.round(skew(data),5)), (0.1,0.9), xycoords='figure fraction')
    return(fig)
