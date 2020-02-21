#!/usr/bin/env python
# coding: utf-8

# Code developed by Rachel Furner to contain modules used in developing stats/nn based versions of GCMs.
# Routines contained here include:
# ReadMITGCM - Routine to read in MITGCM data into input and output arrays, split into test and train
#                        portions of code, and normalise. The arrays are saved, and also passed back on return. 
# create_expname - short routine to create experiment name from model type and run variables.

def create_expname(model_type, run_vars):
   '''
   Small routine to take model type and runs variables and create an experiment name.
   '''
   if run_vars['dimension'] == 2:
      exp_name='2d'
   elif run_vars['dimension'] == 3:
      exp_name='3d'
   else:
      print('ERROR, dimension neither two or three!!!')
   if run_vars['lat']:
      exp_name=exp_name+'Lat'
   if run_vars['lon']:
      exp_name=exp_name+'Lon'
   if run_vars['dep']:
      exp_name=exp_name+'Dep'
   if run_vars['current']:
      exp_name=exp_name+'UV'
   if run_vars['sal']:
      exp_name=exp_name+'Sal'
   if run_vars['eta']:
      exp_name=exp_name+'Eta'
   exp_name = exp_name+'PolyDeg'+str(run_vars['poly_degree'])
   return exp_name 
