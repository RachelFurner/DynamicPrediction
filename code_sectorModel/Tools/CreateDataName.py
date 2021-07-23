#!/usr/bin/env python
# coding: utf-8

# Script written by Rachel Furner 
# contains simple routine to create part of an experiment name from run variables.

def create_dataname(run_vars):
   '''
   Small routine to take model type and runs variables and create part of an experiment name, detailing the input data being used
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
   if run_vars['bolus_vel']:
      exp_name=exp_name+'Bol'
   if run_vars['sal']:
      exp_name=exp_name+'Sal'
   if run_vars['eta']:
      exp_name=exp_name+'Eta'
   if run_vars['density']:
      exp_name=exp_name+'Dns'
   exp_name = exp_name+'PolyDeg'+str(run_vars['poly_degree'])
   exp_name = exp_name+'_Step'+str(run_vars['StepSize'])
   exp_name = exp_name+'_Predict'+str(run_vars['predict'])
   return exp_name 
