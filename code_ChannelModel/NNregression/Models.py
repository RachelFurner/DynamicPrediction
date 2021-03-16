#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.append('../Tools')

import numpy as np
import os
import xarray as xr
import pickle

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from torch.utils import data
import torch.nn as nn
from torch.autograd import Variable
import torch
from torchvision import transforms, utils

from collections import OrderedDict

import netCDF4 as nc4

import time as time
import gc

def CreateModel(model_style, no_input_channels, no_target_channels, lr, reproducible, seed_value):
   # inputs are (no_samples, 115channels, 100y, 240x).  (38 vertical levels, T, U, V 3d, Eta 2d)
   if reproducible:
      os.environ['PYTHONHASHSEED'] = str(seed_value)
      np.random.seed(seed_value)
      torch.manual_seed(seed_value)
      torch.cuda.manual_seed(seed_value)
      torch.cuda.manual_seed_all(seed_value)
      torch.backends.cudnn.enabled = False
      torch.backends.cudnn.deterministic = True
      torch.backends.cudnn.benchmark = False


   if model_style == 'ScherStyle':

      h = nn.Sequential(
         # downscale
         # ****Note scher uses a kernel size of 6, but no idea how he then manages to keep his x and y dimensions
         # unchanged over the conv layers....***
         nn.Conv2d(in_channels=no_input_channels, out_channels=64 , kernel_size=(7,7), padding=(3,3)),
         nn.ReLU(True),
         nn.MaxPool2d((2,2)),
         nn.Conv2d(in_channels=64 , out_channels=64 , kernel_size=(7,7),padding=(3,3)),
         nn.ReLU(True),
         nn.MaxPool2d((2,2)),
   
         nn.ConvTranspose2d(in_channels=64 , out_channels=64  , kernel_size=(7,7), padding=(3,3)),
         nn.ReLU(True),
   
         # upscale
         nn.Conv2d(in_channels=64  , out_channels=64 , kernel_size=(7,7), padding=(3,3)),
         nn.ReLU(True),
         nn.Upsample(scale_factor=(2,2)),
         nn.Conv2d(in_channels=64 , out_channels=64 , kernel_size=(7,7), padding=(3,3)),
         nn.ReLU(True),
         nn.Upsample(scale_factor=(2,2)),
         nn.Conv2d(in_channels=64 , out_channels=no_target_channels, kernel_size=(7,7), padding=(3,3)),
         nn.ReLU(True)
          )
      h = h.cuda()
   
      optimizer = torch.optim.Adam( h.parameters(), lr=lr )

   elif model_style == 'UNet':

     class CustomPad(nn.Module):
       def __init__(self):
         super().__init__()
       def forward(self, x):
         # apply cyclical padding in x dir
         tmp = nn.functional.pad(x  ,(3,3,0,0),mode='circular')
         #tmp = nn.functional.pad(x  ,(3,3,0,0),mode='constant', value=0)
         # apply other    padding in y dir
         #out = nn.functional.pad(tmp,(0,0,3,3),mode='constant', value=0)
         out = nn.functional.pad(tmp,(0,0,3,3),mode='reflect')
         return out

     # based on code at https://github.com/mateuszbuda/brain-segmentation-pytorch/blob/master/unet.py
     class UNet(nn.Module):
        def __init__(self, in_channels=no_input_channels, out_channels=no_target_channels):
           super(UNet, self).__init__()

           features = 128
           self.encoder1 = UNet._block(in_channels, features, name="enc1")
           self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
           self.encoder2 = UNet._block(features, features*2, name="enc2")
           self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

           self.bottleneck = UNet._block(features*2, features*4, name="bottleneck")

           self.upconv2 = nn.ConvTranspose2d(features*4, features*2, kernel_size=2, stride=2)
           self.decoder2 = UNet._block(features*2*2, features*2, name="dec2")
           self.upconv1 = nn.ConvTranspose2d(features*2, features, kernel_size=2, stride=2)
           self.decoder1 = UNet._block(features*2, features, name="dec1")

           self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)

        def forward(self, x):
           enc1 = self.encoder1(x)
           #print('enc1.shape; '+str(enc1.shape))
           enc2 = self.encoder2(self.pool1(enc1))
           #print('enc2.shape; '+str(enc2.shape))

           bottleneck = self.bottleneck(self.pool2(enc2))
           #print('bottleneck.shape; '+str(bottleneck.shape))

           dec2 = self.upconv2(bottleneck)
           #print('dec2.shape  a; '+str(dec2.shape))
           dec2 = torch.cat((dec2, enc2), dim=1)
           #print('dec2.shape  b; '+str(dec2.shape))
           dec2 = self.decoder2(dec2)
           #print('dec2.shape  c; '+str(dec2.shape))
           dec1 = self.upconv1(dec2)
           #print('dec1.shape  a; '+str(dec1.shape))
           dec1 = torch.cat((dec1, enc1), dim=1)
           #print('dec1.shape  b; '+str(dec1.shape))
           dec1 = self.decoder1(dec1)
           #print('dec1.shape  c; '+str(dec1.shape))
           return self.conv(dec1)

        @staticmethod
        def _block(in_channels, features, name):
           return nn.Sequential(
               OrderedDict(
                   [
                       ( name + "pad1", CustomPad() ),
                       (
                           name + "conv1",
                           nn.Conv2d(
                               in_channels=in_channels,
                               out_channels=features,
                               kernel_size=7,
                               bias=False,
                           ),
                       ),
                       (name + "norm1", nn.BatchNorm2d(num_features=features)),
                       (name + "relu1", nn.ReLU(inplace=True)),

                       ( name + "pad2", CustomPad() ),
                       (
                           name + "conv2",
                           nn.Conv2d(
                               in_channels=features,
                               out_channels=features,
                               kernel_size=7,
                               bias=False,
                           ),
                       ),
                       (name + "norm2", nn.BatchNorm2d(num_features=features)),
                       (name + "relu2", nn.ReLU(inplace=True)),
                   ]
               )
           )


     unet = UNet()
     x    = torch.randn(1, 115, 100, 240)
     h = UNet()#.cuda()
     h = h.cuda()   
     unet(x)#.shape
      
     optimizer = torch.optim.Adam( h.parameters(), lr=lr )

   else:
      h=None
      optimizer=None
      print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
      print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
      print('WARNING NO MODEL DEFINED')
      print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
      print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
   
   return h, optimizer
