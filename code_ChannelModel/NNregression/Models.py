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


class CustomPad2d(nn.Module):
   def __init__(self, padding_type):
      super().__init__()
      self.padding_type = padding_type
   def forward(self, x):
      # apply cyclical padding in x dir
      out = nn.functional.pad(x  ,(3,3,0,0),mode='circular')
      # apply other padding in y dir
      if self.padding_type == 'Cons':
         out = nn.functional.pad(out,(0,0,3,3),mode='constant', value=0)
      elif self.padding_type == 'Refl':
         out = nn.functional.pad(out,(0,0,3,3),mode='reflect')
      elif self.padding_type == 'Repl':
         out = nn.functional.pad(out,(0,0,3,3),mode='replicate')
      else:
         raise RuntimeError('ERROR - NO Padding style given!!!')
      return out

class CustomPad2dTransp(nn.Module):
   def __init__(self):
      super().__init__()
   def forward(self, x):
      # apply cyclical padding in x dir
      out = nn.functional.pad(x  ,(1,1,0,0),mode='circular')
      return out

class CustomPad3d(nn.Module):
   def __init__(self, padding_type):
      super().__init__()
      self.padding_type = padding_type
   def forward(self, x):
      # apply cyclical padding in x dir
      out = nn.functional.pad(x  ,(1,1,0,0,0,0),mode='circular')
      # apply other padding in y dir
      if padding_type == 'Cons':
         out = nn.functional.pad(out,(0,0,1,1,1,1),mode='constant', value=0)
      elif padding_type == 'Refl':
         out = nn.functional.pad(out,(0,0,1,1,1,1),mode='reflect')
      elif padding_type == 'Repl':
         out = nn.functional.pad(out,(0,0,1,1,1,1),mode='replicate')
      else:
         raise RuntimeError('ERROR - NO Padding style given!!!')
      return out


class UNet2d(nn.Module):
   def __init__(self, in_channels, out_channels, padding_type):
      super(UNet2d, self).__init__()

      features = 128
      self.encoder1 = UNet2d._block(in_channels, features, padding_type, name="enc1")
      self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
      self.encoder2 = UNet2d._block(features, features*2, padding_type, name="enc2")
      self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

      self.bottleneck = UNet2d._block(features*2, features*4, padding_type, name="bottleneck")

      self.upconv2 = nn.ConvTranspose2d(features*4, features*2, kernel_size=2, stride=2)
      self.decoder2 = UNet2d._block(features*2*2, features*2, padding_type, name="dec2")
      self.upconv1 = nn.ConvTranspose2d(features*2, features, kernel_size=2, stride=2)
      self.decoder1 = UNet2d._block(features*2, features, padding_type, name="dec1")

      self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)

   def forward(self, x):
      #print('x.shape; '+str(x.shape))
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
   def _block(in_channels, features, padding_type, name):
      return nn.Sequential(
          OrderedDict(
              [
                  ( name + "pad1", CustomPad2d(padding_type) ),
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

                  ( name + "pad2", CustomPad2d(padding_type) ),
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

class UNet2dTransp(nn.Module):
   def __init__(self, in_channels, out_channels, padding_type):
      super(UNet2dTransp, self).__init__()

      features = 128
      self.encoder1 = UNet2dTransp._down_block(in_channels, features, padding_type, name="enc1")
      self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
      self.encoder2 = UNet2dTransp._down_block(features, features*2, padding_type, name="enc2")
      self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

      self.bottleneck = UNet2dTransp._down_block(features*2, features*4, padding_type, name="bottleneck")

      self.upsample2 = nn.Upsample(size=(44,120), mode='bilinear')
      self.upconv2 = nn.Conv2d(features*4, features*2, kernel_size=3, stride=1)
      self.decoder2 = UNet2dTransp._up_block(features*2*2, features*2, padding_type, name="dec2")
      self.upsample1 = nn.Upsample(size=(94,240), mode='bilinear')
      self.upconv1 = nn.Conv2d(features*2, features, kernel_size=3, stride=1)
      self.decoder1 = UNet2dTransp._up_block(features*2, features, padding_type, name="dec1")

      self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)

   def forward(self, x):
      #print('x.shape; '+str(x.shape))
      enc1 = self.encoder1(x)
      #print('enc1.shape; '+str(enc1.shape))
      enc2a = self.pool1(enc1)
      #print('enc2a.shape; '+str(enc2a.shape))
      enc2b = self.encoder2(enc2a)
      #print('enc2b.shape; '+str(enc2b.shape))

      bottlenecka = self.pool2(enc2b)
      #print('bottlenecka.shape; '+str(bottlenecka.shape))
      bottleneckb = self.bottleneck(bottlenecka)
      #print('bottleneckb.shape; '+str(bottleneckb.shape))

      dec2a = self.upsample2(bottleneckb)
      #print('dec2a.shape; '+str(dec2a.shape))
      cust_pad = CustomPad2dTransp()
      dec2b = cust_pad.forward(dec2a)
      #print('dec2b.shape; '+str(dec2b.shape))
      dec2c = self.upconv2(dec2b)
      #print('dec2c.shape; '+str(dec2c.shape))
      dec2d = torch.cat((dec2c, enc2b), dim=1)
      #print('dec2d.shape; '+str(dec2d.shape))
      dec2e = self.decoder2(dec2d)
      #print('dec2e.shape; '+str(dec2e.shape))
      dec1a = self.upsample1(dec2e)
      #print('dec1a.shape; '+str(dec1a.shape))
      dec1b = cust_pad.forward(dec1a)
      #print('dec1b.shape; '+str(dec1b.shape))
      dec1c = self.upconv1(dec1b)
      #print('dec1c.shape; '+str(dec1c.shape))
      dec1d = torch.cat((dec1c, enc1), dim=1)
      #print('dec1d.shape; '+str(dec1d.shape))
      dec1e = self.decoder1(dec1d)
      #print('dec1e.shape; '+str(dec1e.shape))
      return self.conv(dec1e)

   @staticmethod
   def _down_block(in_channels, features, padding_type, name):
      return nn.Sequential(
          OrderedDict(
              [
                  ( name + "pad1", CustomPad2dTransp() ),
                  (
                      name + "conv1",
                      nn.Conv2d(
                          in_channels=in_channels,
                          out_channels=features,
                          kernel_size=3,
                          bias=False,
                      ),
                  ),
                  (name + "norm1", nn.BatchNorm2d(num_features=features)),
                  (name + "relu1", nn.ReLU(inplace=True)),

                  ( name + "pad2", CustomPad2dTransp() ),
                  (
                      name + "conv2",
                      nn.Conv2d(
                          in_channels=features,
                          out_channels=features,
                          kernel_size=3,
                          bias=False,
                      ),
                  ),
                  (name + "norm2", nn.BatchNorm2d(num_features=features)),
                  (name + "relu2", nn.ReLU(inplace=True)),
              ]
          )
      )

   @staticmethod
   def _up_block(in_channels, features, padding_type, name):
      return nn.Sequential(
          OrderedDict(
              [
                  ( name + "pad1", CustomPad2dTransp() ),
                  (
                      name + "conv1",
                      nn.ConvTranspose2d(
                          in_channels=in_channels,
                          out_channels=features,
                          kernel_size=3,
                          padding=(0,2),
                          bias=False,
                      ),
                  ),
                  (name + "norm1", nn.BatchNorm2d(num_features=features)),
                  (name + "relu1", nn.ReLU(inplace=True)),

                  ( name + "pad2", CustomPad2dTransp() ),
                  (
                      name + "conv2",
                      nn.ConvTranspose2d(
                          in_channels=features,
                          out_channels=features,
                          kernel_size=3,
                          padding=(0,2),
                          bias=False,
                      ),
                  ),
                  (name + "norm2", nn.BatchNorm2d(num_features=features)),
                  (name + "relu2", nn.ReLU(inplace=True)),
              ]
          )
      )

class UNet3d(nn.Module):
   def __init__(self, in_channels, out_channels, padding_type):
      super(UNet3d, self).__init__()

      features = 16 
      self.encoder1 = UNet3d._block(in_channels, features, padding_type, name="enc1")
      self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
      self.encoder2 = UNet3d._block(features, features*2, padding_type, name="enc2")
      self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

      self.bottleneck = UNet3d._block(features*2, features*4, padding_type, name="bottleneck")

      self.upconv2 = nn.ConvTranspose3d(features*4, features*2, kernel_size=2, stride=2, output_padding=(1,0,0) )
      self.decoder2 = UNet3d._block(features*2*2, features*2, padding_type, name="dec2")
      self.upconv1 = nn.ConvTranspose3d(features*2, features, kernel_size=2, stride=2)
      self.decoder1 = UNet3d._block(features*2, features, padding_type, name="dec1")

      self.conv = nn.Conv3d(in_channels=features, out_channels=out_channels, kernel_size=1)

   def forward(self, x):
      #print('x.shape; '+str(x.shape))
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
   def _block(in_channels, features, padding_type, name):
      return nn.Sequential(
          OrderedDict(
              [
                  ( name + "pad1", CustomPad3d(padding_type) ),
                  (
                      name + "conv1",
                      nn.Conv3d(
                          in_channels=in_channels,
                          out_channels=features,
                          kernel_size=3,
                          bias=False,
                      ),
                  ),
                  (name + "norm1", nn.BatchNorm3d(num_features=features)),
                  (name + "relu1", nn.ReLU(inplace=True)),

                  ( name + "pad2", CustomPad3d(padding_type) ),
                  (
                      name + "conv2",
                      nn.Conv3d(
                          in_channels=features,
                          out_channels=features,
                          kernel_size=3,
                          bias=False,
                      ),
                  ),
                  (name + "norm2", nn.BatchNorm3d(num_features=features)),
                  (name + "relu2", nn.ReLU(inplace=True)),
              ]
          )
      )


def CreateModel(model_style, no_input_channels, no_target_channels, lr, seed_value, padding_type):
   # inputs are (no_samples, 115channels, 100y, 240x).  (38 vertical levels, T, U, V 3d, Eta 2d)
   os.environ['PYTHONHASHSEED'] = str(seed_value)
   np.random.seed(seed_value)
   torch.manual_seed(seed_value)
   torch.cuda.manual_seed(seed_value)
   torch.cuda.manual_seed_all(seed_value)
   torch.backends.cudnn.enabled = False
   torch.backends.cudnn.deterministic = True
   torch.backends.cudnn.benchmark = False

   if model_style == 'UNet2d':
      h = UNet2d(no_input_channels, no_target_channels, padding_type)
   elif model_style == 'UNet2dtransp':
      h = UNet2dTransp(no_input_channels, no_target_channels, padding_type)
   elif model_style == 'UNet3d':
      h = UNet3d(no_input_channels, no_target_channels, padding_type)
   else:
      raise RuntimeError('WARNING NO MODEL DEFINED')

   if torch.cuda.is_available():
       h = h.cuda()
 
   optimizer = torch.optim.Adam( h.parameters(), lr=lr )

   return h, optimizer

