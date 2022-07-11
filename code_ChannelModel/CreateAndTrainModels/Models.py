#!/usr/bin/env python
# coding: utf-8

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

import GPUtil

import logging

class CustomPad2dTransp(nn.Module):
   def __init__(self, kern_size):
      super().__init__()
      self.kern_size = kern_size
   def forward(self, x):
      # apply cyclical padding in x dir
      # pad_size = int( (self.kern_size-1)/2 )
      x = nn.functional.pad(x, ( int( (self.kern_size-1)/2 ), int( (self.kern_size-1)/2 ), 0, 0), mode='circular')
      return x

class CustomPad3dTransp(nn.Module):
   def __init__(self, kern_size):
      super().__init__()
      self.kern_size = kern_size
   def forward(self, x):
      # apply cyclical padding in x dir
      # pad_size = int( (self.kern_size-1)/2 )
      x = nn.functional.pad(x, ( int( (self.kern_size-1)/2 ), int( (self.kern_size-1)/2 ), 0, 0, 0, 0), mode='circular')
      return x

class CustomPad2d(nn.Module):
   def __init__(self, padding_type, kern_size):
      super().__init__()
      self.padding_type = padding_type
      self.pad_size = int( (kern_size-1)/2 )
   def forward(self, x):
      # apply cyclical padding in x dir
      x = nn.functional.pad(x  ,(self.pad_size,self.pad_size,0,0),mode='circular')
      # apply other padding in y dir
      if self.padding_type == 'Cons':
         x = nn.functional.pad(x,(0,0,self.pad_size,self.pad_size),mode='constant', value=0)
      #elif self.padding_type == 'Refl':
      #   x = nn.functional.pad(x,(0,0,self.pad_size,self.pad_size),mode='reflect')
      # Be wary using Repl - mask set up means const with 0 gives land outside of domain, Repl doesn't on S bdy
      #elif self.padding_type == 'Repl':
      #   x = nn.functional.pad(x,(0,0,self.pad_size,self.pad_size),mode='replicate')
      else:
         raise RuntimeError('ERROR - NO Padding style given!!!')
      return x

class CustomPad3d(nn.Module):
   def __init__(self, padding_type, kern_size):
      super().__init__()
      self.padding_type = padding_type
      self.pad_size = int( (kern_size-1)/2 )
   def forward(self, x):
      # apply cyclical padding in x dir
      x = nn.functional.pad(x  ,(self.pad_size,self.pad_size,0,0,0,0),mode='circular')
      # apply other padding in y dir
      if self.padding_type == 'Cons':
         x = nn.functional.pad(x,(0,0,self.pad_size,self.pad_size,self.pad_size,self.pad_size),mode='constant', value=0)
      # Refl not available in 3d
      # Be wary using Repl - mask set up means const with 0 gives land outside of domain, Repl doesn't on S bdy
      #elif self.padding_type == 'Repl':
      #   x = nn.functional.pad(x,(0,0,self.pad_size,self.pad_size,self.pad_size,self.pad_size),mode='replicate')
      else:
         raise RuntimeError('ERROR - NO Padding style given!!!')
      return x


class UNet2dTransp(nn.Module):
   def __init__(self, in_channels, out_channels, padding_type, xdim, ydim, kern_size):
      super(UNet2dTransp, self).__init__()

      features = 2**(in_channels-1).bit_length()  # nearest power of two to input channels
      logging.info('No features ; '+str(features)+'\n')

      self.encoder1 = UNet2dTransp._down_block(in_channels, features, padding_type, kern_size, name="enc1")
      self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
      self.encoder2 = UNet2dTransp._down_block(features, features*2, padding_type, kern_size, name="enc2")
      self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

      self.bottleneck = UNet2dTransp._down_block(features*2, features*4, padding_type, kern_size, name="bottleneck")

      self.upsample2 = nn.Upsample(size=(int(ydim/2-(kern_size-1)*2),int(xdim/2)), mode='bilinear')
      self.upconv2 = nn.Conv2d(features*4, features*2, kernel_size=kern_size, stride=1)
      self.decoder2 = UNet2dTransp._up_block(features*2*2, features*2, padding_type, kern_size, name="dec1")
      self.upsample1 = nn.Upsample(size=(int(ydim-(kern_size-1)),int(xdim)), mode='bilinear')
      self.upconv1 = nn.Conv2d(features*2, features, kernel_size=kern_size, stride=1)
      self.decoder1 = UNet2dTransp._up_block(features*2, features, padding_type, kern_size, name="dec1")

      self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)

      self.kern_size = kern_size

   def forward(self, x):
      #print('x.shape; '+str(x.shape))
      enc1 = self.encoder1(x)
      #print('enc1.shape; '+str(enc1.shape))
      enc2 = self.pool1(enc1)
      #print('enc2a.shape; '+str(enc2.shape))
      enc2 = self.encoder2(enc2)
      #print('enc2b.shape; '+str(enc2.shape))

      tmp = self.pool2(enc2)
      #print('bottlenecka.shape; '+str(tmp.shape))
      tmp = self.bottleneck(tmp)
      #print('bottleneckb.shape; '+str(tmp.shape))

      tmp = self.upsample2(tmp)
      #print('tmp.shape; '+str(tmp.shape))
      cust_pad = CustomPad2dTransp(self.kern_size)
      tmp = cust_pad.forward(tmp)
      #print('tmpb.shape; '+str(tmp.shape))
      tmp = self.upconv2(tmp)
      #print('tmpc.shape; '+str(tmp.shape))
      tmp = torch.cat((tmp, enc2), dim=1)
      #print('tmpd.shape; '+str(tmp.shape))
      tmp = self.decoder2(tmp)
      #print('tmpe.shape; '+str(tmp.shape))
      tmp = self.upsample1(tmp)
      #print('dec1a.shape; '+str(tmp.shape))
      tmp = cust_pad.forward(tmp)
      #print('dec1b.shape; '+str(tmp.shape))
      tmp = self.upconv1(tmp)
      #print('dec1c.shape; '+str(tmp.shape))
      tmp = torch.cat((tmp, enc1), dim=1)
      #print('dec1d.shape; '+str(tmp.shape))
      tmp = self.decoder1(tmp)
      #print('dec1e.shape; '+str(tmp.shape))

      # manually delete as this doesn't seem to be working...
      del x
      del enc1
      del enc2
      del cust_pad
      gc.collect()
      torch.cuda.empty_cache()

      return self.conv(tmp)

   @staticmethod
   def _down_block(in_channels, features, padding_type, kern_size, name):
      return nn.Sequential(
          OrderedDict(
              [
                  ( name + "pad1", CustomPad2dTransp(kern_size) ),
                  (
                      name + "conv1",
                      nn.Conv2d(
                          in_channels=in_channels,
                          out_channels=features,
                          kernel_size=kern_size,
                          bias=False,
                      ),
                  ),
                  (name + "norm1", nn.BatchNorm2d(num_features=features)),
                  (name + "relu1", nn.ReLU(inplace=True)),

                  ( name + "pad2", CustomPad2dTransp(kern_size) ),
                  (
                      name + "conv2",
                      nn.Conv2d(
                          in_channels=features,
                          out_channels=features,
                          kernel_size=kern_size,
                          bias=False,
                      ),
                  ),
                  (name + "norm2", nn.BatchNorm2d(num_features=features)),
                  (name + "relu2", nn.ReLU(inplace=True)),
              ]
          )
      )

   @staticmethod
   def _up_block(in_channels, features, padding_type, kern_size, name):
      return nn.Sequential(
          OrderedDict(
              [
                  ( name + "pad1", CustomPad2dTransp(kern_size) ),
                  (
                      name + "conv1",
                      nn.ConvTranspose2d(
                          in_channels=in_channels,
                          out_channels=features,
                          kernel_size=kern_size,
                          padding=( 0, int(kern_size-1) ),
                          bias=False,
                      ),
                  ),
                  (name + "norm1", nn.BatchNorm2d(num_features=features)),
                  (name + "relu1", nn.ReLU(inplace=True)),

                  ( name + "pad2", CustomPad2dTransp(kern_size) ),
                  (
                      name + "conv2",
                      nn.ConvTranspose2d(
                          in_channels=features,
                          out_channels=features,
                          kernel_size=kern_size,
                          padding=( 0, int(kern_size-1) ),
                          bias=False,
                      ),
                  ),
                  (name + "norm2", nn.BatchNorm2d(num_features=features)),
                  (name + "relu2", nn.ReLU(inplace=True)),
              ]
          )
      )

class UNet2dTranspExcLand(nn.Module):
   def __init__(self, in_channels, out_channels, padding_type, xdim, ydim, kern_size):
      super(UNet2dTranspExcLand, self).__init__()

      features = 2**(in_channels-1).bit_length()  # nearest power of two to input channels
      logging.info('No features ; '+str(features)+'\n')

      self.encoder1 = UNet2dTranspExcLand._down_block(in_channels, features, padding_type, kern_size, name="enc1")
      self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
      self.encoder2 = UNet2dTranspExcLand._down_block(features, features*2, padding_type, kern_size, name="enc2")
      self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

      self.bottleneck = UNet2dTranspExcLand._down_block(features*2, features*4, padding_type, kern_size, name="bottleneck")

      self.upsample2 = nn.Upsample(size=(int(ydim/2-(kern_size-1)*2),int(xdim/2)), mode='bilinear')
      self.upconv2 = nn.Conv2d(features*4, features*2, kernel_size=kern_size, stride=1)
      self.decoder2 = UNet2dTranspExcLand._up_block(features*2*2, features*2, padding_type, kern_size, name="dec1")
      self.upsample1 = nn.Upsample(size=(int(ydim-(kern_size)+1),int(xdim)), mode='bilinear')
      self.upconv1 = nn.Conv2d(features*2, features, kernel_size=kern_size, stride=1)
      self.decoder1 = UNet2dTranspExcLand._up_block(features*2, features, padding_type, kern_size, name="dec1")

      self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)

      self.kern_size = kern_size

   def forward(self, x):
      #print('x.shape; '+str(x.shape))
      enc1 = self.encoder1(x)
      #print('enc1.shape; '+str(enc1.shape))
      enc2 = self.pool1(enc1)
      #print('enc2a.shape; '+str(enc2.shape))
      enc2 = self.encoder2(enc2)
      #print('enc2b.shape; '+str(enc2.shape))

      tmp = self.pool2(enc2)
      #print('bottlenecka.shape; '+str(tmp.shape))
      tmp = self.bottleneck(tmp)
      #print('bottleneckb.shape; '+str(tmp.shape))

      tmp = self.upsample2(tmp)
      #print('tmp.shape; '+str(tmp.shape))
      cust_pad = CustomPad2dTransp(self.kern_size)
      tmp = cust_pad.forward(tmp)
      #print('tmpb.shape; '+str(tmp.shape))
      tmp = self.upconv2(tmp)
      #print('tmpc.shape; '+str(tmp.shape))
      tmp = torch.cat((tmp, enc2), dim=1)
      #print('tmpd.shape; '+str(tmp.shape))
      tmp = self.decoder2(tmp)
      #print('tmpe.shape; '+str(tmp.shape))
      tmp = self.upsample1(tmp)
      #print('dec1a.shape; '+str(tmp.shape))
      tmp = cust_pad.forward(tmp)
      #print('dec1b.shape; '+str(tmp.shape))
      tmp = self.upconv1(tmp)
      #print('dec1c.shape; '+str(tmp.shape))
      tmp = torch.cat((tmp, enc1), dim=1)
      #print('dec1d.shape; '+str(tmp.shape))
      tmp = self.decoder1(tmp)
      #print('dec1e.shape; '+str(tmp.shape))

      # manually delete as this doesn't seem to be working...
      del x
      del enc1
      del enc2
      del cust_pad
      gc.collect()
      torch.cuda.empty_cache()

      return self.conv(tmp)

   @staticmethod
   def _down_block(in_channels, features, padding_type, kern_size, name):
      return nn.Sequential(
          OrderedDict(
              [
                  ( name + "pad1", CustomPad2dTransp(kern_size) ),
                  (
                      name + "conv1",
                      nn.Conv2d(
                          in_channels=in_channels,
                          out_channels=features,
                          kernel_size=kern_size,
                          bias=False,
                      ),
                  ),
                  (name + "norm1", nn.BatchNorm2d(num_features=features)),
                  (name + "relu1", nn.ReLU(inplace=True)),

                  ( name + "pad2", CustomPad2dTransp(kern_size) ),
                  (
                      name + "conv2",
                      nn.Conv2d(
                          in_channels=features,
                          out_channels=features,
                          kernel_size=kern_size,
                          bias=False,
                      ),
                  ),
                  (name + "norm2", nn.BatchNorm2d(num_features=features)),
                  (name + "relu2", nn.ReLU(inplace=True)),
              ]
          )
      )

   @staticmethod
   def _up_block(in_channels, features, padding_type, kern_size, name):
      return nn.Sequential(
          OrderedDict(
              [
                  ( name + "pad1", CustomPad2dTransp(kern_size) ),
                  (
                      name + "conv1",
                      nn.ConvTranspose2d(
                          in_channels=in_channels,
                          out_channels=features,
                          kernel_size=kern_size,
                          padding=( 0, int(kern_size-1) ),
                          bias=False,
                      ),
                  ),
                  (name + "norm1", nn.BatchNorm2d(num_features=features)),
                  (name + "relu1", nn.ReLU(inplace=True)),

                  ( name + "pad2", CustomPad2dTransp(kern_size) ),
                  (
                      name + "conv2",
                      nn.ConvTranspose2d(
                          in_channels=features,
                          out_channels=features,
                          kernel_size=kern_size,
                          padding=( 0, int(kern_size-1) ),
                          bias=False,
                      ),
                  ),
                  (name + "norm2", nn.BatchNorm2d(num_features=features)),
                  (name + "relu2", nn.ReLU(inplace=True)),
              ]
          )
      )

class UNet3dTransp(nn.Module):
   def __init__(self, in_channels, out_channels, padding_type, xdim, ydim, zdim, kern_size):
      super(UNet3dTransp, self).__init__()

      features = 2**(in_channels-1).bit_length()  # nearest power of two to input channels
      logging.info('No features ; '+str(features)+'\n')

      self.encoder1 = UNet3dTransp._down_block(in_channels, features, padding_type, kern_size, name="enc1")
      self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
      self.encoder2 = UNet3dTransp._down_block(features, features*2, padding_type, kern_size, name="enc2")
      self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

      self.bottleneck = UNet3dTransp._down_block(features*2, features*4, padding_type, kern_size, name="bottleneck")

      self.upsample2 = nn.Upsample(size=(int(zdim/2-(kern_size-1)*2),int(ydim/2-(kern_size-1)*2),int(xdim/2)), mode='trilinear')
      self.upconv2 = nn.Conv3d(features*4, features*2, kernel_size=kern_size, stride=1)
      self.decoder2 = UNet3dTransp._up_block(features*2*2, features*2, padding_type, kern_size, name="dec1")
      self.upsample1 = nn.Upsample(size=(int(zdim-(kern_size-1)),int(ydim-(kern_size-1)),int(xdim)), mode='trilinear')
      self.upconv1 = nn.Conv3d(features*2, features, kernel_size=kern_size, stride=1)
      self.decoder1 = UNet3dTransp._up_block(features*2, features, padding_type, kern_size, name="dec1")

      self.conv = nn.Conv3d(in_channels=features, out_channels=out_channels, kernel_size=1)

      self.kern_size = kern_size

   def forward(self, x):
      #print('x.shape; '+str(x.shape))
      enc1 = self.encoder1(x)
      #print('enc1.shape; '+str(enc1.shape))
      enc2 = self.pool1(enc1)
      #print('enc2a.shape; '+str(enc2.shape))
      enc2 = self.encoder2(enc2)
      #print('enc2b.shape; '+str(enc2.shape))

      tmp = self.pool2(enc2)
      #print('bottlenecka.shape; '+str(tmp.shape))
      tmp = self.bottleneck(tmp)
      #print('bottleneckb.shape; '+str(tmp.shape))

      tmp = self.upsample2(tmp)
      #print('tmp.shape; '+str(tmp.shape))
      cust_pad = CustomPad3dTransp(self.kern_size)
      tmp = cust_pad.forward(tmp)
      #print('tmpb.shape; '+str(tmp.shape))
      tmp = self.upconv2(tmp)
      #print('tmpc.shape; '+str(tmp.shape))
      tmp = torch.cat((tmp, enc2), dim=1)
      #print('tmpd.shape; '+str(tmp.shape))
      tmp = self.decoder2(tmp)
      #print('tmpe.shape; '+str(tmp.shape))
      tmp = self.upsample1(tmp)
      #print('dec1a.shape; '+str(tmp.shape))
      tmp = cust_pad.forward(tmp)
      #print('dec1b.shape; '+str(tmp.shape))
      tmp = self.upconv1(tmp)
      #print('dec1c.shape; '+str(tmp.shape))
      tmp = torch.cat((tmp, enc1), dim=1)
      #print('dec1d.shape; '+str(tmp.shape))
      tmp = self.decoder1(tmp)
      #print('dec1e.shape; '+str(tmp.shape))

      # manually delete as this doesn't seem to be working...
      del x
      del enc1
      del enc2
      del cust_pad
      gc.collect()
      torch.cuda.empty_cache()

      return self.conv(tmp)

   @staticmethod
   def _down_block(in_channels, features, padding_type, kern_size, name):
      return nn.Sequential(
          OrderedDict(
              [
                  ( name + "pad1", CustomPad3dTransp(kern_size) ),
                  (
                      name + "conv1",
                      nn.Conv3d(
                          in_channels=in_channels,
                          out_channels=features,
                          kernel_size=kern_size,
                          bias=False,
                      ),
                  ),
                  (name + "norm1", nn.BatchNorm3d(num_features=features)),
                  (name + "relu1", nn.ReLU(inplace=True)),

                  ( name + "pad2", CustomPad3dTransp(kern_size) ),
                  (
                      name + "conv2",
                      nn.Conv3d(
                          in_channels=features,
                          out_channels=features,
                          kernel_size=kern_size,
                          bias=False,
                      ),
                  ),
                  (name + "norm2", nn.BatchNorm3d(num_features=features)),
                  (name + "relu2", nn.ReLU(inplace=True)),
              ]
          )
      )

   @staticmethod
   def _up_block(in_channels, features, padding_type, kern_size, name):
      return nn.Sequential(
          OrderedDict(
              [
                  ( name + "pad1", CustomPad3dTransp(kern_size) ),
                  (
                      name + "conv1",
                      nn.ConvTranspose3d(
                          in_channels=in_channels,
                          out_channels=features,
                          kernel_size=kern_size,
                          padding=(0,0,2),
                          bias=False,
                      ),
                  ),
                  (name + "norm1", nn.BatchNorm3d(num_features=features)),
                  (name + "relu1", nn.ReLU(inplace=True)),

                  ( name + "pad2", CustomPad3dTransp(kern_size) ),
                  (
                      name + "conv2",
                      nn.ConvTranspose3d(
                          in_channels=features,
                          out_channels=features,
                          kernel_size=kern_size,
                          padding=(0,0,2),
                          bias=False,
                      ),
                  ),
                  (name + "norm2", nn.BatchNorm3d(num_features=features)),
                  (name + "relu2", nn.ReLU(inplace=True)),
              ]
          )
      )


class UNet2d(nn.Module):
   def __init__(self, in_channels, out_channels, padding_type, kern_size):
      super(UNet2d, self).__init__()

      features = 2**(in_channels-1).bit_length()  # nearest power of two to input channels
      logging.info('No features ; '+str(features)+'\n')

      self.encoder1 = UNet2d._block(in_channels, features, padding_type, kern_size, name="enc1")
      self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
      self.encoder2 = UNet2d._block(features, features*2, padding_type, kern_size, name="enc2")
      self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

      self.bottleneck = UNet2d._block(features*2, features*4, padding_type, kern_size, name="bottleneck")

      self.upconv2 = nn.ConvTranspose2d(features*4, features*2, kernel_size=2, stride=2)
      self.decoder2 = UNet2d._block(features*2*2, features*2, padding_type, kern_size, name="dec2")
      self.upconv1 = nn.ConvTranspose2d(features*2, features, kernel_size=2, stride=2)
      self.decoder1 = UNet2d._block(features*2, features, padding_type, kern_size, name="dec1")

      self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)

   def forward(self, x):
      #print('x.shape; '+str(x.shape))
      x = x.to(dtype=torch.float)
      enc1 = self.encoder1(x)
      #print('enc1.shape; '+str(enc1.shape))
      enc2 = self.encoder2(self.pool1(enc1))
      #print('enc2.shape; '+str(enc2.shape))

      tmp = self.bottleneck(self.pool2(enc2))
      #print('bottleneck.shape; '+str(bottleneck.shape))

      tmp = self.upconv2(tmp)
      #print('tmp.shape  a; '+str(tmp.shape))
      tmp = torch.cat((tmp, enc2), dim=1)
      #print('tmp.shape  b; '+str(tmp.shape))
      tmp = self.decoder2(tmp)
      #print('tmp.shape  c; '+str(tmp.shape))
      tmp = self.upconv1(tmp)
      #print('tmp.shape  a; '+str(tmp.shape))
      tmp = torch.cat((tmp, enc1), dim=1)
      #print('tmp.shape  b; '+str(tmp.shape))
      tmp = self.decoder1(tmp)
      #print('tmp.shape  c; '+str(tmp.shape))

      # manualy delete as this doesn't seem to be working...
      del x
      del enc1
      del enc2
      gc.collect()
      torch.cuda.empty_cache()

      return self.conv(tmp)

   @staticmethod
   def _block(in_channels, features, padding_type, kern_size, name):
      return nn.Sequential(
          OrderedDict(
              [
                  ( name + "pad1", CustomPad2d(padding_type, kern_size) ),
                  (
                      name + "conv1",
                      nn.Conv2d(
                          in_channels=in_channels,
                          out_channels=features,
                          kernel_size=kern_size,
                          bias=False,
                      ),
                  ),
                  (name + "norm1", nn.BatchNorm2d(num_features=features)),
                  (name + "relu1", nn.ReLU(inplace=True)),

                  ( name + "pad2", CustomPad2d(padding_type, kern_size) ),
                  (
                      name + "conv2",
                      nn.Conv2d(
                          in_channels=features,
                          out_channels=features,
                          kernel_size=kern_size,
                          bias=False,
                      ),
                  ),
                  (name + "norm2", nn.BatchNorm2d(num_features=features)),
                  (name + "relu2", nn.ReLU(inplace=True)),
              ]
          )
      )


class UNet3d(nn.Module):
   def __init__(self, in_channels, out_channels, padding_type, kern_size):
      super(UNet3d, self).__init__()

      features = 2**(in_channels-1).bit_length()  # nearest power of two to input channels
      logging.info('No features ; '+str(features)+'\n')

      self.encoder1 = UNet3d._block(in_channels, features, padding_type, kern_size, name="enc1")
      self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
      self.encoder2 = UNet3d._block(features, features*2, padding_type, kern_size, name="enc2")
      self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

      self.bottleneck = UNet3d._block(features*2, features*4, padding_type, kern_size, name="bottleneck")

      self.upconv2 = nn.ConvTranspose3d(features*4, features*2, kernel_size=2, stride=2, output_padding=(1,0,0) )
      self.decoder2 = UNet3d._block(features*2*2, features*2, padding_type, kern_size, name="tmp")
      self.upconv1 = nn.ConvTranspose3d(features*2, features, kernel_size=2, stride=2)
      self.decoder1 = UNet3d._block(features*2, features, padding_type, kern_size, name="dec1")

      self.conv = nn.Conv3d(in_channels=features, out_channels=out_channels, kernel_size=1)

   def forward(self, x):
      #print('x.shape; '+str(x.shape))
      enc1 = self.encoder1(x)
      #print('enc1.shape; '+str(enc1.shape))
      enc2 = self.encoder2(self.pool1(enc1))
      #print('enc2.shape; '+str(enc2.shape))

      tmp = self.bottleneck(self.pool2(enc2))
      #print('bottleneck.shape; '+str(bottleneck.shape))

      tmp = self.upconv2(tmp)
      #print('tmp.shape  a; '+str(tmp.shape))
      tmp = torch.cat((tmp, enc2), dim=1)
      #print('tmp.shape  b; '+str(tmp.shape))
      tmp = self.decoder2(tmp)
      #print('tmp.shape  c; '+str(tmp.shape))
      tmp = self.upconv1(tmp)
      #print('dec1.shape  a; '+str(dec1.shape))
      tmp = torch.cat((tmp, enc1), dim=1)
      #print('dec1.shape  b; '+str(dec1.shape))
      tmp = self.decoder1(tmp)
      #print('dec1.shape  c; '+str(dec1.shape))

      # manually delete as this doesn't seem to be working...
      del x
      del enc1
      del enc2
      gc.collect()
      torch.cuda.empty_cache()

      return self.conv(tmp)

   @staticmethod
   def _block(in_channels, features, padding_type, kern_size, name):
      return nn.Sequential(
          OrderedDict(
              [
                  ( name + "pad1", CustomPad3d(padding_type, kern_size) ),
                  (
                      name + "conv1",
                      nn.Conv3d(
                          in_channels=in_channels,
                          out_channels=features,
                          kernel_size=kern_size,
                          bias=False,
                      ),
                  ),
                  (name + "norm1", nn.BatchNorm3d(num_features=features)),
                  (name + "relu1", nn.ReLU(inplace=True)),

                  ( name + "pad2", CustomPad3d(padding_type, kern_size) ),
                  (
                      name + "conv2",
                      nn.Conv3d(
                          in_channels=features,
                          out_channels=features,
                          kernel_size=kern_size,
                          bias=False,
                      ),
                  ),
                  (name + "norm2", nn.BatchNorm3d(num_features=features)),
                  (name + "relu2", nn.ReLU(inplace=True)),
              ]
          )
      )

def CreateModel(model_style, no_input_channels, no_target_channels, lr, seed_value, padding_type, xdim, ydim, zdim, kern_size, weight_decay=0):
   # inputs are (no_samples, 115channels, 100y, 240x).  (38 vertical levels, T, U, V 3d, Eta 2d)
   os.environ['PYTHONHASHSEED'] = str(seed_value)
   np.random.seed(seed_value)
   torch.manual_seed(seed_value)
   torch.cuda.manual_seed(seed_value)
   torch.cuda.manual_seed_all(seed_value)
   torch.backends.cudnn.enabled = False
   torch.backends.cudnn.deterministic = True
   torch.backends.cudnn.benchmark = False

   if model_style == 'UNet2dtransp':
      h = UNet2dTransp(no_input_channels, no_target_channels, padding_type, xdim, ydim, kern_size)
   elif model_style == 'UNet3dtransp':
      h = UNet3dTransp(no_input_channels, no_target_channels, padding_type, xdim, ydim, zdim, kern_size)
   elif model_style == 'UNet2d':
      h = UNet2d(no_input_channels, no_target_channels, padding_type, kern_size)
   elif model_style == 'UNet3d':
      h = UNet3d(no_input_channels, no_target_channels, padding_type, kern_size)
   else:
      raise RuntimeError('WARNING NO MODEL DEFINED')

   if torch.cuda.is_available():
       h = h.cuda()
   h = h.float()
 
   optimizer = torch.optim.Adam( h.parameters(), lr=lr, weight_decay=weight_decay )

   return h, optimizer

