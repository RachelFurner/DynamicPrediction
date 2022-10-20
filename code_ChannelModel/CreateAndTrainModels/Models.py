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
      #print('input.shape; '+str(x.shape))
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

class UNet2dInterp(nn.Module):
   def __init__(self, in_channels, out_channels, padding_type, xdim, ydim, kern_size):
      super(UNet2dInterp, self).__init__()

      features = 2**(in_channels-1).bit_length()  # nearest power of two to input channels
      logging.info('No features ; '+str(features)+'\n')

      self.encoder1 = UNet2dInterp._block(in_channels, features, padding_type, kern_size, name="enc1")
      self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
      self.encoder2 = UNet2dInterp._block(features, features*2, padding_type, kern_size, name="enc2")
      self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

      self.bottleneck = UNet2dInterp._block(features*2, features*4, padding_type, kern_size, name="bottleneck")

      self.upsample2 = nn.Upsample(size=(int(ydim/2-(kern_size-1)*2),int(xdim/2)), mode='bilinear')
      self.upconv2 = nn.Conv2d(features*4, features*2, kernel_size=kern_size, stride=1)
      self.decoder2 = UNet2dInterp._block(features*2*2, features*2, padding_type, kern_size, name="dec1")
      self.upsample1 = nn.Upsample(size=(int(ydim-(kern_size-1)),int(xdim)), mode='bilinear')
      self.upconv1 = nn.Conv2d(features*2, features, kernel_size=kern_size, stride=1)
      self.decoder1 = UNet2dInterp._block(features*2, features, padding_type, kern_size, name="dec1")
      self.upsample0 = nn.Upsample(size=(ydim,xdim), mode='bilinear')

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
      tmp = self.upsample0(tmp)
      #print('dec1f.shape; '+str(tmp.shape))

      # manually delete as this doesn't seem to be working...
      del x
      del enc1
      del enc2
      del cust_pad
      gc.collect()
      torch.cuda.empty_cache()

      return self.conv(tmp)

   @staticmethod
   def _block(in_channels, features, padding_type, kern_size, name):
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
#--------------------------------------------------------------------------------------------------------------
# Code for ConvLSTM, taken from https://github.com/ndrplz/ConvLSTM_pytorch/blob/master/convlstm.py and amended
#--------------------------------------------------------------------------------------------------------------

class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=(kernel_size[0] // 2, kernel_size[1] // 2),
                              bias=False)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
 
    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        #print('')
        #print('in ConvLSTMCell')
        #print('h_cur.shape; '+str(h_cur.shape))
        #print('c_cur.shape; '+str(c_cur.shape))
        #print('input_tensor.shape; '+str(input_tensor.shape))

        h_cur = h_cur.to(self.device)
        c_cur = c_cur.to(self.device)

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        #print('combined.shape; '+str(combined.shape))
        combined_conv = self.conv(combined)
        #print('combined_conv.shape; '+str(combined_conv.shape))
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        del input_tensor
        del cur_state
        del h_cur
        del c_cur
        del combined
        del combined_conv 
        del cc_i
        del cc_f
        del cc_o
        del cc_g
        del i
        del f
        del o
        del g
        gc.collect()
        torch.cuda.empty_cache()
        
        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):

    """
    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.
    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i]
                                          ))
        
        self.cell_list = nn.ModuleList(cell_list)

        del kernel_size
        del hidden_dim
        del cell_list
        del cur_input_dim
        gc.collect()
        torch.cuda.empty_cache()

    def forward(self, input_tensor):
        """
        Parameters
        ----------
        input_tensor: todo
            5-D Tensor of shape (b, t, c, h, w)
        Returns
        -------
        last_state_list, layer_output
        """
        b, _, _, h, w = input_tensor.size()

        hidden_state = self._init_hidden(batch_size=b,
                                         image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        del b
        del w
        del hidden_state
        del layer_output_list
        del last_state_list
        del seq_len
        del cur_layer_input
        del c
        del output_inner
        del layer_output
        gc.collect()
        torch.cuda.empty_cache()

        #return layer_output_list, last_state_list
        return h

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

class UNetConvLSTM(nn.Module):

    """
    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.
    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, in_channels, out_channels, xdim, ydim, kernel_size):
        super(UNetConvLSTM, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.features = 2**(in_channels-1).bit_length()  # nearest power of two to input channels

        # My orginal UNet has 2 convuolutions per 'block/cell', along with ReLu and batch Norm. And the upcells are
        # different to the downcells...could try to move closer to that set up.

        self.ConvLSTMCell1 = ConvLSTMCell(input_dim=self.in_channels, hidden_dim=self.features, kernel_size=self.kernel_size)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.ConvLSTMCell2 = ConvLSTMCell(input_dim=self.features, hidden_dim=self.features*2, kernel_size=self.kernel_size)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.ConvLSTMCell3 = ConvLSTMCell(input_dim=self.features*2, hidden_dim=self.features*4, kernel_size=self.kernel_size)

        self.upsample2 = nn.Upsample(size=(int(ydim/2),int(xdim/2)), mode='bilinear')
        self.ConvLSTMCell4 = ConvLSTMCell(input_dim=self.features*4+self.features*2, hidden_dim=self.features*2, kernel_size=self.kernel_size)
        self.upsample1 = nn.Upsample(size=(int(ydim),int(xdim)), mode='bilinear')
        self.ConvLSTMCell5 = ConvLSTMCell(input_dim=self.features*2+self.features, hidden_dim=self.features, kernel_size=self.kernel_size)

        self.ConvLSTMCell6 = ConvLSTMCell(input_dim=self.features, hidden_dim=out_channels, kernel_size=self.kernel_size)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, input_tensor):
        """
        Parameters
        ----------
        input_tensor: todo
            5-D Tensor of shape (b, t, c, h, w)
        Returns
        -------
        last_state_list, layer_output
        """
        b, _, _, ydim, xdim = input_tensor.size()

        seq_len = input_tensor.size(1)

        h = torch.zeros((b, self.features, ydim, xdim), device=self.device)
        c = torch.zeros((b, self.features, ydim, xdim), device=self.device)
        output_enc1 = []
       
        for t in range(seq_len):
            #print('t ; '+str(t))
            h, c = self.ConvLSTMCell1(input_tensor=input_tensor[:, t, :, :, :], cur_state=[h, c])
            output_enc1.append(h)
        enc1 = torch.stack(output_enc1, dim=1)
        #print('enc1.shape; '+str(enc1.shape))

        h = torch.zeros((b, self.features*2, int(ydim/2), int(xdim/2) ))
        c = torch.zeros((b, self.features*2, int(ydim/2), int(xdim/2) ))
        output_enc2 = []
        for t in range(seq_len):
            #print('t ; '+str(t))
            pooled_enc1 = self.pool1(enc1[:, t, :, :, :])
            h, c = self.ConvLSTMCell2(input_tensor=pooled_enc1, cur_state=[h, c])
            output_enc2.append(h)
        enc2 = torch.stack(output_enc2, dim=1)
        #print('enc2.shape; '+str(enc2.shape))

        h = torch.zeros((b, self.features*4, int(ydim/4), int(xdim/4) ))
        c = torch.zeros((b, self.features*4, int(ydim/4), int(xdim/4) ))
        output_bottleneck = []
        for t in range(seq_len):
            #print('t ; '+str(t))
            pooled_enc2 = self.pool2(enc2[:, t, :, :, :])
            h, c = self.ConvLSTMCell3(input_tensor=pooled_enc2, cur_state=[h, c])
            output_bottleneck.append(h)
        bottleneck = torch.stack(output_bottleneck, dim=1)
        #print('bottleneck.shape; '+str(bottleneck.shape))

        h = torch.zeros((b, self.features*2, int(ydim/2), int(xdim/2) ))
        c = torch.zeros((b, self.features*2, int(ydim/2), int(xdim/2) ))
        output_dec2 = []
        for t in range(seq_len):
            #print('t ; '+str(t))
            upsampled_bottleneck = self.upsample2(bottleneck[:, t, :, :, :])
            #print('upsampled_bottleneck.shape '+str(upsampled_bottleneck.shape))
            #print('enc2[:, t, :, :, :].shape '+str(enc2[:, t, :, :, :].shape))
            h, c = self.ConvLSTMCell4(input_tensor=torch.cat((upsampled_bottleneck, enc2[:, t, :, :, :]),dim=1), cur_state=[h, c])
            output_dec2.append(h)
        dec2 = torch.stack(output_dec2, dim=1)
        #print('dec2.shape; '+str(dec2.shape))

        h = torch.zeros((b, self.features, ydim, xdim))
        c = torch.zeros((b, self.features, ydim, xdim))
        output_dec1 = []
        for t in range(seq_len):
            #print('t ; '+str(t))
            upsampled_dec2 = self.upsample1(dec2[:, t, :, :, :])
            h, c = self.ConvLSTMCell5(input_tensor=torch.cat((upsampled_dec2, enc1[:, t, :, :, :]),dim=1), cur_state=[h, c])
            output_dec1.append(h)
        dec1 = torch.stack(output_dec1, dim=1)
        #print('dec1.shape; '+str(dec1.shape))

        h = torch.zeros((b, self.out_channels, ydim, xdim))
        c = torch.zeros((b, self.out_channels, ydim, xdim))
        output_final = []
        for t in range(seq_len):
            #print('t ; '+str(t))
            h, c = self.ConvLSTMCell6(input_tensor=dec1[:, t, :, :, :], cur_state=[h, c])
            output_final.append(h)
        final = torch.stack(output_final, dim=1)
        #print('final.shape; '+str(final.shape))

        del b
        del xdim
        del ydim 
        del seq_len
        del h
        del c
        del output_enc1
        del enc1
        del pooled_enc1
        del output_enc2
        del enc2
        del pooled_enc2
        del output_bottleneck
        del bottleneck
        del upsampled_bottleneck
        del output_dec2
        del dec2
        del upsampled_dec2
        del output_dec1
        del dec1
        del output_final
        gc.collect()
        torch.cuda.empty_cache()

        return final[:,-1,:,:,:]

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param



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
   elif model_style == 'UNet2dinterp':
      h = UNet2dInterp(no_input_channels, no_target_channels, padding_type, xdim, ydim, kern_size)
   elif model_style == 'UNet3dtransp':
      h = UNet3dTransp(no_input_channels, no_target_channels, padding_type, xdim, ydim, zdim, kern_size)
   elif model_style == 'UNet2d':
      h = UNet2d(no_input_channels, no_target_channels, padding_type, kern_size)
   elif model_style == 'UNet3d':
      h = UNet3d(no_input_channels, no_target_channels, padding_type, kern_size)
   elif model_style == 'ConvLSTM':
      features = 2**(no_input_channels-1).bit_length()
      h = ConvLSTM(no_input_channels, [features, features*2, features, no_target_channels], [(3,3),(3,3),(3,3),(3,3)], 4)
   elif model_style == 'UNetConvLSTM':
      h = UNetConvLSTM(no_input_channels, no_target_channels, xdim, ydim, kern_size)
   else:
      raise RuntimeError('WARNING NO MODEL DEFINED')

   if torch.cuda.is_available():
       h = h.cuda()
   h = h.float()
 
   optimizer = torch.optim.Adam( h.parameters(), lr=lr, weight_decay=weight_decay )

   return h, optimizer

#-----------------#
# Models for GANs #
#-----------------#

#class Block(nn.Module):
#    def __init__(self, in_channels, out_channels, down=True, act="relu", use_dropout=False):
#        super(Block, self).__init__()
#        self.conv = nn.Sequential(
#            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False, padding_mode="reflect")
#            if down
#            else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
#            nn.BatchNorm2d(out_channels),
#            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2),
#        )
#
#        self.use_dropout = use_dropout
#        self.dropout = nn.Dropout(0.5)
#        self.down = down
#
#    def forward(self, x):
#        x = self.conv(x)
#        return self.dropout(x) if self.use_dropout else x
#
#
#class Generator(nn.Module):
#    def __init__(self, in_channels=3, features=64):
#        super().__init__()
#        self.initial_down = nn.Sequential(
#            nn.Conv2d(in_channels, features, 4, 2, 1, padding_mode="reflect"),
#            nn.LeakyReLU(0.2),
#        )
#        self.down1 = Block(features, features * 2, down=True, act="leaky", use_dropout=False)
#        self.down2 = Block(
#            features * 2, features * 4, down=True, act="leaky", use_dropout=False
#        )
#        self.down3 = Block(
#            features * 4, features * 8, down=True, act="leaky", use_dropout=False
#        )
#        self.down4 = Block(
#            features * 8, features * 8, down=True, act="leaky", use_dropout=False
#        )
#        self.down5 = Block(
#            features * 8, features * 8, down=True, act="leaky", use_dropout=False
#        )
#        self.down6 = Block(
#            features * 8, features * 8, down=True, act="leaky", use_dropout=False
#        )
#        self.bottleneck = nn.Sequential(
#            nn.Conv2d(features * 8, features * 8, 4, 2, 1), nn.ReLU()
#        )
#
#        self.up1 = Block(features * 8, features * 8, down=False, act="relu", use_dropout=True)
#        self.up2 = Block(
#            features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True
#        )
#        self.up3 = Block(
#            features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True
#        )
#        self.up4 = Block(
#            features * 8 * 2, features * 8, down=False, act="relu", use_dropout=False
#        )
#        self.up5 = Block(
#            features * 8 * 2, features * 4, down=False, act="relu", use_dropout=False
#        )
#        self.up6 = Block(
#            features * 4 * 2, features * 2, down=False, act="relu", use_dropout=False
#        )
#        self.up7 = Block(features * 2 * 2, features, down=False, act="relu", use_dropout=False)
#        self.final_up = nn.Sequential(
#            nn.ConvTranspose2d(features * 2, in_channels, kernel_size=4, stride=2, padding=1),
#            nn.Tanh(),
#        )
#
#    def forward(self, x):
#        d1 = self.initial_down(x)
#        d2 = self.down1(d1)
#        d3 = self.down2(d2)
#        d4 = self.down3(d3)
#        d5 = self.down4(d4)
#        d6 = self.down5(d5)
#        d7 = self.down6(d6)
#        bottleneck = self.bottleneck(d7)
#        up1 = self.up1(bottleneck)
#        up2 = self.up2(torch.cat([up1, d7], 1))
#        up3 = self.up3(torch.cat([up2, d6], 1))
#        up4 = self.up4(torch.cat([up3, d5], 1))
#        up5 = self.up5(torch.cat([up4, d4], 1))
#        up6 = self.up6(torch.cat([up5, d3], 1))
#        up7 = self.up7(torch.cat([up6, d2], 1))
#        return self.final_up(torch.cat([up7, d1], 1))
#
#class CNNBlock(nn.Module):
#    def __init__(self, in_channels, out_channels, stride):
#        super(CNNBlock, self).__init__()
#        self.conv = nn.Sequential(
#            nn.Conv2d(
#                in_channels, out_channels, 4, stride, 1, bias=False, padding_mode="reflect"
#            ),
#            nn.BatchNorm2d(out_channels),
#            nn.LeakyReLU(0.2),
#        )
#
#    def forward(self, x):
#        return self.conv(x)
#
#class Discriminator(nn.Module):
#    def __init__(self, in_channels, kern_size):
#        super().__init__()
#        in_channels = in_channels*2 # We cat the input and prediction/target together
#        features = 2**(in_channels-1).bit_length()  # nearest power of two to input channels
#        self.layer1 = nn.Sequential(
#            nn.Conv2d(
#                in_channels * 2,
#                features,
#                kernel_size=kern_size,
#                stride=2,
#                padding_mode=padding_type, 
#            ),
#            nn.LeakyReLU(0.2),
#        )
#
#        self.layer2 = CNNBlock(features, features*2, stride=2)
#        self.layer3 = CNNBlock(features*2, features*4, stride=2)
#        self.layer4 = CNNBlock(features*4, features*8, stride=1)
#        self.layer5 = nn.Conv2d(features*8, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect")
#
#    def forward(self, x, y):
#        x = torch.cat([x, y], dim=1)
#        x = self.layer1(x)
#        x = self.layer2(x)
#        x = self.layer3(x)
#        x = self.layer4(x)
#        x = self.layer5(x)
#        return x
#
