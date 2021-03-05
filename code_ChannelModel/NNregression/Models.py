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

      #h = nn.Sequential(
      #   # downscale
      #   # ****Note scher uses a kernel size of 6, but no idea how he then manages to keep his x and y dimensions
      #   # unchanged over the conv layers....***
      #   nn.Conv2d(in_channels=no_input_channels, out_channels=64, kernel_size=(7,7), padding=(3,3)),
      #   nn.ReLU(True),
      #   nn.MaxPool2d((2,2)),
      #   nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(7,7),padding=(3,3)),
      #   nn.ReLU(True),
      #   nn.MaxPool2d((2,2)),
   
      #   nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=(7,7), padding=(3,3)),
      #   nn.ReLU(True),
   
      #   # upscale
      #   nn.Upsample(scale_factor=(2,2)),
      #   nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(7,7), padding=(3,3)),
      #   nn.ReLU(True),
      #   nn.Upsample(scale_factor=(2,2)),
      #   nn.Conv2d(in_channels=64, out_channels=no_target_channels, kernel_size=(7,7), padding=(3,3)),
      #   nn.ReLU(True)
      #    )
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

     # based on code at https://github.com/mateuszbuda/brain-segmentation-pytorch/blob/master/unet.py
     class UNet(nn.Module):
        def __init__(self, in_channels=no_input_channels, out_channels=no_target_channels):
           super(UNet, self).__init__()

           features = 128
           self.encoder1 = UNet._block(in_channels, features, name="enc1")
           self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
           self.encoder2 = UNet._block(features, features*2, name="enc2")
           self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
           #self.encoder3 = UNet._block(in_channels*4, in_channels*8 name="enc3")
           #self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
           #self.encoder4 = UNet._block(in_channels*8, in_channels*16 name="enc4")
           #self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

           self.bottleneck = UNet._block(features*2, features*4, name="bottleneck")

           #self.upconv4 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2, padding=1)
           #self.decoder4 = UNet._block(2*64, 64, name="dec4")
           #self.upconv3 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
           #self.decoder3 = UNet._block(2*64, 64, name="dec3")
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
           #enc3 = self.encoder3(self.pool2(enc2))
           #print('enc3.shape; '+str(enc3.shape))
           #enc4 = self.encoder4(self.pool3(enc3))
           ##print('enc4.shape; '+str(enc4.shape))

           #bottleneck = self.bottleneck(self.pool4(enc4))
           #print('bottleneck.shape; '+str(bottleneck.shape))

           #dec4 = self.upconv4(bottleneck)
           #print('dec4.shape  a; '+str(dec4.shape))
           #dec4 = torch.cat((dec4, enc4), dim=1)
           #print('dec4.shape  b; '+str(dec4.shape))
           #dec4 = self.decoder4(dec4)
           #print('dec4.shape  c; '+str(dec4.shape))
           #dec3 = self.upconv3(dec4)
           #print('dec3.shape  a; '+str(dec3.shape))
           #dec3 = torch.cat((dec3, enc3), dim=1)
           #print('dec3.shape  b; '+str(dec3.shape))
           #dec3 = self.decoder3(dec3)
           #print('dec3.shape  c; '+str(dec3.shape))

           bottleneck = self.bottleneck(self.pool2(enc2))
           #print('bottleneck.shape; '+str(bottleneck.shape))

           #dec2 = self.upconv2(dec3)
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
                       (
                           name + "conv1",
                           nn.Conv2d(
                               in_channels=in_channels,
                               out_channels=features,
                               kernel_size=7,
                               padding=3,
                               bias=False,
                           ),
                       ),
                       (name + "norm1", nn.BatchNorm2d(num_features=features)),
                       (name + "relu1", nn.ReLU(inplace=True)),
                       (
                           name + "conv2",
                           nn.Conv2d(
                               in_channels=features,
                               out_channels=features,
                               kernel_size=7,
                               padding=3,
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

   #   optimizer = torch.optim.Adam( h.parameters(), lr=lr )

   #elif model_style == 'UNet':
   ## Taken from tutorial at https://amaarora.github.io/2020/09/13/unet.html#u-net and amended

   #   class Block(nn.Module):
   #      def __init__(self, in_ch, out_ch):
   #          super(Block, self).__init__()
   #          self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=(7,7), padding=(3,3))
   #          self.relu  = nn.ReLU()
   #          self.conv2 = nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=(7,7), padding=(3,3))
   #      
   #      def forward(self, x):
   #          print('    in block forward')
   #          print('    '+str(x.shape))
   #          print('    '+str(self.conv1(x).shape))
   #          print('    '+str(self.relu(self.conv1(x)).shape))
   #          print('    '+str(self.conv2(self.relu(self.conv1(x))).shape))
   #          print('    end of block forward...')
   #          return self.conv2(self.relu(self.conv1(x)))
   #  
   #  
   #   class Encoder(nn.Module):
   #      def __init__(self, chs=(no_input_channels, 2*no_input_channels, 4*no_input_channels, 8*no_input_channels)):
   #          super(Encoder, self).__init__()
   #          self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)])
   #          self.pool       = nn.MaxPool2d(2)
   #      
   #      def forward(self, x):
   #          print('in encoder forward')
   #          ftrs = []
   #          for block in self.enc_blocks:
   #              print(x.shape)
   #              x = block(x)
   #              print(x.shape)
   #              ftrs.append(x)
   #              x = self.pool(x)
   #          print('end of encoder forward')
   #          print(len(ftrs))
   #          return x, ftrs
   #  
   #  
   #   class Decoder(nn.Module):
   #      def __init__(self, chs=(8*no_input_channels, 4*no_input_channels, 2*no_input_channels, no_target_channels)):
   #          super(Decoder, self).__init__()
   #          self.chs         = chs
   #          self.upconvs    = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i+1], 2, 2) for i in range(len(chs)-1)])
   #          self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)]) 
   #          
   #      def forward(self, x, encoder_features):
   #          print('in decoder forward')
   #          print(x.shape)
   #          for i in range(len(self.chs)-1):
   #              x        = self.upconvs[i](x)
   #              print(x.shape)
   #              #enc_ftrs = self.crop(encoder_features[i], x)
   #              enc_ftrs = encoder_features[i]
   #              print(enc_ftrs.shape)
   #              x        = torch.cat([x, enc_ftrs], dim=1)
   #              print(x.shape)
   #              x        = self.dec_blocks[i](x)
   #              print(x.shape)
   #          print('decoder forward working')
   #          return x
   #      
   #      #def crop(self, enc_ftrs, x):
   #      #    _, _, H, W = x.shape
   #      #    enc_ftrs   = transforms.CenterCrop([H, W])(enc_ftrs)
   #      #    return enc_ftrs
   #  
   #  
   #   class UNet(nn.Module):
   #      def __init__(self, enc_chs=(no_input_channels, 2*no_input_channels, 4*no_input_channels, 8*no_input_channels), 
   #                         dec_chs=(8*no_input_channels, 4*no_input_channels, 2*no_input_channels, no_target_channels), num_class=1, retain_dim=False, out_sz=(572,572)):
   #          super(UNet, self).__init__()
   #          self.encoder     = Encoder(enc_chs)
   #          self.decoder     = Decoder(dec_chs)
   #          self.head        = nn.Conv2d(dec_chs[-1], num_class, 1)
   #          self.retain_dim  = retain_dim
   #          self.pool       = nn.MaxPool2d(2)
   #          print('init done...')
   #  
   #      def forward(self, x):
   #          print('rachel')
   #          x, enc_ftrs = self.encoder(x)
   #          # should there be a 'bottleneck' layer here? I've added one...
   #          x = self.pool(x)
   #          out      = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
   #          out      = self.head(out)
   #          if self.retain_dim:
   #              out = F.interpolate(out, out_sz)
   #          return out

   #   unet = UNet()
   #   x    = torch.randn(1, 153, 100, 240)
   #   h = UNet()#.cuda()
   #   h = h.cuda()   
   #   unet(x)#.shape
   #   

   #   optimizer = torch.optim.Adam( h.parameters(), lr=lr )

   else:
      h=None
      optimizer=None
      print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
      print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
      print('WARNING NO MODEL DEFINED')
      print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
      print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
   
   return h, optimizer
