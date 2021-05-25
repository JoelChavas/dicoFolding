# -*- coding: utf-8 -*-
# /usr/bin/env python3
#
#  This software and supporting documentation are distributed by
#      Institut Federatif de Recherche 49
#      CEA/NeuroSpin, Batiment 145,
#      91191 Gif-sur-Yvette cedex
#      France
#
# This software is governed by the CeCILL license version 2 under
# French law and abiding by the rules of distribution of free software.
# You can  use, modify and/or redistribute the software under the
# terms of the CeCILL license version 2 as circulated by CEA, CNRS
# and INRIA at the following URL "http://www.cecill.info".
#
# As a counterpart to the access to the source code and  rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty  and the software's author,  the holder of the
# economic rights,  and the successive licensors  have only  limited
# liability.
#
# In this respect, the user's attention is drawn to the risks associated
# with loading,  using,  modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean  that it is complicated to manipulate,  and  that  also
# therefore means  that it is reserved for developers  and  experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and,  more generally, to use and operate it in the
# same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license version 2 and that you accept its terms.

""" beta-VAE

"""

######################################################################
# Imports and global variables definitions
######################################################################

import os
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stat
import time
import argparse
import torch
from tqdm import tqdm
import pandas as pd
from torch.autograd import Variable
import torch.nn as nn
from torchvision import models
#from torchsummary import summary

from deep_folding.preprocessing import *
from deep_folding.utils.pytorchtools import EarlyStopping

from ..postprocessing.test_tools import compute_loss, plot_loss

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


class VAE(nn.Module):
    def __init__(self, in_shape, n_latent, depth, norm, skeleton):
        super().__init__()
        self.skeleton = skeleton
        self.in_shape = in_shape
        self.n_latent = n_latent
        c,h,w,d = in_shape
        self.depth = depth
        self.norm = norm
        self.z_dim = h//2**depth # receptive field downsampled 2 times

        modules_encoder = []
        for step in range(depth):
            in_channels = 1 if step == 0 else out_channels
            out_channels = 16 if step == 0  else 16 * (2**step)
            modules_encoder.append(('conv%s' %step, nn.Conv3d(in_channels, out_channels,
                    kernel_size=3, stride=1, padding=1)))
            modules_encoder.append(('norm%s' %step, nn.BatchNorm3d(out_channels)))
            modules_encoder.append(('LeakyReLU%s' %step, nn.LeakyReLU()))
            modules_encoder.append(('conv%sa' %step, nn.Conv3d(out_channels, out_channels,
                    kernel_size=4, stride=2, padding=1)))
            modules_encoder.append(('norm%sa' %step, nn.BatchNorm3d(out_channels)))
            modules_encoder.append(('LeakyReLU%sa' %step, nn.LeakyReLU()))
        self.encoder = nn.Sequential(OrderedDict(modules_encoder))

        self.z_mean = nn.Linear(64 * self.z_dim**3, n_latent) # 8000 -> n_latent = 3
        self.z_var = nn.Linear(64 * self.z_dim**3, n_latent) # 8000 -> n_latent = 3
        self.z_develop = nn.Linear(n_latent, 64 * self.z_dim**3) # n_latent -> 8000

        modules_decoder = []
        for step in range(depth-1):
            in_channels = out_channels
            out_channels = in_channels // 2
            modules_decoder.append(('convTrans3d%s' %step, nn.ConvTranspose3d(in_channels,
                        out_channels, kernel_size=2, stride=2, padding=0)))
            modules_decoder.append(('normup%s' %step, nn.BatchNorm3d(out_channels)))
            modules_decoder.append(('ReLU%s' %step, nn.ReLU()))
            modules_decoder.append(('convTrans3d%sa' %step, nn.ConvTranspose3d(out_channels,
                        out_channels, kernel_size=3, stride=1, padding=1)))
            modules_decoder.append(('normup%sa' %step, nn.BatchNorm3d(out_channels)))
            modules_decoder.append(('ReLU%sa' %step, nn.ReLU()))
        modules_decoder.append(('convtrans3dn', nn.ConvTranspose3d(16, 1, kernel_size=2,
                        stride=2, padding=0)))
        if self.skeleton:
            modules_decoder.append(('conv_final', nn.Conv3d(1, 2, kernel_size=1, stride=1)))
        else:
            modules_decoder.append(('sigmoid', nn.Sigmoid()))
        self.decoder = nn.Sequential(OrderedDict(modules_decoder))
        self.weight_initialization()

    def weight_initialization(self):
        """
        Initializes model parameters according to Gaussian Glorot initialization
        """
        for module in self.modules():
            if isinstance(module, nn.ConvTranspose3d) or isinstance(module, nn.Conv3d):
                #print('weight module conv')
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                #print('weight module batchnorm')
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)

    def sample_z(self, mean, logvar):
        device = torch.device("cpu")
        stddev = torch.exp(0.5 * logvar)
        noise = Variable(torch.randn(stddev.size(), device=device))
        return (noise * stddev) + mean

    def encode(self, x):
        x = self.encoder(x)
        x = nn.functional.normalize(x, p=2)
        x = x.view(x.size(0), -1)
        mean = self.z_mean(x)
        var = self.z_var(x)
        return mean, var

    def decode(self, z):
        out = self.z_develop(z)
        out = out.view(z.size(0), 16 * 2**(self.depth-1), self.z_dim, self.z_dim, self.z_dim)
        out = self.decoder(out)
        return out

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.sample_z(mean, logvar)
        out = self.decode(z)
        return out, mean, logvar, z


def vae_loss(output, input, mean, logvar, loss_func, kl_weight):
    #print('entering loss')
    #print('input shape', input.shape)
    #print('output shape', output.shape)
    #print('input unsqueezed shape', input.unsqueeze(0).shape)
    #print(np.unique(output.detach().cpu()), np.unique(input.detach().cpu()))
    recon_loss = loss_func(output, input)

    #print(mean.shape, logvar.shape)
    #print('recon_loss ok')
    """kl_loss = torch.mean(0.5 * torch.sum(
                        torch.exp(logvar) + mean**2 - 1. - logvar, 1),
                        dim=0)"""
    kl_loss = -0.5 * torch.sum(-torch.exp(logvar) - mean**2 + 1. + logvar)
    #kl_loss = -0.5 * torch.sum(-torch.exp(logvar) - mean.pow(2) + 1. + logvar)
    #print('kl ok')
    return recon_loss + kl_weight * kl_loss


class ModelTrainer():

    def __init__(self, model, train_loader, val_loader, loss_func, nb_epoch, optimizer, kl_weight,
                n_latent, depth, norm, skeleton, root_dir):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_func = loss_func
        self.nb_epoch = nb_epoch
        self.optimizer = optimizer
        self.kl_weight = kl_weight
        #self.lr = lr
        self.n_latent = n_latent
        self.depth = depth
        self.norm = norm
        self.skeleton = skeleton
        self.root_dir = root_dir

    def train(self):
        id_arr, phase_arr, input_arr, output_arr = [], [], [], []
        conv1, conv2, conv3 = [], [], []
        self.list_loss_train, self.list_loss_val = [], []
        device = torch.device("cuda", index=0)
        early_stopping = EarlyStopping(patience=20, verbose=True)

        print('skeleton', self.skeleton)

        for epoch in range(self.nb_epoch):
            loss_tot_train, loss_tot_val = 0, 0
            self.model.train()
            for inputs, path in self.train_loader:
                self.optimizer.zero_grad()
                #print(inputs.shape)
                #print(path)
                #print('input', np.unique(inputs))
                #if -1 in np.unique(inputs.detach().cpu()):
                    #for i in range(len(inputs)):
                        #print(path[i], np.unique(inputs[i].detach().cpu()))
                inputs = Variable(inputs).to(device, dtype=torch.float32)
                #print('input 2', np.unique(inputs.detach().cpu()))
                #print(inputs.shape)
                #print('ici')
                output, mean, logvar, z = self.model(inputs)
                #print('input ter', np.unique(inputs.detach().cpu()))
                #print('output',np.unique(output.detach().cpu()))
                #print('outpu', output.shape)
                if self.skeleton:
                    #print(1)
                    #target = inputs.long()
                    #print('input bis', np.unique(inputs.detach().cpu()))
                    target = torch.squeeze(inputs, dim=1).long()
                    #print('target', np.unique(target.cpu()))
                    #print(output.shape,target.shape,mean.shape,logvar.shape)
                    assert(len(np.unique(target.cpu())<=2))
                    loss_train = vae_loss(output, target, mean, logvar, self.loss_func,
                                         kl_weight=self.kl_weight)
                    output = torch.argmax(output, dim=1)


                else:
                    #print('ici main')
                    loss_train = vae_loss(output, inputs, mean, logvar, self.loss_func,
                                      kl_weight=self.kl_weight)
                    #print(loss_train)

                loss_tot_train += loss_train.item()

                #self.optimizer.zero_grad()
                loss_train.backward()
                self.optimizer.step()

            if epoch == self.nb_epoch-1:
                phase = 'train'
                for k in range(len(path)):
                    id_arr.append(path[k])
                    phase_arr.append(phase)
                    input_arr.append(np.array(np.squeeze(inputs[k]).cpu().detach().numpy()))
                    output_arr.append(np.squeeze(output[k]).cpu().detach().numpy())

            self.model.eval()
            for inputs, path in self.val_loader:
                #print(path)
                inputs = Variable(inputs).to(device, dtype=torch.float32)

                output, mean, logvar, z = self.model(inputs)
                if self.skeleton:
                    target = torch.squeeze(inputs, 0).long()
                    loss_val = vae_loss(output, target, mean, logvar, self.loss_func,
                                         kl_weight=self.kl_weight)
                    output = torch.argmax(output, dim=1)
                else:
                    loss_val = vae_loss(output, inputs, mean, logvar, self.loss_func,
                                    self.kl_weight)
                loss_tot_val += loss_val.item()

            self.list_loss_train.append(loss_tot_train/len(self.train_loader))
            self.list_loss_val.append(loss_tot_val/len(self.val_loader))

            if epoch == self.nb_epoch-1:
                phase = 'val'
                for k in range(len(path)):
                    id_arr.append(path[k])
                    phase_arr.append(phase)
                    input_arr.append(np.array(np.squeeze(inputs[k]).cpu().detach().numpy()))
                    output_arr.append(np.squeeze(output[k]).cpu().detach().numpy())

            print('epoch [{}/{}], loss_train:{:.4f}, loss_val:{:.4f}'.format(epoch,
            self.nb_epoch, loss_tot_train/len(self.train_loader), loss_tot_val/len(self.val_loader)))

            if early_stopping.early_stop:
                print("EarlyStopping")
                phase = 'val'
                for k in range(len(path)):
                    id_arr.append(path[k])
                    phase_arr.append(phase)
                    input_arr.append(np.array(np.squeeze(inputs[k]).cpu().detach().numpy()))
                    output_arr.append(np.squeeze(output[k]).cpu().detach().numpy())

        for key, array in {'input': input_arr, 'output' : output_arr,
                'phase': phase_arr, 'id': id_arr}.items():
                    np.save(self.root_dir+key, np.array([array]))

        plot_loss(self.list_loss_train[1:], self.list_loss_val[1:], self.root_dir)
        return min(self.list_loss_train), min(self.list_loss_val), id_arr, phase_arr, input_arr, output_arr


class ModelTester():

    def __init__(self, model, dico_set_loaders, loss_func, kl_weight,
                n_latent, depth, norm, skeleton, root_dir):
        self.model = model
        self.dico_set_loaders = dico_set_loaders
        self.loss_func = loss_func
        self.kl_weight = kl_weight
        self.n_latent = n_latent
        self.depth = depth
        self.norm = norm
        self.skeleton = skeleton
        self.root_dir = root_dir


    def test(self):
        id_arr, input_arr, phase_arr, output_arr = [], [], [], []
        self.list_loss_train, self.list_loss_val = [], []
        device = torch.device("cuda", index=0)

        results = {k:{} for k in self.dico_set_loaders.keys()}

        for loader_name, loader in self.dico_set_loaders.items():
            print(loader_name)
            self.model.eval()
            with torch.no_grad():
                for inputs, path in loader:
                    inputs = Variable(inputs).to(device, dtype=torch.float32)

                    output, mean, logvar, z = self.model(inputs)

                    if self.skeleton:
                        target = torch.squeeze(inputs, dim=0).long()
                        loss = vae_loss(output, target, mean, logvar, self.loss_func,
                                         kl_weight=self.kl_weight)
                        output = torch.argmax(output, dim=1)
                    else:
                        loss = vae_loss(output, inputs, mean, logvar, self.loss_func,
                                              kl_weight=self.kl_weight)

                    results[loader_name][path] = (loss.item(), output, inputs, np.array(torch.squeeze(z, dim=0).cpu().detach().numpy()))

                    for k in range(len(path)):
                        id_arr.append(path)
                        input_arr.append(np.array(np.squeeze(inputs).cpu().detach().numpy()))
                        output_arr.append(np.squeeze(output).cpu().detach().numpy())
                        phase_arr.append(loader_name)

        for key, array in {'input': input_arr, 'output' : output_arr,
                            'phase': phase_arr, 'id': id_arr}.items():
            np.save(self.root_dir+key+'val', np.array([array]))
        return results
