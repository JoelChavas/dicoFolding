#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
"""
A test to just analyze randomly generated input images
"""

import torch
import numpy as np
from dicoFolding.losses import NTXenLoss
from dicoFolding.models.densenet import DenseNet
from sklearn.manifold import TSNE

from dicoFolding.postprocessing.visualize_tsne import plot_tsne
from dicoFolding.postprocessing.visualize_tsne import plot_img
from dicoFolding.postprocessing.visualize_tsne import plot_output

from toolz.itertoolz import last, first

class ContrastiveLearnerTest(DenseNet):

    def __init__(self, config, mode, drop_rate, sample_data):
        super(ContrastiveLearnerTest, self).__init__(growth_rate=32,
                                                 block_config=(6, 12, 24, 16),
                                                 num_init_features=64,
                                                 mode=mode,
                                                 drop_rate=drop_rate)
        self.config = config
        self.sample_data = sample_data
        self.sample_i = []
        self.sample_j = []
        self.val_sample_i = []
        self.val_sample_j = []
         
    def custom_histogram_adder(self):

        # iterating through all parameters
        for name, params in self.named_parameters():

            self.logger.experiment.add_histogram(
                name, params, self.current_epoch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.config.lr,
                                     weight_decay=self.config.weight_decay)
        return optimizer

    def nt_xen_loss(self, z_i, z_j):
        loss = NTXenLoss(temperature=self.config.temperature,
                         return_logits=True)
        return loss.forward(z_i, z_j)

    def training_step(self, train_batch, batch_idx):
        (inputs, filenames) = train_batch
        if batch_idx == 0:
            self.sample_i.append(inputs[:, 0, :].cpu())
            self.sample_j.append(inputs[:, 1, :].cpu())

    def training_epoch_end(self, outputs):
        image_input_i = plot_img(self.sample_i[-1], buffer=True)
        self.logger.experiment.add_image(
            'input_i train', image_input_i, self.current_epoch)
        image_input_j = plot_img(self.sample_j[-1], buffer=True)
        self.logger.experiment.add_image(
            'input_j train', image_input_j, self.current_epoch)

    def validation_step(self, val_batch, batch_idx):
        (inputs, filenames) = val_batch
        if batch_idx == 0:
            self.val_sample_i.append(inputs[:, 0, :].cpu())
            self.val_sample_j.append(inputs[:, 1, :].cpu())

    def validation_epoch_end(self, outputs):
        image_input_i = plot_img(self.val_sample_i[-1], buffer=True)
        self.logger.experiment.add_image(
            'input_i val', image_input_i, self.current_epoch)
        image_input_j = plot_img(self.val_sample_j[-1], buffer=True)
        self.logger.experiment.add_image(
            'input_j val', image_input_j, self.current_epoch)

