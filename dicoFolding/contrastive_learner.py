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
Some helper functions are taken from:
https://learnopencv.com/tensorboard-with-pytorch-lightning

"""
import PIL
import torch
import torchvision
from dicoFolding.losses import NTXenLoss
from dicoFolding.models.densenet import DenseNet
from torchvision.transforms import ToTensor

from dicoFolding.postprocessing.visualize_tsne import compute_tsne
from dicoFolding.postprocessing.visualize_tsne import plot_tsne


class ContrastiveLearner(DenseNet):

    def __init__(self, config, mode, drop_rate, sample_data):
        super(ContrastiveLearner, self).__init__(growth_rate=32,
                                                 block_config=(6, 12, 24, 16),
                                                 num_init_features=64,
                                                 mode=mode,
                                                 drop_rate=drop_rate)
        self.config = config
        self.sample_data = sample_data

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
        z_i = self.forward(inputs[:, 0, :])
        z_j = self.forward(inputs[:, 1, :])
        batch_loss, _, _ = self.nt_xen_loss(z_i, z_j)
        self.log('train_loss', float(batch_loss))

        # Only computes graph on first step
        if self.global_step == 1:
            self.logger.experiment.add_graph(self, inputs[:, 0, :])

        # logs- a dictionary
        logs = {"train_loss": float(batch_loss)}

        batch_dictionary = {
            # REQUIRED: It ie required for us to return "loss"
            "loss": batch_loss,

            # optional for batch logging purposes
            "log": logs,

        }

        return batch_dictionary

    # On n'a pas accès ici à toutes les données
    # def training-epoch_end(self, outputs):
    #     X_tsne = compute_tsne(data_module.train_dataloader(), model)
    #     plot_tsne(X_tsne)

    def training_epoch_end(self, outputs):
        X_tsne = compute_tsne(self.sample_data.train_dataloader(), self)
        buf = plot_tsne(X_tsne, buffer=True)
        image = PIL.Image.open(buf)
        image = ToTensor()(image).unsqueeze(0)[0]
        self.logger.experiment.add_image(
            'TSNE image', image, self.current_epoch)

        # calculating average loss
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        # logging histograms
        self.custom_histogram_adder()

        # logging using tensorboard logger
        self.logger.experiment.add_scalar("Loss/Train",
                                          avg_loss,
                                          self.current_epoch)

    def validation_step(self, val_batch, batch_idx):
        (inputs, filenames) = val_batch
        z_i = self.forward(inputs[:, 0, :])
        z_j = self.forward(inputs[:, 1, :])
        batch_loss, logits, target = self.nt_xen_loss(z_i, z_j)
        self.log('val_loss', float(batch_loss))

        # logs- a dictionary
        logs = {"val_loss": float(batch_loss)}

        batch_dictionary = {
            # REQUIRED: It ie required for us to return "loss"
            "loss": batch_loss,

            # optional for batch logging purposes
            "log": logs,
        }

        return batch_dictionary

    def validation_epoch_end(self, outputs):

        # calculating average loss
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        # logging using tensorboard logger
        self.logger.experiment.add_scalar("Loss/Validation",
                                          avg_loss,
                                          self.current_epoch)
