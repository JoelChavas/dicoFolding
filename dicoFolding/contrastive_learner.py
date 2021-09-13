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

import torch
import numpy as np
from dicoFolding.losses import NTXenLoss
from dicoFolding.models.densenet import DenseNet
from sklearn.manifold import TSNE

from dicoFolding.postprocessing.visualize_tsne import plot_tsne
from dicoFolding.postprocessing.visualize_tsne import plot_img
from dicoFolding.postprocessing.visualize_tsne import plot_output

from toolz.itertoolz import last, first

class SaveOutput:
    def __init__(self):
        self.outputs = {}
        
    def __call__(self, module, module_in, module_out):
        self.outputs[module] = module_out.cpu()
        
    def clear(self):
        self.outputs = {}


class ContrastiveLearner(DenseNet):

    def __init__(self, config, mode, sample_data):
        super(ContrastiveLearner, self).__init__(growth_rate=config.growth_rate,
                                                 block_config=config.block_config,
                                                 num_init_features=config.num_init_features,
                                                 num_representation_features=config.num_representation_features,
                                                 num_outputs=config.num_outputs,
                                                 mode=mode,
                                                 drop_rate=config.drop_rate)
        self.config = config
        self.sample_data = sample_data
        self.sample_i = np.array([])
        self.sample_j = np.array([])
        self.save_output = SaveOutput()
        self.hook_handles = []
        self.get_layers()

    def get_layers(self):
        for layer in self.modules():
            if type(layer) == torch.nn.Linear:
                handle = layer.register_forward_hook(self.save_output)
                self.hook_handles.append(handle)


         
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
        
        if batch_idx == 0:
            self.sample_i = inputs[:, 0, :].cpu()
            self.sample_j = inputs[:, 1, :].cpu()

        # logs - a dictionary
        logs = {"train_loss": float(batch_loss)}

        batch_dictionary = {
            # REQUIRED: It is required for us to return "loss"
            "loss": batch_loss,

            # optional for batch logging purposes
            "log": logs,

        }

        return batch_dictionary

    def compute_embeddings_skeletons(self, loader):
        X = torch.zeros([0, self.config.num_outputs]).cpu()
        filenames_list = []
        with torch.no_grad():
            for (inputs, filenames) in loader:
                # First views of the whole batch
                inputs = inputs.cuda()
                model = self.cuda()
                X_i = model.forward(inputs[:, 0, :])
                # Second views of the whole batch
                X_j = model.forward(inputs[:, 1, :])
                # First views and deep_folding
                # second views are put side by side
                X_reordered = torch.cat([X_i, X_j], dim=-1)
                X_reordered = X_reordered.view(-1, X_i.shape[-1])
                X = torch.cat((X, X_reordered.cpu()), dim=0)
                filenames_duplicate = [ item for item in filenames for repetitions in range(2) ]
                filenames_list = filenames_list + filenames_duplicate
                del inputs
        return X, filenames_list
    
    def compute_representations(self, loader):
        X = torch.zeros([0, self.config.num_representation_features]).cpu()
        filenames_list = []
        with torch.no_grad():
            for (inputs, filenames) in loader:
                # First views of the whole batch
                inputs = inputs.cuda()
                model = self.cuda()
                model.forward(inputs[:, 0, :])
                X_i = first(self.save_output.outputs.values())
                # Second views of the whole batch
                model.forward(inputs[:, 1, :])
                X_j = first(self.save_output.outputs.values())
                # First views and second views are put side by side
                X_reordered = torch.cat([X_i, X_j], dim=-1)
                X_reordered = X_reordered.view(-1, X_i.shape[-1])
                X = torch.cat((X, X_reordered.cpu()), dim=0)
                filenames_duplicate = [ item for item in filenames for repetitions in range(2) ]
                filenames_list = filenames_list + filenames_duplicate
                del inputs
        return X, filenames_list

    def compute_tsne(self, loader, register):
        if register == "output":
            X, _ = self.compute_embeddings_skeletons(loader)
        elif register == "representation":
            X, _ = self.compute_representations(loader)
        else:
            raise ValueError("Argument register must be either output or representation")
    
        tsne = TSNE(n_components=2, perplexity=5, init='pca', random_state=50)
        X_tsne = tsne.fit_transform(X.detach().numpy())

        return X_tsne



    def training_epoch_end(self, outputs):
        if self.current_epoch % 10 == 0:
            X_tsne = self.compute_tsne(self.sample_data.train_dataloader(), "output")
            image_TSNE = plot_tsne(X_tsne, buffer=True)
            self.logger.experiment.add_image(
                'TSNE output image', image_TSNE, self.current_epoch)
            X_tsne = self.compute_tsne(self.sample_data.train_dataloader(), "representation")
            image_TSNE = plot_tsne(X_tsne, buffer=True)
            self.logger.experiment.add_image(
                'TSNE representation image', image_TSNE, self.current_epoch)
        
        image_input_i = plot_img(self.sample_i, buffer=True)
        self.logger.experiment.add_image(
            'input_i', image_input_i, self.current_epoch)
        image_input_j = plot_img(self.sample_j, buffer=True)
        self.logger.experiment.add_image(
            'input_j', image_input_j, self.current_epoch)

        # Plots one representation and one output image
        image_output = plot_output(first(self.save_output.outputs.values()), buffer=True)
        self.logger.experiment.add_image(
            'representation', image_output, self.current_epoch)
        
        image_output = plot_output(last(self.save_output.outputs.values()), buffer=True)
        self.logger.experiment.add_image(
            'output', image_output, self.current_epoch)
        print("Length of outputs: ", len(self.save_output.outputs))
        
        # calculating average loss
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        # logging histograms
        # self.custom_histogram_adder()

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
        # if self.current_epoch % 10 == 0 or self.current_epoch == self.config.max_epochs:
        #     X_tsne = self.compute_tsne(self.sample_data.val_dataloader(), "output")
        #     image_TSNE = plot_tsne(X_tsne, buffer=True)
        #     self.logger.experiment.add_image(
        #         'TSNE output validation image', image_TSNE, self.current_epoch)
        #     X_tsne = self.compute_tsne(self.sample_data.val_dataloader(), "representation")
        #     image_TSNE = plot_tsne(X_tsne, buffer=True)
        #     self.logger.experiment.add_image(
        #         'TSNE representation validation image', image_TSNE, self.current_epoch)
        
        # Plots one representation image and one output image
        image_output = plot_output(first(self.save_output.outputs.values()), buffer=True)
        self.logger.experiment.add_image(
            'representation val', image_output, self.current_epoch)
        
        image_output = plot_output(last(self.save_output.outputs.values()), buffer=True)
        self.logger.experiment.add_image(
            'output val', image_output, self.current_epoch)
        
        # calculating average loss
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        # logging using tensorboard logger
        self.logger.experiment.add_scalar("Loss/Validation",
                                          avg_loss,
                                          self.current_epoch)
