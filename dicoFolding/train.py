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
""" Training SimCLR on skeleton images

"""
######################################################################
# Imports and global variables definitions
######################################################################
import logging

import hydra
import torch
import pytorch_lightning as pl
from dicoFolding.contrastive_learner import ContrastiveLearner
from dicoFolding.contrastive_learner_test import ContrastiveLearnerTest
from dicoFolding.datamodule import DataModule
from dicoFolding.utils import process_config
from dicoFolding.postprocessing.visualize_tsne import plot_output
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.utilities.seed import seed_everything
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid

from soma import aims

tb_logger = pl_loggers.TensorBoardLogger('logs')
writer = SummaryWriter()
log = logging.getLogger(__name__)

"""
We call:
- embedding, the space before the projection head.
  The elements of the space are features
- output, the space after the projection head.
  The elements are called output vectors
"""

      
@hydra.main(config_name='config', config_path="config")
def train(config):
    config = process_config(config)

    # Sets seed for pseudo-random number generators
    # in: pytorch, numpy, python.random
    # seed_everything(config.seed)

    data_module = DataModule(config)

    if config.test == True:
          # Show the views of the first batch
          data_module.setup(stage='fit')
          fig = plt.figure(figsize=(4., 8.), dpi=400)
          grid = ImageGrid(fig, 111,
                           nrows_ncols = (config.batch_size//4, 8),
                           axes_pad=0.2,)
          (inputs, filenames) = next(iter(data_module.train_dataloader()))
          input_i = inputs[:, 0, :]
          input_j = inputs[:, 1, :]
          print("input_i : {}".format(np.unique(input_i)))
          print("input_j : {}".format(np.unique(input_j)))
          images = []
          np.save("input_i.npy", input_i[0, 0, :, :, :])
          np.save("input_j.npy", input_j[0, 0, :, :, :])
          vol_i = aims.Volume(1, 80, 80, 80, dtype=np.int32)
          np.asarray(vol_i)[:] = input_i[0, :, :, :, :]
          aims.write(vol_i, 'input_i.nii')
          print(np.unique(input_i[0, :, :, :, :]))
          for i in range(config.batch_size):
                images.append(input_i[i, 0, input_i.shape[2]//2, :, :])
                images.append(input_j[i, 0, input_i.shape[2]//2, :, :])
          for ax, im in zip(grid, images):
                ax.imshow(im)
                ax.axis('off')
                ax.set_title(np.unique(im)[1], fontsize=4)
          plt.show()
          
          # Show the views of the first skeleton after each epoch
          model = ContrastiveLearnerTest(config,
                                    mode="encoder",
                                    drop_rate=0.0,
                                    sample_data=data_module)
          summary(model, tuple(config.input_size), device="cpu")
          trainer = pl.Trainer(
              gpus=1,
              max_epochs=config.max_epochs,
              logger=tb_logger,
              flush_logs_every_n_steps=config.nb_steps_per_flush_logs,
              resume_from_checkpoint=config.checkpoint_path)
          trainer.fit(model, data_module)
          
          nb_elements = len(model.sample_i)
          # Plots the views of the first element
          fig = plt.figure(figsize=(4., 8.), dpi=400)
          grid = ImageGrid(fig, 111,
                           nrows_ncols = (nb_elements//4, 8),
                           axes_pad=0.2,)
          images = []
          for i in range(nb_elements):
                first_view = model.sample_i[i]
                second_view = model.sample_j[i]
                images.append(first_view[0, 0, first_view.shape[2]//2, :, :])
                images.append(second_view[0, 0, second_view.shape[2]//2, :, :])
          for ax, im in zip(grid, images):
                ax.imshow(im)
                ax.axis('off')
          plt.show()
          
          # Plots one representation image
          fig = plt.figure(figsize=(4., 8.), dpi=400)
          plot_output(first(self.save_output.outputs.values()), buffer=False)
          plt.show()   
          
    else:
          model = ContrastiveLearner(config,
                                    mode="encoder",
                                    sample_data=data_module)
          summary(model, tuple(config.input_size), device="cpu")
          trainer = pl.Trainer(
              gpus=1,
              max_epochs=config.max_epochs,
              logger=tb_logger,
              flush_logs_every_n_steps=config.nb_steps_per_flush_logs,
              resume_from_checkpoint=config.checkpoint_path)
          trainer.fit(model, data_module)
          
    print("Number of hooks: ", len(model.save_output.outputs))


if __name__ == "__main__":
    train()
