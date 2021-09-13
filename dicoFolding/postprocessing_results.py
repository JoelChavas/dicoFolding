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
from sklearn.cluster import KMeans

from dicoFolding.postprocessing.visualize_tsne import plot_tsne
from dicoFolding.postprocessing.visualize_nearest_neighhbours import plot_knn_examples
from dicoFolding.postprocessing.visualize_nearest_neighhbours import plot_knn_meshes

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
def postprocessing_results(config):
    config = process_config(config)

    # Sets seed for pseudo-random number generators
    # in: pytorch, numpy, python.random
    # seed_everything(config.seed)

    data_module = DataModule(config)
    data_module.setup(stage='validate')

    # Show the views of the first batch
    # fig = plt.figure(figsize=(4., 8.), dpi=400)
    # grid = ImageGrid(fig, 111,
    #                 nrows_ncols = (config.batch_size//4, 8),
    #                 axes_pad=0.2,)
    # (inputs, filenames) = next(iter(data_module.val_dataloader()))
    # input_i = inputs[:, 0, :]
    # input_j = inputs[:, 1, :]
    # print("input_i : {}".format(np.unique(input_i)))
    # print("input_j : {}".format(np.unique(input_j)))
    # images = []
    # np.save("input_i.npy", input_i[0, 0, :, :, :])
    # np.save("input_j.npy", input_j[0, 0, :, :, :])
    # vol_i = aims.Volume(1, 80, 80, 80, dtype=np.int32)
    # np.asarray(vol_i)[:] = input_i[0, :, :, :, :]
    # aims.write(vol_i, 'input_i.nii')
    # print(np.unique(input_i[0, :, :, :, :]))
    # for i in range(config.batch_size):
    #     images.append(input_i[i, 0, input_i.shape[2]//2, :, :])
    #     images.append(input_j[i, 0, input_i.shape[2]//2, :, :])
    # for ax, im in zip(grid, images):
    #     ax.imshow(im)
    #     ax.axis('off')
    #     ax.set_title(np.unique(im)[1], fontsize=4)
    # plt.show()
    
    # Show the views of the first skeleton after each epoch
    model = ContrastiveLearner(config,
                            mode="encoder",
                            sample_data=data_module)
    model = model.load_from_checkpoint(config.checkpoint_path,
                                       config=config,
                                       mode="encoder",
                                       sample_data=data_module)
    summary(model, tuple(config.input_size), device="cpu")
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=config.max_epochs,
        logger=tb_logger,
        flush_logs_every_n_steps=config.nb_steps_per_flush_logs,
        resume_from_checkpoint=config.checkpoint_path)
    dataset_val = data_module.dataset_val
    trainer.validate(model, data_module.val_dataloader()) 
    embeddings, filenames = model.compute_representations(data_module.val_dataloader())
    
    data_module_visu = DataModule(config)
    data_module_visu.setup(stage='validate', mode='visualization')
    
    plot_knn_meshes(embeddings=embeddings,
                      filenames=filenames,
                      dataset=data_module_visu.dataset_val,
                      n_neighbors=6,
                      num_examples=3
                      )
    
    plot_knn_examples(embeddings=embeddings,
                      filenames=filenames,
                      dataset=data_module_visu.dataset_val,
                      n_neighbors=6,
                      num_examples=3
                      )
    
    # Makes Kmeans and represents it on a t-SNE plot
    X_tsne = model.compute_tsne(data_module.val_dataloader(), "representation")
    n_clusters = 5
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embeddings)
    plot_tsne(X_tsne=X_tsne, buffer=False, labels=kmeans.labels_)


if __name__ == "__main__":
    postprocessing_results()
