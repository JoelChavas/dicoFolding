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
import os

import hydra
import pytorch_lightning as pl
from lightly.loss import NTXentLoss
from lightly.models import SimCLR
from pytorch_lightning.utilities.seed import seed_everything

from dicoFolding.contrastive_learner import ContrastiveLearner
from dicoFolding.datamodule import DataModule
from dicoFolding.utils import process_config

log = logging.getLogger(__name__)


@hydra.main(config_name='config', config_path="config")
def train(config):
    config = process_config(config)
    
    # Sets seed for pseudo-random number generators in: pytorch, numpy, python.random
    seed_everything(config.seed)

    data_module = DataModule(config)
    model = ContrastiveLearner(config,
                               mode="encoder",
                               drop_rate=0.0)
    trainer = pl.Trainer(gpus=1, max_epochs=config.max_epochs)
    trainer.fit(model, data_module)


if __name__ == "__main__":
    train()
