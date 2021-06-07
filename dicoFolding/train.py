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

import os

from dicoFolding.contrastive_learner import ContrastiveLearner
from dicoFolding.datamodule import DataModule

import hydra
from omegaconf import OmegaConf

import pytorch_lightning as pl

import logging
log = logging.getLogger(__name__)


def process_config(config):
    """Does whatever operations on the config file
    """

    log.info(OmegaConf.to_yaml(config))
    log.info("Working directory : {}".format(os.getcwd()))
    config.input_size = eval(config.input_size)
    log.info("config type: {}".format(type(config)))
    return config


@hydra.main(config_name='config', config_path="experiments")
def train(config):
    config = process_config(config)

    data_module = DataModule(config)
    model = ContrastiveLearner(config,
                               mode="encoder",
                               drop_rate=0.0)
    trainer = pl.Trainer(gpus=1, max_epochs=config.max_epochs)
    trainer.fit(model, data_module)


if __name__ == "__main__":
    train()
