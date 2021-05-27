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
import time
from datetime import date
import argparse
import pandas as pd
import scipy.stats as stat
import itertools
import matplotlib.pyplot as plt
import random

import numpy as np
from torch.utils import data

import tqdm
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
from torchvision import models
# from torchsummary import summary
from torch.utils.data import DataLoader, Dataset, RandomSampler

from dicoFolding.models.densenet import densenet121
from dicoFolding.losses import NTXenLoss
from dicoFolding.contrastiveLearning import ContrastiveLearningModel
from dicoFolding.datasets import MRIDataset
from dicoFolding.datasets import create_sets

from dataclasses import dataclass
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_name='config', config_path="experiments")
def train(config):

    print(OmegaConf.to_yaml(config))
    print("Working directory : {}".format(os.getcwd()))
    config.input_size = eval(config.input_size)

    net = densenet121(mode="encoder", drop_rate=0.0)
    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Number of parameters to estimate: ", params)
    # summary(net, (1, 80, 80, 80), device="cpu")

    loss = NTXenLoss(temperature=config.temperature,
                     return_logits=True)

    dataset_train, dataset_val = create_sets(config)

    loader_train = DataLoader(dataset_train,
                              batch_size=config.batch_size,
                              sampler=RandomSampler(dataset_train),
                              pin_memory=config.pin_mem,
                              num_workers=config.num_cpu_workers
                              )
    loader_val = DataLoader(dataset_val,
                            batch_size=config.batch_size,
                            sampler=RandomSampler(dataset_val),
                            pin_memory=config.pin_mem,
                            num_workers=config.num_cpu_workers
                            )

    model = ContrastiveLearningModel(net, loss,
                                     loader_train, loader_val,
                                     config)

    model.training()


if __name__ == "__main__":
    train()
