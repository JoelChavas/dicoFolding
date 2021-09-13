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
Tools to create pytorch dataloaders
"""
import logging

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from deep_folding.preprocessing.pynet_transforms import PaddingTensor
from dicoFolding.augmentations import OnlyBottomTensor
from dicoFolding.augmentations import RotateTensor
from dicoFolding.augmentations import SimplifyTensor
from dicoFolding.augmentations import MixTensor

_ALL_SUBJECTS = -1

log = logging.getLogger(__name__)


class ContrastiveDataset():
    """Custom dataset that includes image file paths.

    Applies different transformations to data depending on the type of input.
    """

    def __init__(self, data_tensor, filenames, config):
        """
        Args:
            data_tensor (tensor): contains MRIs as numpy arrays
            filenames (list of strings): list of subjects' IDs
            config (Omegaconf dict): contains configuration information
        """
        self.data_tensor = data_tensor.type(torch.float32)
        self.transform = True
        self.nb_train = len(filenames)
        log.info(self.nb_train)
        self.filenames = filenames
        self.config = config

    def __len__(self):
        return (self.nb_train)

    def __getitem__(self, idx):
        """Returns the two views corresponding to index idx

        The two views are generated on the fly.
        The first view is the reference view (only padding is applied)

        Returns:
            tuple of (views, subject ID)
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.data_tensor[idx]
        filename = self.filenames[idx]

        self.transform1 = transforms.Compose([
            SimplifyTensor(),
            PaddingTensor(self.config.input_size,
                          fill_value=self.config.fill_value),
            MixTensor(from_skeleton=True, patch_size=self.config.patch_size),
            RotateTensor(max_angle=self.config.max_angle)
        ])
        
        # - padding to 80x80x80 by default, see config file
        # - + random rotation
        self.transform2 = transforms.Compose([
            SimplifyTensor(),
            PaddingTensor(self.config.input_size,
                          fill_value=self.config.fill_value),
            MixTensor(from_skeleton=False, patch_size=self.config.patch_size),
            RotateTensor(max_angle=self.config.max_angle)
        ])

        view1 = self.transform1(sample)
        view2 = self.transform2(sample)

        views = torch.stack((view1, view2), dim=0)

        tuple_with_path = (views, filename)
        return tuple_with_path


class ContrastiveDataset_Visualization():
    """Custom dataset that includes image file paths.

    Applies different transformations to data depending on the type of input.
    """

    def __init__(self, data_tensor, filenames, config):
        """
        Args:
            data_tensor (tensor): contains MRIs as numpy arrays
            filenames (list of strings): list of subjects' IDs
            config (Omegaconf dict): contains configuration information
        """
        self.data_tensor = data_tensor.type(torch.float32)
        self.transform = True
        self.nb_train = len(filenames)
        log.info(self.nb_train)
        self.filenames = filenames
        self.config = config

    def __len__(self):
        return (self.nb_train)

    def __getitem__(self, idx):
        """Returns the two views corresponding to index idx

        The two views are generated on the fly.
        The first view is the reference view (only padding is applied)

        Returns:
            tuple of (views, subject ID)
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.data_tensor[idx]
        filename = self.filenames[idx]

        self.transform1 = transforms.Compose([
            PaddingTensor(self.config.input_size,
                          fill_value=self.config.fill_value),
            MixTensor(from_skeleton=True, patch_size=self.config.patch_size),
            RotateTensor(max_angle=self.config.max_angle)
        ])
        
        self.transform2 = transforms.Compose([
            PaddingTensor(self.config.input_size,
                          fill_value=self.config.fill_value)
        ])

        view1 = self.transform1(sample)
        view2 = self.transform2(sample)

        views = torch.stack((view1, view2), dim=0)

        tuple_with_path = (views, filename)
        return tuple_with_path
    
    
def create_sets(config, mode='training'):
    """Creates train, validation and test sets

    Args:
        config (Omegaconf dict): contains configuration parameters
        mode (str): either 'training' or 'visualization'
    Returns:
        train_set, val_set, test_set (tuple)
    """

    pickle_file_path = config.pickle_file
    all_data = pd.read_pickle(pickle_file_path)

    len_data = (
        len(all_data.columns)
        if config.nb_subjects == _ALL_SUBJECTS
        else min(config.nb_subjects, len(all_data.columns)))

    filenames = (
        list(all_data.columns)
        if config.nb_subjects == _ALL_SUBJECTS
        else list(all_data.columns)[:len_data])  # files names = subject IDs

    log.info("length of dataframe: {}".format(len_data))
    log.info("column names : {}".format(filenames))

    # Creates a tensor object from the DataFrame
    # (through a conversion into a numpy array)
    tensor_data = torch.from_numpy(np.array([all_data.loc[0].values[k]
                                             for k in range(len_data)]))
    log.info("Tensor data shape: {}".format(tensor_data.shape))

    # Creates the dataset from this tensor by doing some preprocessing
    if mode == 'visualization':
        hcp_dataset = ContrastiveDataset_Visualization(filenames=filenames,
                                     data_tensor=tensor_data,
                                     config=config)
    else:
        hcp_dataset = ContrastiveDataset(filenames=filenames,
                                     data_tensor=tensor_data,
                                     config=config)
    log.info("Length of complete data set: {}".format(len(hcp_dataset)))

    # Split training set into train, val and test set
    partition = config.partition

    log.info([round(i * (len(hcp_dataset))) for i in partition])
    np.random.seed(1)
    train_set, val_set, test_set = torch.utils.data.random_split(
        hcp_dataset,
        [round(i * (len(hcp_dataset))) for i in partition])

    return train_set, val_set, test_set
