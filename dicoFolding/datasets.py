# -*- coding: utf-8 -*-
# /usr/bin/env python3
#
# This software and supporting documentation are distributed by
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
Tools in order to create pytorch dataloaders
"""

import pandas as pd
import torch
import torchvision.transforms as transforms
from scipy.ndimage import rotate
import numpy as np

from os.path import join

from torch.utils.data import Dataset
from dicoFolding.augmentations import Transformer, Crop, CutoutTensor, Cutout
from dicoFolding.augmentations import Noise, Normalize, Blur, Flip

from deep_folding.preprocessing.pynet_transforms import PaddingTensor, Padding


class TensorDataset():
    """Custom dataset that includes image file paths.

    Applies different transformations to data depending on the type of input.
    Args:
        data_tensor: tensor containing MRIs as numpy arrays
        filenames: list of subjects' IDs
    Returns:
        tensor of [batch, sample, subject ID]
    """
    def __init__(self, data_tensor, filenames, config):
        self.data_tensor = data_tensor.type(torch.float32)
        self.transform = True
        self.nb_train = len(filenames)
        print(self.nb_train)
        self.filenames = filenames
        self.config = config

    def __len__(self):
        return (self.nb_train)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.data_tensor[idx]
        filename = self.filenames[idx]

        self.transform1 = transforms.Compose([
                    PaddingTensor(self.config.input_size,
                                  fill_value=self.config.fill_value)
        ])

        patch_size = np.ceil(np.array(self.config.input_size)/4)
        self.transform2 = transforms.Compose([
            PaddingTensor(self.config.input_size,
                          fill_value=self.config.fill_value),
            CutoutTensor(patch_size=patch_size)
        ])

        view1 = self.transform1(sample)
        view2 = self.transform2(sample)

        views = torch.stack((view1, view2, view2), dim=0)

        tuple_with_path = (views, filename)
        return tuple_with_path


def create_sets(config):

    pickle_file_path = config.pickle_file
    tmp = pd.read_pickle(pickle_file_path)

    len_tmp = len(tmp.columns)
    filenames = list(tmp.columns)  # files names = subject IDs
    print("length of dataframe", len_tmp)
    print("column names : ", filenames)

    # Creates a tensor object from the DataFrame
    # (through a conversion into a numpy array)
    tmp = torch.from_numpy(np.array([tmp.loc[0].values[k]
                                     for k in range(len_tmp)]))
    print(tmp.shape)

    # Creates the dataset from this tensor by doing some preprocessing:
    # - padding to 80x80x80 by default, see config file
    hcp_dataset = TensorDataset(filenames=filenames,
                                data_tensor=tmp,
                                config=config)
    print(len(hcp_dataset))

    # Split training set into train and val
    partition = [0.8, 0.2]

    random_seed = config.seed
    torch.manual_seed(random_seed)

    print([round(i*(len(hcp_dataset))) for i in partition])
    train_set, val_set = torch.utils.data.random_split(
        hcp_dataset,
        [round(i*(len(hcp_dataset))) for i in partition])

    return train_set, val_set
