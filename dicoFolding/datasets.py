# -*- coding: utf-8 -*-
# /usr/bin/env python3
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
Tools in order to create pytorch dataloaders
"""

import pandas as pd
import torch
import torchvision.transforms as transforms
from scipy.ndimage import rotate
import numpy as np

from deep_folding.preprocessing.pynet_transforms import PaddingTensor

class TensorDataset():
    """Custom dataset that includes image file paths.
    Applies different transformations to data depending on the type of input.
    IN: data_tensor: tensor containing MRIs as numpy arrays
        filenames: list of subjects' IDs
    OUT: tensor of [batch, sample, subject ID]
    """
    def __init__(self, data_tensor, filenames):
        self.data_tensor = data_tensor
        self.transform = True
        self.nb_train = len(filenames)
        print(self.nb_train)
        self.filenames = filenames

    def __len__(self):
        return(self.nb_train)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.data_tensor[idx]
        file = self.filenames[idx]

        fill_value = 11
        self.transform = transforms.Compose([
                    PaddingTensor([1, 80, 80, 80], fill_value=fill_value)
        ])

        sample = self.transform(sample)

        tuple_with_path = (sample, file)
        return tuple_with_path

class SkeletonDataset():
    """Custom dataset for skeleton images that includes image file paths.
    dataframe: dataframe containing training and testing arrays
    filenames: optional, list of corresponding filenames
    Works on CPUs
    """
    def __init__(self, dataframe):
        self.df = dataframe

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.df.iloc[idx][0]
        print("keys", self.df.iloc[idx].keys())

        fill_value = 11  # value for the part outside the brain
        self.transform = transforms.Compose([Padding([1, 80, 80, 80], fill_value=fill_value)
                         ])
        sample = self.transform(sample)
        tuple_with_path = (sample, 'ID'+str(idx))
        return tuple_with_path

def create_sets(config):
    tmp = pd.read_pickle('/home/jc225751/Runs/05_2021-05-03_premiers_essais_simclr/Input/crops/Lskeleton.pkl')

    len_tmp = len(tmp.columns)
    filenames = list(tmp.columns)
    print("length of dataframe", len_tmp)
    print("column names : ", filenames)

    """
    Creates a tensor object from the DataFrame (through a conversion into a numpy array)
    """
    tmp = torch.from_numpy(np.array([tmp.loc[0].values[k] for k in range(len_tmp)]))

    """
    Creates the dataset from this tensor by doing some preprocessing:
	- padding to 80x80x80
    """
    hcp_dataset = TensorDataset(filenames=filenames, data_tensor=tmp)

    print(len(hcp_dataset))

    # Split training set into train and val
    partition = [0.8, 0.2]

    random_seed = 42
    torch.manual_seed(random_seed)

    print([round(i*(len(hcp_dataset))) for i in partition])
    train_set, val_set = torch.utils.data.random_split(hcp_dataset,
                         [round(i*(len(hcp_dataset))) for i in partition])

    return train_set, val_set