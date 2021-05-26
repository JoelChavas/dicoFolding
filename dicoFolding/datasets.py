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
from dicoFolding.augmentations import Transformer, Crop, Cutout, Noise, Normalize, Blur, Flip

from deep_folding.preprocessing.pynet_transforms import PaddingTensor, Padding

class MRIDataset(Dataset):

    def __init__(self, config, training=False, validation=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert training != validation

        self.transforms = Transformer()
        self.config = config
        self.transforms.register(Normalize(), probability=1.0)

        if config.tf == "all_tf":
            self.transforms.register(Flip(), probability=0.5)
            self.transforms.register(Blur(sigma=(0.1, 1)), probability=0.5)
            self.transforms.register(Noise(sigma=(0.1, 1)), probability=0.5)
            self.transforms.register(Cutout(patch_size=np.ceil(np.array(config.input_size)/4)), probability=0.5)
            self.transforms.register(Crop(np.ceil(0.75*np.array(config.input_size)), "random", resize=True),
                                     probability=0.5)

        elif config.tf == "cutout":
            self.transforms.register(Cutout(patch_size=np.ceil(np.array(config.input_size)/4)), probability=1)

        elif config.tf == "crop":
            self.transforms.register(Crop(np.ceil(0.75*np.array(config.input_size)), "random", resize=True),
                                     probability=1)
                                     
        pickle_file_path = config.pickle_file

        if training:
            #self.data = np.load(config.data_train)
            #self.labels = pd.read_csv(config.label_train)
            tmp = pd.read_pickle(pickle_file_path)
            len_tmp = len(tmp.columns)
            data = np.array([tmp.loc[0].values[k] for k in range(len_tmp-2)])
            s = data.shape
            # Does the padding here by pushing data onthe "left"
            self.data = np.zeros((len_tmp-2, 1, 80, 80, 80))
            self.data[:s[0], :s[1], :s[2], :s[3], :s[4]] = data
        elif validation:
            #self.data = np.load(config.data_val)
            #self.labels = pd.read_csv(config.label_val)
            tmp = pd.read_pickle(pickle_file_path)
            len_tmp = len(tmp.columns)
            data = np.array([tmp.loc[0].values[k] for k in range(len_tmp-2,len_tmp)])
            s = data.shape
            self.data = np.zeros((2, 1, 80, 80, 80))
            self.data[:s[0], :s[1], :s[2], :s[3], :s[4]] = data

        assert self.data.shape[1:] == tuple(config.input_size), "3D images must have shape {}".\
            format(config.input_size)

    def collate_fn(self, list_samples):
        list_x = torch.stack([torch.as_tensor(x, dtype=torch.float) for (x, y) in list_samples], dim=0)
        list_y = torch.stack([torch.as_tensor(y, dtype=torch.float) for (x, y) in list_samples], dim=0)
 
        return (list_x, list_y)

    def __getitem__(self, idx):

        # For a single input x, samples (t, t') ~ T to generate (t(x), t'(x))
        np.random.seed()
        x1 = self.transforms(self.data[idx])
        x2 = self.transforms(self.data[idx])
        x = np.stack((x1, x2), axis=0)

        labels = 1
        return (x,labels)

    def __len__(self):
        return len(self.data)

class TensorDataset():
    """Custom dataset that includes image file paths.
    
    Applies different transformations to data depending on the type of input.
    Args: 
        data_tensor: tensor containing MRIs as numpy arrays
        filenames: list of subjects' IDs
    Returns: 
        tensor of [batch, sample, subject ID]
    """
    def __init__(self, data_tensor, filenames, fill_value):
        self.data_tensor = data_tensor.type(torch.float32)
        self.transform = True
        self.nb_train = len(filenames)
        print(self.nb_train)
        self.filenames = filenames
        self.fill_value = fill_value

    def __len__(self):
        return (self.nb_train)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.data_tensor[idx]
        filename = self.filenames[idx]

        self.transform = transforms.Compose([
                    PaddingTensor([1, 80, 80, 80], 
                                  fill_value=self.fill_value)
        ])

        view1 = self.transform(sample)
        view2 = self.transform(sample)
        
        views = torch.stack((view1, view2), dim=0)

        tuple_with_path = (views, filename)
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
    
    pickle_file_path = config.pickle_file
    tmp = pd.read_pickle(pickle_file_path)

    len_tmp = len(tmp.columns)
    filenames = list(tmp.columns)  # files names = subject IDs
    print("length of dataframe", len_tmp)
    print("column names : ", filenames)

    """
    Creates a tensor object from the DataFrame (through a conversion into a numpy array)
    """
    tmp = torch.from_numpy(np.array([tmp.loc[0].values[k] for k in range(len_tmp)]))
    print(tmp.shape)
    """
    Creates the dataset from this tensor by doing some preprocessing:
	- padding to 80x80x80
    """
    hcp_dataset = TensorDataset(filenames=filenames, 
                                data_tensor=tmp, 
                                fill_value=config.fill_value)

    print(len(hcp_dataset))

    # Split training set into train and val
    partition = [0.8, 0.2]

    random_seed = config.seed
    torch.manual_seed(random_seed)

    print([round(i*(len(hcp_dataset))) for i in partition])
    train_set, val_set = torch.utils.data.random_split(hcp_dataset,
                         [round(i*(len(hcp_dataset))) for i in partition])

    return train_set, val_set
