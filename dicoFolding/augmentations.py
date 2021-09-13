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

import numbers
from collections import namedtuple

import numpy as np
import torch
from scipy.ndimage import rotate
import skimage
from sklearn.preprocessing import OneHotEncoder

class SimplifyTensor(object):
    """Puts all internal and external values to background value 0
    """

    def __init__(self):
        None
        
    def __call__(self, tensor):
        arr = tensor.numpy()
        arr[arr == 11] = 0    
        return torch.from_numpy(arr)  
    
class OnlyBottomTensor(object):
    """Keeps only bottom '30' values, puts everything else to '0'
    """

    def __init__(self):
        None
        
    def __call__(self, tensor):
        arr = tensor.numpy()
        arr = arr * (arr==30)   
        return torch.from_numpy(arr)  
    

class RotateTensor(object):
    """Apply a random rotation on the images
    """

    def __init__(self, max_angle):
        self.max_angle = max_angle

    def __call__(self, tensor):

        arr = tensor.numpy()[0]
        flat_im = np.reshape(arr, (-1, 1))
        im_encoder = OneHotEncoder(sparse=False, categories='auto')
        onehot_im = im_encoder.fit_transform(flat_im)
        ## rotate one hot im
        onehot_im = onehot_im.reshape(80, 80, 80, -1)
        onehot_im_rot = np.empty_like(onehot_im)
        n_cat = onehot_im.shape[-1]
        for axes in (0,1), (0,2), (1,2):
            np.random.seed()
            angle = np.random.uniform(-self.max_angle, self.max_angle)
            onehot_im_rot = rotate(onehot_im,
                                angle=angle,
                                axes=axes,
                                reshape=False,
                                mode='constant',
                                cval=0)
        im_rot_flat = im_encoder.inverse_transform(np.reshape(onehot_im_rot, (-1, n_cat)))
        im_rot = np.reshape(im_rot_flat, (80, 80, 80))
        arr_rot = np.expand_dims(
            im_rot,
            axis=0)
        return torch.from_numpy(arr_rot)


class MixTensor(object):
    """Apply a cutout on the images and puts only bottom value inside the cutout
    cf. Improved Regularization of Convolutional Neural Networks with Cutout,
    arXiv, 2017
    We assume that the cube to be cut is inside the image.
    """

    def __init__(self, from_skeleton=True, patch_size=None, random_size=False,
                 inplace=False, localization=None):
        """[summary]

        If from_skeleton==True, takes skeleton image, cuts it out and fills with bottom_only image
        If from_skeleton==False, takes bottom_only image, cuts it out and fills with skeleton image

        Args:
            from_skeleton (bool, optional): [description]. Defaults to True.
            patch_size (either int or list of int): [description]. Defaults to None.
            random_size (bool, optional): [description]. Defaults to False.
            inplace (bool, optional): [description]. Defaults to False.
            localization ([type], optional): [description]. Defaults to None.
        """
        self.patch_size = patch_size
        self.random_size = random_size
        self.inplace = inplace
        self.localization = localization
        self.from_skeleton = from_skeleton

    def __call__(self, tensor):

        arr = tensor.numpy()
        img_shape = np.array(arr.shape)
        if isinstance(self.patch_size, int):
            size = [self.patch_size for _ in range(len(img_shape))]
        else:
            size = np.copy(self.patch_size)
        assert len(size) == len(img_shape), "Incorrect patch dimension."
        indexes = []
        for ndim in range(len(img_shape)):
            if size[ndim] > img_shape[ndim] or size[ndim] < 0:
                size[ndim] = img_shape[ndim]
            if self.random_size:
                size[ndim] = np.random.randint(0, size[ndim])
            if self.localization is not None:
                delta_before = max(
                    self.localization[ndim] - size[ndim] // 2, 0)
            else:
                np.random.seed()
                delta_before = np.random.randint(
                    0, img_shape[ndim] - size[ndim] + 1)
            indexes.append(slice(int(delta_before),
                                 int(delta_before + size[ndim])))
        if self.from_skeleton:
            if self.inplace:
                arr_cut = arr[tuple(indexes)]
                arr[tuple(indexes)] = arr_cut * (arr_cut==30)
                return torch.from_numpy(arr)
            else:
                arr_copy = np.copy(arr)
                arr_cut = arr_copy[tuple(indexes)]
                arr_copy[tuple(indexes)] = arr_cut * (arr_cut==30)
                return torch.from_numpy(arr_copy)
        else:
            arr_bottom = arr * (arr==30)
            arr_cut = arr[tuple(indexes)]
            arr_bottom[tuple(indexes)] = np.copy(arr_cut)
            return torch.from_numpy(arr_bottom)

class CutoutTensor(object):
    """Apply a cutout on the images
    cf. Improved Regularization of Convolutional Neural Networks with Cutout,
    arXiv, 2017
    We assume that the cube to be cut is inside the image.
    """

    def __init__(self, patch_size=None, value=0, random_size=False,
                 inplace=False, localization=None):
        self.patch_size = patch_size
        self.value = value
        self.random_size = random_size
        self.inplace = inplace
        self.localization = localization

    def __call__(self, arr):

        img_shape = np.array(arr.shape)
        if isinstance(self.patch_size, int):
            size = [self.patch_size for _ in range(len(img_shape))]
        else:
            size = np.copy(self.patch_size)
        assert len(size) == len(img_shape), "Incorrect patch dimension."
        indexes = []
        for ndim in range(len(img_shape)):
            if size[ndim] > img_shape[ndim] or size[ndim] < 0:
                size[ndim] = img_shape[ndim]
            if self.random_size:
                size[ndim] = np.random.randint(0, size[ndim])
            if self.localization is not None:
                delta_before = max(
                    self.localization[ndim] - size[ndim] // 2, 0)
            else:
                delta_before = np.random.randint(
                    0, img_shape[ndim] - size[ndim] + 1)
            indexes.append(slice(int(delta_before),
                                 int(delta_before + size[ndim])))
            
        if self.inplace:
            arr[tuple(indexes)] = self.value
            return torch.from_numpy(arr)
        else:
            arr_cut = np.copy(arr)
            arr_cut[tuple(indexes)] = self.value
            return torch.from_numpy(arr_cut)


def interval(obj, lower=None):
    """ Listify an object.

    Parameters
    ----------
    obj: 2-uplet or number
        the object used to build the interval.
    lower: number, default None
        the lower bound of the interval. If not specified, a symetric
        interval is generated.

    Returns
    -------
    interval: 2-uplet
        an interval.
    """
    if isinstance(obj, numbers.Number):
        if obj < 0:
            raise ValueError("Specified interval value must be positive.")
        if lower is None:
            lower = -obj
        return (lower, obj)
    if len(obj) != 2:
        raise ValueError("Interval must be specified with 2 values.")
    min_val, max_val = obj
    if min_val > max_val:
        raise ValueError("Wrong interval boudaries.")
    return tuple(obj)


class Transformer(object):
    """ Class that can be used to register a sequence of transformations.
    """
    Transform = namedtuple("Transform", ["transform", "probability"])

    def __init__(self):
        """ Initialize the class.
        """
        self.transforms = []

    def register(self, transform, probability=1):
        """ Register a new transformation.
        Parameters
        ----------
        transform: callable
            the transformation object.
        probability: float, default 1
            the transform is applied with the specified probability.
        """
        trf = self.Transform(transform=transform, probability=probability, )
        self.transforms.append(trf)

    def __call__(self, arr):
        """ Apply the registered transformations.
        """
        transformed = arr.copy()
        for trf in self.transforms:
            if np.random.rand() < trf.probability:
                transformed = trf.transform(transformed)
        return transformed

    def __str__(self):
        if len(self.transforms) == 0:
            return '(Empty Transformer)'
        s = 'Composition of:'
        for trf in self.transforms:
            s += '\n\t- ' + trf.__str__()
        return s

