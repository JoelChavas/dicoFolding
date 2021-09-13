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

import PIL
import logging
import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib.markers as mmarkers
from sklearn.manifold import TSNE
import io
from torchvision.transforms import ToTensor
from torchvision.transforms import Resize

logger = logging.getLogger(__name__)


def mscatter(x, y, ax=None, m=None, **kw):
    if not ax:
        ax = plt.gca()
    sc = ax.scatter(x, y, **kw)
    if (m is not None) and (len(m) == len(x)):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(
                marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    return sc


def compute_embeddings_skeletons(loader, model, num_outputs):
    X = torch.zeros([0, num_outputs]).cpu()
    with torch.no_grad():
        for (inputs, filenames) in loader:
            # First views of the whole batch
            inputs = inputs.cuda()
            model = model.cuda()
            X_i = model.forward(inputs[:, 0, :])
            # Second views of the whole batch
            X_j = model.forward(inputs[:, 1, :])
            # First views and second views are put side by side
            X_reordered = torch.cat([X_i, X_j], dim=-1)
            X_reordered = X_reordered.view(-1, X_i.shape[-1])
            X = torch.cat((X, X_reordered.cpu()), dim=0)
            del inputs
    return X


def compute_tsne(loader, model, num_outputs):
    X = compute_embeddings_skeletons(loader, model, num_outputs)
    tsne = TSNE(n_components=2, perplexity=5, init='pca', random_state=50)
    X_tsne = tsne.fit_transform(X.detach().numpy())

    return X_tsne


def plot_tsne(X_tsne, buffer, labels=None):
    """Generates TSNE plot either in a PNG image buffer or as a plot

    Args:
        X_tsne: TSNE N_features rows x 2 columns
        buffer (boolean): True -> returns PNG image buffer
                          False -> plots the figure
    """
    fig, ax = plt.subplots(1)
    logger.info(X_tsne.shape)
    nb_points = X_tsne.shape[0]
    m = np.repeat(["o"], nb_points)
    if labels is None:
        c = np.tile(np.array(["b", "r"]), nb_points // 2)
    else:
        c = labels
        
    mscatter(X_tsne[:, 0], X_tsne[:, 1], c=c, m=m, s=8, ax=ax)

    if buffer:
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close('all')
        image = PIL.Image.open(buf)
        image = ToTensor()(image).unsqueeze(0)[0] 
        return image
    else:
        plt.show()
        
def plot_img(img, buffer):
    """Plots one 2D slice of one of the 3D images of the batch

    Args:
        img: batch of images of size [N_batch, 1, size_X, size_Y, size_Z]
        buffer (boolean): True -> returns PNG image buffer
                          False -> plots the figure
    """
    plt.imshow(img[0, 0, img.shape[2]//2, :, :])

    if buffer:
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close('all')
        image = PIL.Image.open(buf)
        image = ToTensor()(image).unsqueeze(0)[0] 
        return image
    else:
        plt.show()

def prime_factors(n):
    i = 2
    factors = []
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return factors

def plot_output(img, buffer):
    
    arr = (img[0,:]).detach().numpy()
    # Reshapes the array into a 2D array
    primes = prime_factors(arr.size)
    row_size = np.prod(primes[:len(primes)//2])
    arr = arr.reshape(row_size, -1)
    
    plt.imshow(arr)
    
    if buffer:
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close('all')
        image = PIL.Image.open(buf)
        image = ToTensor()(image).unsqueeze(0)[0] 
        return image
    else:
        plt.show()