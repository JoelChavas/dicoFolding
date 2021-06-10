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

import logging
from dicoFolding.models.densenet import densenet121
import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib.markers as mmarkers
from sklearn.manifold import TSNE


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


def compute_tsne(loader, model):
    X = torch.zeros([0, 128]).cpu()
    with torch.no_grad():
        model.model.eval()
        for (inputs, filenames) in loader:
            X_i = model.model(inputs[:, 0, :])
            X_j = model.model(inputs[:, 1, :])
            X = torch.cat((X, X_i.cpu(), X_j.cpu()), dim=0)
            del inputs
    tsne = TSNE(n_components=2, perplexity=5, init='pca', random_state=50)
    X_tsne = tsne.fit_transform(X.detach().numpy())

    return X_tsne


def plot_tsne(X_tsne_before, X_tsne_after, config):
    fig, ax = plt.subplots(2)
    print(X_tsne_before.shape)
    print(X_tsne_after.shape)
    m = np.repeat(["o", "o", "s", "s", "D", "D", "*", "*", "<", "<",
                   "P", "P", "h", "h", "v", "v"], 10)
    mscatter(X_tsne_before[:, 0], X_tsne_before[:, 1], c='b', m=m, ax=ax[0])
    mscatter(X_tsne_after[:, 0], X_tsne_after[:, 1], c='b', m=m, ax=ax[1])
    plt.show()


logger = logging.getLogger(__name__)


def load(path='/home/jc225751/Runs/09_CUDA_hcp/Output/Contrastive_MRI_epoch_3.pth'):
    checkpoint = None
    try:
        checkpoint = torch.load(
            path, map_location=lambda storage, loc: storage)
    except BaseException as e:
        logger.error('Impossible to load the checkpoint: %s' % str(e))

    model = densenet121(mode="encoder", drop_rate=0.0)
    if checkpoint is not None:
        try:
            if hasattr(checkpoint, "state_dict"):
                unexpected = model.load_state_dict(checkpoint.state_dict())
                logger.info('Model loading info: {}'.format(unexpected))
            elif isinstance(checkpoint, dict):
                if "model" in checkpoint:
                    unexpected = model.load_state_dict(
                        checkpoint["model"], strict=False)
                    logger.info('Model loading info: {}'.format(unexpected))
            else:
                unexpected = model.load_state_dict(checkpoint)
                logger.info('Model loading info: {}'.format(unexpected))
        except BaseException as e:
            raise ValueError(
                'Error while loading the model\'s weights: %s' %
                str(e))


if __name__ == "__main__":
    load()
