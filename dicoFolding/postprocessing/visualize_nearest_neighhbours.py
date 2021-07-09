import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

import anatomist.api as anatomist
from soma import aims
import colorado as cld

"""Inspired from lightly https://docs.lightly.ai/tutorials/package/tutorial_simclr_clothing.html
"""

log = logging.getLogger(__name__)

def get_image_as_np_array(filename: str):
    """Returns an image as an numpy array
    """
    img = Image.open(filename)
    return np.asarray(img)

def get_input(dataset, filenames, idx):
    """gets input numbered idx"""
    
    (views, filename) = dataset[idx//2]
    if filename != filenames[idx]:
        log.error("filenames dont match: {} != {}".format(filename, filenames[idx]))
    return views[idx%2]


def plot_knn_examples(embeddings, filenames, dataset, n_neighbors=3, num_examples=6):
    """Plots multiple rows of random images with their nearest neighbors
    """
    # lets look at the nearest neighbors for some samples
    # we use the sklearn library
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)

    # get 5 random samples
    samples_idx = np.random.choice(
        len(indices), size=num_examples, replace=False)

    # loop through our randomly picked samples
    for idx in samples_idx:
        fig = plt.figure()
        # loop through their nearest neighbors
        for plot_x_offset, neighbor_idx in enumerate(indices[idx]):
            # add the subplot
            ax = fig.add_subplot(1, len(indices[idx]), plot_x_offset + 1)
            # Recovers input
            view = get_input(dataset, filenames, neighbor_idx)
            # plot the image
            plt.imshow(view[0,view.shape[1]//2, :, :].numpy())
            # set the title to the distance of the neighbor
            ax.set_title(f'd={distances[idx][plot_x_offset]:.3f}')
            # let's disable the axis
            plt.axis('off')
    plt.show()
    
def create_array_without_hull(view):
    im2 = view[0,:].numpy()
    for k in range(len(im2)):
        for i in range(len(im2[k])):
            for j in range(len(im2[k][i])):
                vox = im2[k][i][j]
                if vox>1 and vox != 11: # On est sur un sillon
                        if im2[k-1][i][j]==11 or im2[k+1][i][j]==11 or im2[k][i-1][j]==11 or im2[k][i+1][j]==11 or im2[k][i][j-1]==11 or im2[k][i][j+1]==11:
                            im2[k][i][j]=11
    im2[im2==0] = 10
    im2[im2!=10] =0
    return im2

def create_mesh_from_array(im):
    input_vol = aims.Volume(im)
    input_mesh = cld.aims_tools.volume_to_mesh(input_vol)
    return input_mesh
    
def plot_knn_meshes(embeddings, filenames, dataset, n_neighbors=3, num_examples=6):
    """Plots multiple rows of random images with their nearest neighbors
    """
    # lets look at the nearest neighbors for some samples
    # we use the sklearn library
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)

    # get 5 random samples
    samples_idx = np.random.choice(
        len(indices), size=num_examples, replace=False)

    a = anatomist.Anatomist()
    view = get_input(dataset, filenames, 0)
    im = create_array_without_hull(view)
    mesh = create_mesh_from_array(im)
    
    aw = a.createWindow('3D')
    am = a.toAObject(mesh)
    a.addObjects(am, aw)
    # block = a.AWindowsBlock(a, n_neighbors)
    

    # # loop through our randomly picked samples
    # for idx in samples_idx:
    #     # loop through their nearest neighbors
    #     for plot_x_offset, neighbor_idx in enumerate(indices[idx]):
    #         # add the subplot
    #         ax = fig.add_subplot(1, len(indices[idx]), plot_x_offset + 1)
    #         # Recovers input
    #         view = get_input(dataset, filenames, neighbor_idx)
    #         # plot the image
    #         plt.imshow(view[0,view.shape[1]//2, :, :].numpy())
    #         # set the title to the distance of the neighbor
    #         ax.set_title(f'd={distances[idx][plot_x_offset]:.3f}')
    #         # let's disable the axis
    #         plt.axis('off')

if __name__ == "__main__":
    n_samples = 20
    n_features = 10
    embeddings = np.random.rand(n_samples, n_features)
    plot_knn_examples(embeddings)