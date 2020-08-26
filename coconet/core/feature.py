import h5py
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


class Feature:

    def __init__(self, ftype=None, path=None):
        '''
        Extract the latent representation of each contig's fragments in the h5 file
        ftype: Feature type (composition or coverage)
        data: {dtype: path} dictionnary

        Returns: list of neighbors for each contigs
        '''
        self.ftype = ftype
        self.path = path
        
    def get_handle(self, key='h5'):
        return h5py.File(self.path[key], 'r')

    def get_neighbors(self):
        if not 'latent' in self.path:
            return
        
        handle = self.get_handle('latent')

        # data.shape = ( n_contigs, n_frags, latent_dim )
        data = np.stack([np.array(handle.get(ctg)[:]) for ctg in handle.keys()])
        # center of each contigs (n_contigs, latent_dim)
        contig_centers = np.mean(data, axis=1)
        # pairwise distances between contig centers
        pairwise_distances = euclidean_distances(contig_centers)
        # distance of each fragment to its center
        distance_to_resp_center = np.sqrt(np.sum(
            (data - contig_centers[:, None, :])**2, axis=2
        ))
        # radius of each contig (mean+2*std)(distance) from fragment to center
        radii = np.mean(distance_to_resp_center, axis=1) + 2*np.std(distance_to_resp_center, axis=1)
        # Condition: neighbors need to be within [radius] units from the center
        within_range = pairwise_distances < radii.reshape(-1, 1)

        # get neighbors indices and sort them by distance
        indices = np.arange(len(radii))
        neighbors_ordered = [indices[wr][pairwise_distances[i, wr].argsort()]
                             for i, wr in enumerate(within_range)]

        handle.close()
        
        return neighbors_ordered
