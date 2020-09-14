import logging
import h5py
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


logger = logging.getLogger('preprocessing')

class Feature:

    def __init__(self, name=None, path=None):
        '''
        Extract the latent representation of each contig's fragments in the h5 file
        ftype: Feature type (composition or coverage)
        data: {dtype: path} dictionnary

        Returns: list of neighbors for each contigs
        '''
        self.name = name
        self.path = path

    def __str__(self):
        paths = [f'{name}: {str(path)}' for (name, path) in self.path.items()
                 if path.is_file()]
        msg = [
            f'Feature-{self.name}',
            f'Associated files-{",".join(paths)}'
        ]
        return '\n'.join(msg)

    def check_paths(self):
        for p in self.path.values():
            if isinstance(p, list):
                if all(pi.is_file() for pi in p):
                    return True
            if p is not None and p.is_file():
                return True

    def get_handle(self, key, mode='r'):
        return h5py.File(self.path[key], mode)

    def synchronize(self, other, keys):
        contigs_1 = self.get_contigs(keys[0])
        contigs_2 = other.get_contigs(keys[1])

        diff12 = np.setdiff1d(contigs_1, contigs_2)
        diff21 = np.setdiff1d(contigs_2, contigs_1)
        inter = np.intersect1d(contigs_1, contigs_2)

        warning = []
        if diff12.size > 0:
            self.filter_by_ids(ids=diff12)
            warning.append(f'{diff12.size:,} contigs are only present in the {self.name}')
        if diff21.size > 0:
            other.filter_by_ids(ids=diff21)
            warning.append(f'{diff21.size:,} contigs are only present in the {other.name}')

        warning = f"{' and '.join(warning)}. Taking the intersection ({inter.size} contigs)"
        logger.warning(warning)


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
