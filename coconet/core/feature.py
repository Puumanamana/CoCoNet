"""
Abstract feature class
"""

from pathlib import Path
import logging
import h5py
import numpy as np
import pysam


logger = logging.getLogger('<preprocessing>')

class Feature:
    """
    Abstract class for composition and coverage features
    """

    def __init__(self, name=None, path=None):
        """
        Extract the latent representation of each contig's fragments in the h5 file
        ftype: Feature type (composition or coverage)
        data: {dtype: path} dictionnary

        Returns: list of neighbors for each contigs
        """
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

    def check_file(self, key):
        if key not in self.path:
            return False
        try:
            open(self.path[key])
        except (IOError, TypeError):
            return False
        return True
    
    def check_h5(self, key):
        if self.check_file(key):
            try:
                ids = next(iter(h5py.File(self.path[key])))
            except (OSError, UnicodeDecodeError, StopIteration):
                return False
            return True
        return False

    def check_bam(self, key):
        if self.path.get(key) is None:
            return False
        for bam in self.path[key]:
            try:
                pysam.AlignmentFile(bam, 'rb').check_index()
            except (IOError, ValueError):
                return False
            return True
        return False

    def synchronize(self, other, keys):
        contigs_1 = self.get_contigs(keys[0])
        contigs_2 = other.get_contigs(keys[1])

        diff12 = np.setdiff1d(contigs_1, contigs_2)
        diff21 = np.setdiff1d(contigs_2, contigs_1)
        inter = np.intersect1d(contigs_1, contigs_2)

        infos = []
        if diff12.size > 0:
            self.filter_by_ids(ids=diff12)
            infos.append(f'{diff12.size:,} contigs are only present in the {self.name}')
        if diff21.size > 0:
            other.filter_by_ids(ids=diff21)
            infos.append(f'{diff21.size:,} contigs are only present in the {other.name}')

        if infos:
            info = f"{' and '.join(infos)}. Taking the intersection ({inter.size} contigs)"
            logger.info(info)

    def get_h5_data(self, key='latent'):
        with h5py.File(self.path[key], 'r') as handle:
            data = {contig: matrix[:] for contig, matrix in handle.items()}
            return data
