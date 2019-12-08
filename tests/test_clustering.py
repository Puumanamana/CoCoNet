'''
Tests for clustering procedure
'''

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from coconet.clustering import get_neighbors, compute_pairwise_comparisons, fill_adjacency_matrix
from coconet.clustering import get_communities, iterate_clustering
from .data import generate_coverage_file, generate_rd_model

LOCAL_DIR = Path(__file__).parent

def test_get_neighbors():
    h5_data = generate_coverage_file(*[100]*4, n_samples=5, baselines=[10, 10, 50, 50])
    neighbors = get_neighbors(h5_data)
    h5_data.unlink()

    assert set(neighbors[0]) == {0, 1}
    assert set(neighbors[2]) == {2, 3}

def test_pairwise_comparisons():
    model = generate_rd_model()
    h5_data = {k: generate_coverage_file(8, 8 , 8, n_samples=5, filename=k+'.h5')
               for k in ['composition', 'coverage']}

    handles = {k: h5py.File(v) for k, v in h5_data.items()}
    contigs = ['V0', 'V1', 'V2']
    neighbors = [np.array([0, 1]), np.array([0, 1]), np.array([2])]

    matrix = compute_pairwise_comparisons(model, contigs, handles, neighbors=neighbors, n_frags=5)

    for v in h5_data.values():
        v.unlink()

    assert matrix[0, 0] == 25
    assert matrix[0, 2] == -1
    assert 0 <= matrix[0, 1] <= 25

def test_pairwise_comparisons_no_neighbors():
    model = generate_rd_model()
    h5_data = {k: generate_coverage_file(8, 8 , 8, n_samples=5, filename=k+'.h5')
               for k in ['composition', 'coverage']}

    handles = {k: h5py.File(v) for k, v in h5_data.items()}
    contigs = ['V0', 'V1', 'V2']

    matrix = compute_pairwise_comparisons(model, contigs, handles, n_frags=5)

    for v in h5_data.values():
        v.unlink()

    assert np.all(matrix >= 0) & np.all(matrix <= 25)

def test_fill_adj():
    output = Path('adj_test.npy')
    model = generate_rd_model()
    h5_data = {k: generate_coverage_file(8, 8 , 8, n_samples=5, filename=k+'.h5')
               for k in ['composition', 'coverage']}

    fill_adjacency_matrix(model, h5_data, output, n_frags=5)

    for v in h5_data.values():
        v.unlink()

    assert output.is_file()

    output.unlink()

def test_get_communities():
    adj_mat = np.array([
        [1, 1, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1],
        [0, 0, 1, 1, 1],
        [0, 0, 0, 1, 1]]
    )
    contigs = ['V{}|0'.format(i) for i in range(len(adj_mat))]

    assignments = get_communities(adj_mat, contigs, gamma=0.5)

    assert assignments.shape == (len(adj_mat), 2)
    assert all(assignments.clusters[:2] == 0) and all(assignments.clusters[2:] == 1)

def test_iterate_clustering():
    model = generate_rd_model()
    h5_data = {k: generate_coverage_file(*[10]*5, n_samples=5, filename=k+'.h5')
               for k in ['composition', 'coverage']}

    files = ['singletons.txt', 'adj_mat.npy', 'assignments.csv',
             'refined_assignments.csv', 'refined_adjacency_matrix.npy']

    np.save('adj_mat.npy', np.array([
        [1, 1, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1],
        [0, 0, 1, 1, 1],
        [0, 0, 0, 1, 1]]
    ))

    (pd.DataFrame([['V0', 5, 10, 0, 0]], columns=['contigs', 'length'] + list(range(3)))
     .to_csv(files[0], sep='\t'))

    iterate_clustering(model, h5_data, files[1], files[0], n_frags=5)

    assert all(Path(f).is_file() for f in files)

    for f in files + list(h5_data.values()):
        Path(f).unlink()

    
