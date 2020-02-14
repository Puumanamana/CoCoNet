'''
Tests for clustering procedure
'''

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import igraph

from coconet.clustering import get_neighbors, compute_pairwise_comparisons, make_pregraph
from coconet.clustering import get_communities, iterate_clustering
from .data import generate_h5_file, generate_rd_model

LOCAL_DIR = Path(__file__).parent

def test_get_neighbors():
    h5_data = generate_h5_file(*[100]*4, n_samples=5, baselines=[10, 10, 50, 50])
    neighbors = get_neighbors(h5_data)
    h5_data.unlink()

    assert set(neighbors[0]) == {0, 1}
    assert set(neighbors[2]) == {2, 3}

def test_pairwise_comparisons():
    model = generate_rd_model()
    h5_data = {k: generate_h5_file(8, 8, 8, n_samples=5, filename=k+'.h5')
               for k in ['composition', 'coverage']}

    handles = {k: h5py.File(v) for k, v in h5_data.items()}
    contigs = ['V0', 'V1', 'V2']

    graph = igraph.Graph()
    graph.add_vertices(contigs)
    graph.es['weight'] = []
    neighbors = [np.array([0, 1]), np.array([0, 1]), np.array([2])]

    compute_pairwise_comparisons(model, graph, handles, neighbors=neighbors, n_frags=5)

    for v in h5_data.values():
        v.unlink()

    assert (0, 1) in graph.get_edgelist()
    assert (1, 2) not in graph.get_edgelist()
    assert 0 <= graph.es['weight'][0] <= 25

def test_make_pregraph():
    output = Path('pregraph.pkl')
    model = generate_rd_model()
    h5_data = {k: generate_h5_file(8, 8, 8, n_samples=5, filename=k+'.h5')
               for k in ['composition', 'coverage']}

    make_pregraph(model, h5_data, output, n_frags=5)

    for v in h5_data.values():
        v.unlink()

    assert output.is_file()

    output.unlink()

def test_get_communities():
    adj = np.array([
        [10, 10, 0, -1, -1],
        [9, 10, 0, 0, 0],
        [0, 0, 10, 8, 9],
        [0, -1, 9, 10, 10],
        [0, 0, 0, 10, 10]])

    contigs = ['V{}|0'.format(i) for i in range(len(adj))]

    graph = igraph.Graph()
    graph.add_vertices(contigs)

    edges = [(i, j, adj[i,j]) for i in range(len(contigs)) for j in range(len(contigs))
             if adj[i, j] >= 0]
    for i, j, w in edges:
        graph.add_edge(i, j, weight=w)

    get_communities(graph, 8, gamma=0.5)
    clusters = np.array(graph.vs['cluster'])

    assert clusters[0] == clusters[1]
    assert all(clusters[2:] == clusters[2])
    assert clusters[1] != clusters[2]

def test_iterate_clustering():
    model = generate_rd_model()
    h5_data = {k: generate_h5_file(*[8]*5, n_samples=5, filename=k+'.h5')
               for k in ['composition', 'coverage']}

    files = ['singletons.txt', 'pre_graph.pkl', 'graph.pkl', 'assignments.csv']

    adj = np.array([[25, 24, 0, 0, -1],
                    [23, 25, 0, -1, 0],
                    [0, -1, 25, 25, 25],
                    [-1, 0, 24, 25, 24],
                    [0, -1, 18, 23, 25]])

    contigs = ["V{}".format(i) for i in range(5)]
    edges = [(i, j, adj[i, j]) for i in range(len(contigs)) for j in range(len(contigs))
             if adj[i, j] >= 0]

    graph = igraph.Graph()
    graph.add_vertices(contigs)
    for i, j, w in edges:
        graph.add_edge(i, j, weight=w)

    graph.write_pickle(files[1])

    # Singleton to be included in the end
    (pd.DataFrame([['W0', 5, 10, 0, 0]], columns=['contigs', 'length'] + list(range(3)))
     .to_csv(files[0], sep='\t'))

    iterate_clustering(model, h5_data, files[1],
                       singletons_file=files[0],
                       graph_file=files[2],
                       assignments_file=files[3],
                       n_frags=5)

    clustering = pd.read_csv(files[3], header=None, index_col=0)[1]

    all_files = all(Path(f).is_file() for f in files)

    for f in files + list(h5_data.values()):
        Path(f).unlink()

    assert all_files
    assert clustering.loc['V0'] == clustering.loc['V1']
    assert clustering.loc['V2'] == clustering.loc['V3']
    assert clustering.loc['V3'] == clustering.loc['V4']
    assert len(clustering[clustering == clustering.loc['W0']]) == 1

if __name__ == '__main__':
    # test_get_communities()
    test_pairwise_comparisons()
    # test_iterate_clustering()
    # test_make_pregraph()
