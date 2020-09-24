'''
Tests for clustering procedure
'''

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import igraph

from coconet.core.composition_feature import CompositionFeature
from coconet.core.coverage_feature import CoverageFeature
from coconet.clustering import (compute_pairwise_comparisons,
                                contig_pair_iterator,
                                make_pregraph,
                                refine_clustering,
                                get_communities)
from .data import generate_h5_file, generate_rd_model

LOCAL_DIR = Path(__file__).parent

def test_pairwise_comparisons():
    model = generate_rd_model()
    h5_data = {k: generate_h5_file(8, 8, 8, n_samples=5, filename=k+'.h5')
               for k in ['composition', 'coverage']}

    handles = [(k, h5py.File(v, 'r')) for k, v in h5_data.items()]

    pair_generators = (((x, y) for (x,y) in [('V0', 'V1'), ('V0', 'V2')]),)

    edges = compute_pairwise_comparisons(model, handles, pair_generators, vote_threshold=0.5)

    for v in h5_data.values():
        v.unlink()

    assert ('V0', 'V1') in edges
    assert ('V0', 'V2') in edges
    assert ('V1', 'V2') not in edges

def test_contig_pair_iterator():
    contigs = ['V0', 'V1', 'V2', 'V3']
    neighbors = [0, 2, 1, 3]
    
    graph = igraph.Graph()
    graph.add_vertices(contigs)
    graph.add_edge('V0', 'V1')
    
    generator = contig_pair_iterator('V0', neighbors, graph, max_neighbors=2)
    all_pairs = list(generator)

    assert all_pairs == [('V0', 'V2'), ('V0', 'V3')]

def test_make_pregraph():
    output = Path('pregraph.pkl')
    model = generate_rd_model()

    h5_files = [generate_h5_file(8, 8, 8, n_samples=5, filename=f'{name}.h5')
                for name in ['compo', 'cover']]
    features = [
        CompositionFeature(path=dict(latent=h5_files[0])),
        CoverageFeature(path=dict(latent=h5_files[1]))
    ]

    make_pregraph(model, features, output)

    features[0].path['latent'].unlink()
    features[1].path['latent'].unlink()

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

def test_refine_clustering():
    model = generate_rd_model()
    h5_files = [generate_h5_file(*[8]*5, n_samples=5, filename=k+'.h5')
                for k in ['composition', 'coverage']]

    features = [
        CompositionFeature(path=dict(latent=h5_files[0])),
        CoverageFeature(path=dict(latent=h5_files[1]))
    ]


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

    refine_clustering(model, features, files[1],
                       singletons_file=files[0],
                       graph_file=files[2],
                       assignments_file=files[3])

    clustering = pd.read_csv(files[3], header=None, index_col=0)[1]

    all_files = all(Path(f).is_file() for f in files)

    for f in files + h5_files:
        Path(f).unlink()

    assert all_files
    assert clustering.loc['V0'] == clustering.loc['V1']
    assert clustering.loc['V2'] == clustering.loc['V3']
    assert clustering.loc['V3'] == clustering.loc['V4']
    assert len(clustering[clustering == clustering.loc['W0']]) == 1


if __name__ == '__main__':
    # test_pairwise_comparisons()
    # test_contig_pair_iterator()
    # test_make_pregraph()
    # test_get_communities()
    test_refine_clustering()
