import numpy as np
from collections import defaultdict


def construct_tree(x, features):
    def en2t(node, neighbours, features, begin_index, dimf):
        edges = []
        n = len(neighbours)
        dict_node = {node: []}
        tree_features = np.zeros((n * 3 + 1, dimf))
        for i in range(n):
            edges.append([begin_index + i, begin_index + i + n * 2])
            edges.append([begin_index + i + n * 2, begin_index + i])
            edges.append([begin_index + i + n, begin_index + i + n * 2])
            edges.append([begin_index + i + n * 2, begin_index + i + n])
            edges.append([begin_index + i + n * 2, begin_index + n * 3])
            edges.append([begin_index + n * 3, begin_index + n * 2 + i])
            dict_node[node].append(begin_index + i)
            dict_node[neighbours[n]] = [begin_index + i + n]
            tree_features[i] = features[node]
            tree_features[i + n] = features[neighbours[n]]
        return edges, dict_node, tree_features, begin_index + n * 3 + 1

    n = len(x)
    edges = []
    dict_node = defaultdict(list)
    dimf = len(features[0])
    tree_features = np.empty(shape=(0, dimf))
    index = 0
    for i in range(n):
        ne, nd, nf, index = en2t(i, x[i], features, index, dimf)
        edges.append(ne)
        for k, items in nd.items():
            dict_node[k].append(items)
        tree_features = np.concatenate((tree_features, nf), axis=0)
    return edges, tree_features, dict_node