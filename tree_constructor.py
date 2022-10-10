import numpy as np
import argparse
from torch_geometric.utils import sort_edge_index
import pickle
import math


def encode(x, m):
    n, d = x.size()
    alpha = max(x)
    beta = min(x)
    em = math.exp(math.exp(args.epsilon / m))
    p = (x - alpha) / (beta - alpha)
    p = (p * (em - 1) + 1) / (em + 1)
    t = np.random.binomial(p)
    x_star = 2 * t - 1
    x_prime = d * (beta - alpha) / (2 * m)
    x_prime = x_prime * (em + 1) * x_star / (em - 1)
    x_prime = x_prime + (alpha + beta) / 2
    return x_prime


def construct_tree(x, features):
    def en2t(node, neighbours, features, begin_index, dimf):
        edges = []
        n = len(neighbours)
        dict_node = np.zeros(n * 3 + 1)
        tree_features = np.zeros((n * 3 + 1, dimf))
        for i in range(n):
            edges.append([begin_index + i, begin_index + i + n * 2])
            edges.append([begin_index + i + n * 2, begin_index + i])
            edges.append([begin_index + i + n, begin_index + i + n * 2])
            edges.append([begin_index + i + n * 2, begin_index + i + n])
            edges.append([begin_index + i + n * 2, begin_index + n * 3])
            edges.append([begin_index + n * 3, begin_index + n * 2 + i])
            dict_node[i] = node
            dict_node[i + n] = neighbours[i]
            tree_features[i] = features[node]
            tree_features[i + n] = encode(features[neighbours[i]], len(neighbours[neighbours[i]]))
        return edges, dict_node, tree_features, begin_index + n * 3 + 1

    n = len(x)
    edges = []
    dict_node = np.empty(0, dtype=np.int32)
    dimf = len(features[0])
    tree_features = np.empty(shape=(0, dimf))
    index = 0
    for i in range(n):
        ne, nd, nf, index = en2t(i, x[i], features, index, dimf)
        edges.append(ne)
        dict_node = np.append(dict_node, nd)
        tree_features = np.concatenate((tree_features, nf), axis=0)
        if i % int(n / 10) == 0:
            print("Finish " + str(i) + " nodes out of " + str(n))
    return edges, tree_features, dict_node


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Facebook')
parser.add_argument('--epsilon', type=float, default=2)
args = parser.parse_args()
if args.dataset == "Facebook":
    from torch_geometric.datasets import FacebookPagePage

    dataset = FacebookPagePage("./facebook")
    data = dataset[0]
else:
    from torch_geometric.datasets import LastFMAsia

    dataset = LastFMAsia("./lastfm")
    data = dataset[0]
edge_index = sort_edge_index(data.edge_index)
with open("./" + args.dataset + "/solution.pck", "rb") as file:
    so = pickle.load(file)
edges, tree_features, dict_node = construct_tree(so, data.x)
with open("./" + args.dataset + "/edges.pck", "wb") as file:
    pickle.dump(edges, file)
with open("./" + args.dataset + "/tf.pck", "wb") as file:
    pickle.dump(tree_features, file)
with open("./" + args.dataset + "/dn.pck", "wb") as file:
    pickle.dump(dict_node, file)
