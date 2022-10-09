import numpy as np
import random
import argparse
from torch_geometric.utils import sort_edge_index
import torch_geometric.transforms as T
import pickle


def neighbours(edges, num_nodes):
    src, tag = edges
    n = []
    si = 0
    for i in range(num_nodes):
        l = []
        while si < len(src) and src[si] == i:
            l.append(int(tag[si]))
            si += 1
        n.append(l)
    return n


def init(neighbours):
    n = len(neighbours)
    ans = []
    for i in range(n):
        s = set()
        for node in neighbours[i]:
            if np.log(len(neighbours[node])) > np.log(len(neighbours[i])):
                s.add(node)
        ans.append(s)
    return ans


def largest(x):
    wl = [len(v) for v in x]
    for i in range(len(x)):
        if wl[i] == max(wl):
            return i, max(wl)


def mcmc(T, x0):
    x = x0
    for t in range(T):
        u, f = largest(x)
        print("Round "+str(t)+" f(x)="+str(f))
        k = random.randint(1, np.round(np.log(len(x[u]))))
        newx = x
        nodes = random.sample(x[u], k)
        for node in nodes:
            if u not in newx[node]:
                newx[node].add(u)
            newx[u].remove(node)
        _, newf = largest(newx)
        p = random.random()
        if p < min(1, np.exp(f-newf)):
            x = newx
    x = [list(i) for i in x]
    return x


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Facebook')
parser.add_argument('--mcmcepochs', type=int, default=10)
args = parser.parse_args()
if args.dataset == "Facebook":
    from torch_geometric.datasets import FacebookPagePage
    dataset = FacebookPagePage("./facebook", transform=T.NormalizeFeatures())
    data = dataset[0]
else:
    from torch_geometric.datasets import LastFMAsia
    dataset = LastFMAsia("./lastfm", transform=T.NormalizeFeatures())
    data = dataset[0]
edge_index = sort_edge_index(data.edge_index)
neighs = neighbours(edge_index, data.num_nodes)
init_s = init(neighs)
final_s = mcmc(args.mcmcepochs, init_s)
with open("./"+args.dataset+"/solution.pck", "wb") as file:
    pickle.dump(final_s, file)

