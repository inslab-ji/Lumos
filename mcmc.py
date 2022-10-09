import numpy as np
import random


def init(neighbours):
    n = len(neighbours)
    ans = []
    for i in range(n):
        set = {}
        for node in neighbours[i]:
            if np.log(len(neighbours[node])) > np.log(len(neighbours[i])):
                set.add(node)
        ans.append(list(set))
    return ans


def largest(x):
    wl = [len(v) for v in x]
    for i in range(len(x)):
        if wl[i] == max(wl):
            return i, max(wl)


def mcmc(T, x0):
    x = x0
    for _ in range(T):
        u, f = largest(x)
        k = random.randint(1, np.round(np.log(len(x[u]))))
        newx = x
        for i in range(k):
            node = x[u][i]
            if u not in newx[node]:
                newx[node].append(u)
            newx[u].remove(i)
        _, newf = largest(newx)
        p = random.random()
        if p < min(1, np.exp(f-newf)):
            x = newf
    return x
