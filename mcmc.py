import numpy as np
import random


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
