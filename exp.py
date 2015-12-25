# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 19:06:02 2015

@author: Gurvan
"""

import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import rv_discrete  

from feedback_graph import FeedbackGraph

n = 5

A = np.eye(5, dtype=bool)
np_rng = np.random.RandomState(0)
graph = FeedbackGraph(A, np_rng=np_rng)
T = 10000
n_runs = 1
rewards = np.zeros(T, dtype=float)

# Learning rate.
eta = 0.1
# Regularisation parameter.
gamma = 0.01
# List of nodes.
U = [0,1,3]
# Uniform distribution over the nodes of U.
u = np.zeros(n, dtype=float)
u[U] = 1.0/len(U)

for run in range(n_runs):
    q = np.ones(n, dtype=float)/n
    for t in range(T):
        p = (1-gamma)*q + gamma*u
        rv = rv_discrete(values=(range(n), p))
        I = rv.rvs()
        reward, feedback = graph.draw(I)
        rewards[t] += float(reward)/n_runs
        r = np.zeros(n, dtype=float)
        for i in range(n):
            if feedback[i] is None:
                r[i] = 0.0
            else:
                r[i] = feedback[i]/np.sum(p[graph.in_neighbors(i)])
        q = q*np.exp(eta*r)
        q /= np.sum(q)

for arm in graph.arms:
    print arm.get_mean()
print p

plt.plot(range(T), graph.best_mean()*np.array(range(T)) - np.cumsum(rewards))

    