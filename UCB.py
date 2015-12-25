# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 19:06:02 2015

@author: Gurvan
"""

import matplotlib.pyplot as plt
import numpy as np

from feedback_graph import FeedbackGraph

n = 5

A = np.eye(5, dtype=bool)
np_rng = np.random.RandomState(0)
graph = FeedbackGraph(A, np_rng=np_rng)
T = 10000
n_runs = 1
rewards = np.zeros(T, dtype=float)



for run in range(n_runs):
    X = np.zeros(n, dtype=float)
    O = np.zeros(n, dtype=int)
    for t in range(T):
        I = np.argmax(X + np.sqrt(2*np.log(t)/O))
        reward, feedback = graph.draw(I)
        rewards[t] += float(reward)/n_runs
        r = np.zeros(n, dtype=float)
        for i in range(n):
            if not feedback[i] is None:
                O[i] += 1
                X[i] = float(feedback[i])/O[i] + (1.0 - 1.0/O[i])*X[i]

for arm in graph.arms:
    print arm.get_mean()
print X

plt.plot(range(T), graph.best_mean()*np.array(range(T)) - np.cumsum(rewards))

    