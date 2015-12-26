# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 16:02:18 2015

@author: navrug
"""

import numpy as np

from scipy.stats import rv_discrete  



def Exp3G(eta, gamma, U, graph, T, n_runs):
    n = len(graph.arms)
    u = np.zeros(n, dtype=float)
    u[U] = 1.0/len(U)
    rewards = np.zeros(T, dtype=float)
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
    return rewards


def UCB_N(graph, T, n_runs):
    n = len(graph.arms)
    rewards = np.zeros(T, dtype=float)
    for run in range(n_runs):
        X = np.zeros(n, dtype=float)
        O = np.zeros(n, dtype=int)
        for t in range(T):
            I = np.argmax(X + np.sqrt(2*np.log(t)/O))
            reward, feedback = graph.draw(I)
            rewards[t] += float(reward)/n_runs
            for i in range(n):
                if not feedback[i] is None:
                    O[i] += 1
                    X[i] = float(feedback[i])/O[i] + (1.0 - 1.0/O[i])*X[i]
    return rewards


def UCB_maxN(graph, T, n_runs):
    n = len(graph.arms)
    rewards = np.zeros(T, dtype=float)
    for run in range(n_runs):
        X = np.zeros(n, dtype=float)
        O = np.zeros(n, dtype=int)
        for t in range(T):
            I = np.argmax(X + np.sqrt(2*np.log(t)/O))
            # Argmax of I's in-neighbours.
            J = np.nonzero(graph.in_neighbors(I))[0][np.argmax(X[graph.in_neighbors(I)])]
            reward, feedback = graph.draw(J)
            rewards[t] += float(reward)/n_runs
            for i in range(n):
                if not feedback[i] is None:
                    O[i] += 1
                    X[i] = float(feedback[i])/O[i] + (1.0 - 1.0/O[i])*X[i]
    return rewards

def generalized_UCB(graph, T, n_runs):
    n = len(graph.arms)
    rewards = np.zeros(T, dtype=float)
    for run in range(n_runs):
        X = np.zeros(n, dtype=float)
        O = np.zeros(n, dtype=int)
        for t in range(T):
            I = np.argmax(np.hstack((X, np.sqrt(2*np.log(t)/O))))
            if I < n: # Exploitation move.
                J = I
            else: # Exploration move.
                # Argmax of I's in-neighbours.
                J = np.nonzero(graph.in_neighbors(I-n))[0][np.argmax(X[graph.in_neighbors(I-n)])]
            reward, feedback = graph.draw(J)
            rewards[t] += float(reward)/n_runs
            for i in range(n):
                if not feedback[i] is None:
                    O[i] += 1
                    X[i] = float(feedback[i])/O[i] + (1.0 - 1.0/O[i])*X[i]
    return rewards

