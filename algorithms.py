# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 16:02:18 2015

@author: navrug
"""

import numpy as np

from scipy.stats import rv_discrete  


# Algorithm from "Online Learning with Feedback Graphs: Beyond Bandits".
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


# Algorithm from "Leveraging Side Observations in Stochastic Bandits".
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


# Algorithm from "Leveraging Side Observations in Stochastic Bandits".
def UCB_MaxN(graph, T, n_runs):
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


# Algorithm attempt for the stochastic general feedback graph case.
# This not a UCB algorithm but it was inspired from it: exploration and 
# exploitation moves are decoupled.
def Generalized_UCB(graph, T, n_runs):
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
    
def Generalized_UCB_inv(graph, alpha, T, n_runs):
    n = len(graph.arms)
    rewards = np.zeros(T, dtype=float)
    for run in range(n_runs):
        X = np.zeros(n, dtype=float)
        O = np.zeros(n, dtype=int)
        for t in range(T):
            if np.random.rand() > 1/(1+alpha*t): # Exploitation move.
                I = np.argmax(np.hstack(X))
            else: # Exploration move.
                I = np.argmax(np.hstack(X + np.sqrt(2*np.log(t)/O)))
                # Argmax of I's in-neighbours.
                J = np.nonzero(graph.in_neighbors(I))[0][np.argmax(X[graph.in_neighbors(I)])]
            reward, feedback = graph.draw(J)
            rewards[t] += float(reward)/n_runs
            for i in range(n):
                if not feedback[i] is None:
                    O[i] += 1
                    X[i] = float(feedback[i])/O[i] + (1.0 - 1.0/O[i])*X[i]
    return rewards
    
def Generalized_UCB_exp(graph, alpha, T, n_runs):
    n = len(graph.arms)
    rewards = np.zeros(T, dtype=float)
    for run in range(n_runs):
        X = np.zeros(n, dtype=float)
        O = np.zeros(n, dtype=int)
        for t in range(T):
            if np.random.rand() > np.exp(-alpha*t): # Exploitation move.
                J = np.argmax(np.hstack(X))
            else: # Exploration move.
                I = np.argmax(np.hstack(X + np.sqrt(2*np.log(t)/O)))
                # Argmax of I's in-neighbours.
                J = np.nonzero(graph.in_neighbors(I))[0][np.argmax(X[graph.in_neighbors(I)])]
            reward, feedback = graph.draw(J)
            rewards[t] += float(reward)/n_runs
            for i in range(n):
                if not feedback[i] is None:
                    O[i] += 1
                    X[i] = float(feedback[i])/O[i] + (1.0 - 1.0/O[i])*X[i]
    return rewards


# Generalization of Thompson sampling to side-observation graphs.
def Thompson_N(graph, T, n_runs):
    n = len(graph.arms)
    rewards = np.zeros(T, dtype=float)
    for run in range(n_runs):
        S = np.ones(n)
        F = np.ones(n)
        for t in range(T):
            theta = np.random.beta(S, F)
            I = np.argmax(theta)
            reward, feedback = graph.draw(I)
            rewards[t] += float(reward)/n_runs
            for i in range(n):
                if not feedback[i] is None:
                    S[i] += feedback[i]
                    F[i] += 1 - feedback[i]
    return rewards
           
           
# Generalization of Thompson sampling to side-observation graphs with the idea
# of using the in-neighbor of max reward taken from UCB-MaxN.
def Thompson_MaxN(graph, T, n_runs):
    n = len(graph.arms)
    rewards = np.zeros(T, dtype=float)
    for run in range(n_runs):
        S = np.zeros(n)
        F = np.zeros(n)
        for t in range(T):
            theta = np.random.beta(S+1, F+1)
            I = np.argmax(theta)
            # Argmax of I's in-neighbours as regards mean reward.
            J = np.nonzero(graph.in_neighbors(I))[0][np.argmax((S/(S+F))[graph.in_neighbors(I)])]
            reward, feedback = graph.draw(J)
            rewards[t] += float(reward)/n_runs
            for i in range(n):
                if not feedback[i] is None:
                    S[i] += feedback[i]
                    F[i] += 1 - feedback[i]
    return rewards
