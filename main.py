# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 17:13:45 2015

@author: navrug
"""

import matplotlib.pyplot as plt
import numpy as np

from algorithms import *
from bandit import BernoulliArm
from feedback_graph import FeedbackGraph

#%%
np.random.seed(seed=0)
np_rng = np.random.RandomState(0)
n = 5
T = 1000
n_runs = 100


A_full = np.ones((n,n), dtype=bool)
A_bandit = np.eye(n, dtype=bool)
A_loopless = A_full ^ A_bandit
A_revealing = np.vstack((np.ones((1,n), dtype=bool),np.zeros((n-1, n), dtype=bool)))
A_unobservable = np.ones((n,n), dtype=bool)
A_weak[0:2,0] = False

arms = []
for i in range(n):
    arms.append(BernoulliArm(mean=float(i+1)/(n+1), np_rng=np_rng))

full = FeedbackGraph(A_full, arms=arms, np_rng=np_rng)
bandit = FeedbackGraph(A_bandit, arms=arms, np_rng=np_rng)
loopless = FeedbackGraph(A_loopless, arms=arms, np_rng=np_rng)
revealing = FeedbackGraph(A_revealing, arms=arms, np_rng=np_rng)
weak = FeedbackGraph(A_weak, arms=arms, np_rng=np_rng)



# Learning rate.
eta = 0.1
# Regularisation parameter.
gamma = 0.01
# List of nodes.
U = range(n)



graphs = [full, bandit, loopless, revealing, weak]
names = ["Full", "Bandit", "Loopless", "Revealing", "Weak"]

full = False
if full:
    for graph, name in zip(graphs, names):
        print "Working with", name
        Exp3G_rew = Exp3G(eta, gamma, U, graph, T, n_runs)
        UCB_N_rew = UCB_N(graph, T, n_runs)
        UCB_MaxN_rew = UCB_MaxN(graph, T, n_runs)
        Generalized_UCB_rew = Generalized_UCB(graph, T, n_runs)
        Generalized_UCB_inv_rew = Generalized_UCB_inv(graph, T, n_runs)
        Generalized_UCB_exp_rew = Generalized_UCB_exp(graph, T, n_runs)
        Thompson_N_rew = Thompson_N(graph, T, n_runs)
        Thompson_MaxN_rew = Thompson_MaxN(graph, T, n_runs)
        plt.figure()
        plt.plot(range(T), graph.best_mean()*np.array(range(T)) - np.cumsum(Exp3G_rew))
        plt.plot(range(T), graph.best_mean()*np.array(range(T)) - np.cumsum(UCB_N_rew))
        plt.plot(range(T), graph.best_mean()*np.array(range(T)) - np.cumsum(UCB_MaxN_rew))
        plt.plot(range(T), graph.best_mean()*np.array(range(T)) - np.cumsum(Generalized_UCB_rew))
        plt.plot(range(T), graph.best_mean()*np.array(range(T)) - np.cumsum(Thompson_N_rew))
        plt.plot(range(T), graph.best_mean()*np.array(range(T)) - np.cumsum(Thompson_MaxN_rew))
        plt.legend(["Exp3G","UCB-N","UCB-MaxN","Generalized-UCB","Thompson-N","Thompson-MaxN"], loc="upper left")
        plt.suptitle(name, fontsize=20)
        plt.xlabel("Time")
        plt.ylabel("Regret")
        
else:
# Comparison of generalized UCBs.
    for graph, name in zip(graphs, names):
        print "Working with", name
        Exp3G_rew = Exp3G(eta, gamma, U, graph, T, n_runs)
        Generalized_UCB_rew = Generalized_UCB(graph, T, n_runs)
        Generalized_UCB_inv_rew = Generalized_UCB_inv(graph, 0.1, T, n_runs)
        Generalized_UCB_exp_rew = Generalized_UCB_exp(graph, 0.1, T, n_runs)
        f = plt.figure()
        plt.plot(range(T), graph.best_mean()*np.array(range(T)) - np.cumsum(Exp3G_rew))
        plt.plot(range(T), graph.best_mean()*np.array(range(T)) - np.cumsum(Generalized_UCB_rew))
        plt.plot(range(T), graph.best_mean()*np.array(range(T)) - np.cumsum(Generalized_UCB_inv_rew))
        plt.plot(range(T), graph.best_mean()*np.array(range(T)) - np.cumsum(Generalized_UCB_exp_rew))
        plt.legend(["Exp3G","Generalized-UCB","Generalized-UCB-inv","Generalized-UCB-exp"], loc="upper left")
        plt.suptitle(name, fontsize=20)
        plt.xlabel("Time")
        plt.ylabel("Regret")
        f.savefig("figure/"+name+'.pdf', bbox_inches='tight')