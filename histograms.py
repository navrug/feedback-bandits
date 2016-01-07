# -*- coding: utf-8 -*-
"""
Created on Wed Jan 06 16:09:24 2016

@author: Gurvan
"""

import numpy as np

from algorithms import *
from bandit import BernoulliArm
from feedback_graph import FeedbackGraph

#%%
np.random.seed(seed=0)
np_rng = np.random.RandomState(0)
n = 5
T = 1000
n_runs = 1000


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


biggest = T
Exp3G_hist = np.zeros((len(graphs), 2))
UCB_N_hist =np.zeros((len(graphs), 2))
UCB_MaxN_hist = np.zeros((len(graphs), 2))
Thompson_N_hist = np.zeros((len(graphs), 2))
Thompson_MaxN_hist = np.zeros((len(graphs), 2))
Generalized_UCB_hist = np.zeros((len(graphs), 2))
Generalized_UCB_inv_hist = np.zeros((len(graphs), 2))
Generalized_UCB_exp_hist = np.zeros((len(graphs), 2))
for run in range(n_runs):
    # Drawing a bandit configuration
    means = np.random.rand(n)
    arms = []
    for i in range(n):
        arms.append(BernoulliArm(mean=means[i], np_rng=np_rng))
    for k, graph in enumerate(graphs):
        graph.arms = arms
        opt = graph.best_mean()*T
        Exp3G_rew = Exp3G(eta, gamma, U, graph, T, 1)
        UCB_N_rew = UCB_N(graph, T, 1)
        UCB_MaxN_rew = UCB_MaxN(graph, T, 1)
        Thompson_N_rew = Thompson_N(graph, T, 1)
        Thompson_MaxN_rew = Thompson_MaxN(graph, T, 1)
        Generalized_UCB_rew = Generalized_UCB(graph, T, 1)
        Generalized_UCB_inv_rew = Generalized_UCB_inv(graph, 0.1, T, 1)
        Generalized_UCB_exp_rew = Generalized_UCB_exp(graph, 0.1, T, 1)
        Exp3G_hist[k, 0] += float(opt-np.sum(Exp3G_rew))
        Exp3G_hist[k, 1] += float((opt-np.sum(Exp3G_rew))**2)
        UCB_N_hist[k, 0] += float(opt-np.sum(UCB_N_rew))
        UCB_N_hist[k, 1] += float((opt-np.sum(UCB_N_rew))**2)
        UCB_MaxN_hist[k, 0] += float(opt-np.sum(UCB_MaxN_rew))
        UCB_MaxN_hist[k, 1] += float((opt-np.sum(UCB_MaxN_rew))**2)
        Thompson_N_hist[k, 0] += float(opt-np.sum(Thompson_N_rew))
        Thompson_N_hist[k, 1] += float((opt-np.sum(Thompson_N_rew))**2)
        Thompson_MaxN_hist[k, 0] += float(opt-np.sum(Thompson_MaxN_rew))
        Thompson_MaxN_hist[k, 1] += float((opt-np.sum(Thompson_MaxN_rew))**2)
        Generalized_UCB_hist[k, 0] += float(opt-np.sum(Generalized_UCB_rew))
        Generalized_UCB_hist[k, 1] += float((opt-np.sum(Generalized_UCB_rew))**2)
        Generalized_UCB_inv_hist[k, 0] += float(opt-np.sum(Generalized_UCB_inv_rew))
        Generalized_UCB_inv_hist[k, 1] += float((opt-np.sum(Generalized_UCB_inv_rew))**2)
        Generalized_UCB_exp_hist[k, 0] += float(opt-np.sum(Generalized_UCB_exp_rew))
        Generalized_UCB_exp_hist[k, 1] += float((opt-np.sum(Generalized_UCB_exp_rew))**2)
#%%

def confidence(x, v, n):
    return 1.96 * np.sqrt((v/n_runs - (x/n_runs)**2)/n_runs)


regret = []
for k, name in enumerate(names):
    print name
    regret.append((Exp3G_hist[k, 0]/n_runs, 1.96 * np.sqrt((Exp3G_hist[k, 1]/n_runs - (Exp3G_hist[k, 0]/n_runs)**2)/n_runs)))
    regret.append((UCB_N_hist[k, 0]/n_runs, 1.96 * np.sqrt((UCB_N_hist[k, 1]/n_runs - (UCB_N_hist[k, 0]/n_runs)**2)/n_runs)))
    regret.append(( UCB_MaxN_hist[k, 0]/n_runs,1.96 * np.sqrt((UCB_MaxN_hist[k, 1]/n_runs - (UCB_MaxN_hist[k, 0]/n_runs)**2)/n_runs)))
    regret.append((Thompson_N_hist[k, 0]/n_runs, 1.96 * np.sqrt((Thompson_N_hist[k, 1]/n_runs - (Thompson_N_hist[k, 0]/n_runs)**2)/n_runs)))
    regret.append(( Thompson_MaxN_hist[k, 0]/n_runs, 1.96 * np.sqrt((Thompson_MaxN_hist[k, 1]/n_runs - (Thompson_MaxN_hist[k, 0]/n_runs)**2)/n_runs)))
    regret.append((Generalized_UCB_hist[k, 0]/n_runs, 1.96 * np.sqrt((Generalized_UCB_hist[k, 1]/n_runs - (Generalized_UCB_hist[k, 0]/n_runs)**2)/n_runs)))
    regret.append(( Generalized_UCB_inv_hist[k, 0]/n_runs,1.96 * np.sqrt((Generalized_UCB_inv_hist[k, 1]/n_runs - (Generalized_UCB_inv_hist[k, 0]/n_runs)**2)/n_runs)))
    regret.append(( Generalized_UCB_exp_hist[k, 0]/n_runs,1.96 * np.sqrt((Generalized_UCB_exp_hist[k, 1]/n_runs - (Generalized_UCB_exp_hist[k, 0]/n_runs)**2)/n_runs)))
for k, name in enumerate(names):
    print name
    print "ExpG yields", Exp3G_hist[k, 0]/n_runs,"+/-", 1.96 * np.sqrt((Exp3G_hist[k, 1]/n_runs - (Exp3G_hist[k, 0]/n_runs)**2)/n_runs)
    print "UCB_N yields", UCB_N_hist[k, 0]/n_runs,"+/-", 1.96 * np.sqrt((UCB_N_hist[k, 1]/n_runs - (UCB_N_hist[k, 0]/n_runs)**2)/n_runs)
    print "UCB_MaxN yields", UCB_MaxN_hist[k, 0]/n_runs,"+/-", 1.96 * np.sqrt((UCB_MaxN_hist[k, 1]/n_runs - (UCB_MaxN_hist[k, 0]/n_runs)**2)/n_runs)
    print "Thompson_N yields", Thompson_N_hist[k, 0]/n_runs,"+/-", 1.96 * np.sqrt((Thompson_N_hist[k, 1]/n_runs - (Thompson_N_hist[k, 0]/n_runs)**2)/n_runs)
    print "Thompson_MaxN yields", Thompson_MaxN_hist[k, 0]/n_runs,"+/-", 1.96 * np.sqrt((Thompson_MaxN_hist[k, 1]/n_runs - (Thompson_MaxN_hist[k, 0]/n_runs)**2)/n_runs)
    print "Generalized_UCB yields", Generalized_UCB_hist[k, 0]/n_runs,"+/-", 1.96 * np.sqrt((Generalized_UCB_hist[k, 1]/n_runs - (Generalized_UCB_hist[k, 0]/n_runs)**2)/n_runs)
    print "Generalized_UCB_inv yields", Generalized_UCB_inv_hist[k, 0]/n_runs,"+/-", 1.96 * np.sqrt((Generalized_UCB_inv_hist[k, 1]/n_runs - (Generalized_UCB_inv_hist[k, 0]/n_runs)**2)/n_runs)
    print "Generalized_UCB_exp yields", Generalized_UCB_exp_hist[k, 0]/n_runs,"+/-", 1.96 * np.sqrt((Generalized_UCB_exp_hist[k, 1]/n_runs - (Generalized_UCB_exp_hist[k, 0]/n_runs)**2)/n_runs)