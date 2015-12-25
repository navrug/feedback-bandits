# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 16:56:07 2015

@author: Gurvan
"""
import numpy as np

import bandit

class FeedbackGraph(object):
    """ General feedback graph.
    The underlying directed graph is specified by the adjacency matrix.
    """
    # TODO: Extend to graphs with observation probabilities
    def __init__(self, adjacency, arms=None, np_rng=None):
        if np_rng is None:
            self.np_rng = np.random.RandomState(0)
        else:
            self.np_rng = np_rng
        self.n_arms = len(adjacency)
        self.adj = adjacency
        if arms is None:
            self.arms = []
            for i in range(self.n_arms):
                self.arms.append(bandit.BernoulliArm(np_rng=self.np_rng))
        else:
            self.arms = arms

    """ Draw from an arm.
    Returns the reward for regret computations and the feedback according to
    the given graph. 
    The feedback is a list of size n_arms with None when there is no feedback.
    """
    def draw(self, index):
        if index >= self.n_arms:
            raise IndexError
        reward = self.arms[index].draw()
        feedback = [self.arms[j].draw() if self.adj[index][j] else None \
                        for j in np.array(range(self.n_arms))]
        if self.adj[index][index]:
            feedback[index] = reward
        return reward, feedback
        
    def best_mean(self):
        result = -np.inf
        for arm in self.arms:
            if arm.get_mean() > result:
                result = arm.get_mean()
        return result

    def in_neighbors(self, index):
        return self.adj[:,index] == True

    def out_neighbors(self, index):
        return self.adj[index,:] == True