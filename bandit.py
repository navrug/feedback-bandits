# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 11:51:21 2015

@author: Gurvan
"""

import numpy as np


class Arm(object):
    """ Abstract arm. """
    def __init__(self, np_rng=None):
        if np_rng is None:
            self.np_rng = np.random.RandomState(0)
        else:
            self.np_rng = np_rng
        
        
class BernoulliArm(Arm):
    """ Bernoulli arm. 
    The parameter is initialized at random if none is given.
    """
    def __init__(self, mean=None, np_rng=None):
        Arm.__init__(self, np_rng)
        if mean is None:
            self.mean = np_rng.rand()
        else:
            self.mean = mean

    def draw(self):
        return self.np_rng.binomial(1,self.mean)
        

class FiniteArm(Arm):
    """ Finite arm. 
    The caller must specify the possible values and the relative weights.
    Uniform weights if none is specified.
    """
    def __init__(self, values, weights=None, np_rng=None):
        Arm.__init__(self, np_rng)
        self.values = values
        if weights is None:
            self.probabilities = np.ones(len(values))/len(values)
        else:
            self.probabilities = weights/np.sum(weights)
        
    def draw(self):
        r = self.np_rng.rand()
        cmf = 0
        for i in range(len(self.probabilities)):        
            cmf += self.probabilities[i]
            if r < cmf:
                return self.values[i]


class ExpArm(Arm):
    """ Exponential law arm. 
    The parameter is initialized at random if none is given.
    """
    def __init__(self, theta, np_rng=None):
        Arm.__init__(self, np_rng)
        if theta is None:
            self.theta = 10*np_rng.rand()
        else:
            self.theta = theta

    def draw(self):
        return self.np_rng.exponential(self.theta)


class BetaArm(Arm):
    """ Beta law arm. 
    The parameter is initialized at random if none is given.
    """
    def __init__(self, a=1, b=1, np_rng=None):
        Arm.__init__(self, np_rng)
        self.a = a
        self.b = b

    def draw(self):
        return self.np_rng.beta(self.a,self.b)
