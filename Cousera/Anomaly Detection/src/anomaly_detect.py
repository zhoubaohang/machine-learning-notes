# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 09:59:30 2018

@author: 周宝航
"""

import numpy as np
import numpy.linalg as la


class AnomalyDetector(object):
    
    def __init__(self):
        pass
    
    def estimateGaussian(self, X):
        m, n = X.shape
        mu = np.zeros([n, 1])
        sigma2 = np.zeros([n, 1])
        for i in range(n):
            mu[i] = np.mean(X[:, i])
            sigma2[i] = np.mean((X[:, i] - mu[i])**2)
        return mu,sigma2
    
    def multivariateGaussian(self, X, mu, sigma2):
        k = len(mu)
        sigma2 = np.diag(sigma2.flatten())
        X = X - mu.T
        p = (2 * np.pi) ** (- k / 2) * la.det(sigma2) ** (- 0.5) * \
            np.exp(-0.5 * np.sum(X.dot(la.pinv(sigma2)) * X, 1))
        return p.reshape([-1, 1])
    
    def selectThreshold(self, yval, pval):
        bestEpsilon = 0
        bestF1 = 0
        F1 = 0
        maxPval, minPval = max(pval), min(pval)
        stepsize = (maxPval - minPval) / 1000
        for epsilon in np.arange(minPval, maxPval, stepsize):
            predictions = pval < epsilon
            tp = np.sum((predictions == 1) & (yval == 1))
            fn = np.sum((predictions == 0) & (yval == 1))
            fp = np.sum((predictions == 1) & (yval == 0))
            prec = tp / (tp + fp)
            rec = tp / (tp + fn)
            F1 = 2 * prec * rec / (prec + rec)
            if F1 > bestF1:
                bestF1 = F1
                bestEpsilon = epsilon
        return bestEpsilon,bestF1