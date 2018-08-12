# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 20:41:36 2018

@author: 周宝航
"""

import numpy as np
import numpy.linalg as la

class PCA(object):
    
    def __init__(self):
        pass
    
    def featureNormalize(self, X):
        m, n = X.shape
        X_norm = X;
        mu = np.zeros((1, n))
        sigma = np.zeros((1, n))
        for i in range(n):
            mu[0, i] = np.mean(X[:,i])
            sigma[0, i] = np.std(X[:,i])
        X_norm  = (X - mu) / sigma
        return X_norm,mu,sigma
    
    def pca(self, X):
        m, n = X.shape
        X_norm = X.T.dot(X) / m
        U,S,_ = la.svd(X_norm)
        return U,S
    
    def projectData(self, X, U, K):
        U_reduce = U[:, :K]
        Z = X.dot(U_reduce)
        return Z
    
    def recoverData(self, Z, U, K):
        U_reduce = U[:, :K]
        X = Z.dot(U_reduce.T)
        return X