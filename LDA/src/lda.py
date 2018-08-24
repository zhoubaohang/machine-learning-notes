# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 09:28:27 2018

@author: 周宝航
"""

import numpy as np

class LDA(object):
    
    def __init__(self, K):
        
        self.K = K
        
    def reduceDimen(self, X):
        
        return X.dot(self.w)
    
    def fit(self, X, y):
        
        self.w = self.lda(X, y)
        
    def lda(self, X, y):
        
        m,n = X.shape
        
        mu_t = np.mean(X, axis=0)
        
        classes = np.unique(y)
        
        # 全局散度矩阵
        St = (X - mu_t).T.dot(X - mu_t)
        
        # 类内散度矩阵
        Sw = np.zeros((n, n))
        
        for c in classes:
            index = np.where(y == c)[0]
            subData = X[index, :]
            mu = np.mean(subData, axis=0)
            Sw += (subData - mu).T.dot(subData - mu)
        
        Sb = St - Sw
        
        eigVals, eigVecs = np.linalg.eig(np.linalg.inv(Sw).dot(Sb))
        
        sorted_index = np.argsort(eigVals)
        
        eigenVectors = eigVecs[:, sorted_index[:-self.K-1:-1]]
        
        return eigenVectors
