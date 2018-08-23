# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 16:09:44 2018

@author: 周宝航
"""

import numpy as np
from collections import Counter

class kNN(object):
    
    def __init__(self):
        pass
    
    def fit(self, X, y, k):
        self.K = k
        self.X = X
        self.y = y
        
    def predict(self, X):
        m_ = self.X.shape[0]
        m, n = X.shape
        
        y = np.zeros((m, 1))
        
        for i in range(m):
            dist = np.zeros((m_))
            for j in range(m_):
                dist[j] = np.linalg.norm((X[i, :] - self.X[j, :]))
            sorted_dist = np.argsort(dist)[:self.K]
            count = Counter([i[0] for i in self.y[sorted_dist]])
            item, cnt = -1, -1
            for k, v in count.items():
                if v > cnt:
                    item = k
                    cnt = v
            y[i] = item
        
        return y