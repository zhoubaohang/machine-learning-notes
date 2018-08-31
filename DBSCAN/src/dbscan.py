# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 14:08:24 2018

@author: 周宝航
"""

import numpy as np

class DBSCAN(object):
    
    def __init__(self, epsilon, MinPts):
        
        self.epsilon = epsilon
        self.MinPts = MinPts
    
    
    def nearestNeighbor(self, i, X):
        
        m, n = X.shape
        idxs = []
        
        for j in range(m):
            if i != j:
                dis = np.linalg.norm(X[i, :] - X[j, :])
                if dis <= self.epsilon:
                    idxs.append(j)
        
        return idxs
    
    def initCoreSet(self, X):
        
        m, n = X.shape
        Omega = set()
        
        for i in range(m):
            idxs = self.nearestNeighbor(i, X)            
            if len(idxs) >= self.MinPts:
                Omega.add(i)
                
        return Omega
    
    def fit_label_(self):
        
        return self.y
    
    def fit(self, X):
        m, n = X.shape
        self.X = X
        # 初始化核心对象集合
        Omega = self.initCoreSet(X)
        # 初始化未访问样本集合
        D = set([i for i in range(m)])
        
        C = self.dbscan(D, Omega)
        
        self.y = np.zeros((m, ))
        for i, idxs in enumerate(C):
            self.y[list(idxs)] = i
        
    def dbscan(self, D, Omega):
        
        k = 0
        C = []
        
        while len(Omega) != 0:
            D_old = D.copy()
            
            o = np.random.choice(list(Omega))
            Q = [o]
            D.remove(o)
            
            while len(Q) != 0:
                
                q = Q.pop(0)
                idxs = self.nearestNeighbor(q, self.X)
                
                if len(idxs) >= self.MinPts:
                    delta = set(idxs) & D
                    Q += list(delta)
                    D -= delta
            
            k += 1
            Ck = D_old - D
            Omega -= Ck
            
            C.append(Ck)
        
        return C