# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 13:46:29 2018

@author: 周宝航
"""

import numpy as np

class LVQ(object):
    
    def __init__(self, q=0):
        
        self.num_cluters = q
    
    def predict(self, X):
        
        m, n = X.shape
        
        y = np.zeros((m,))
        
        for i in range(m):
            
            dis = [np.linalg.norm(self.p[idx, :] - X[i, :]) for idx in range(self.num_cluters)]
            minDis_idx = np.where(dis == np.min(dis))[0][0]
            
            y[i] = self.t[minDis_idx]
        
        return y
    
    def evalute(self, X, y):
        
        y_pred = self.predict(X)
        
        return np.sum(y_pred == y) / len(y)
    
    def fit(self, X, y, num_iters, learning_rate):
        # 迭代次数
        self.num_iters = num_iters
        # 学习率
        self.learning_rate = learning_rate
        m,n = X.shape
        # 样本类别
        types = np.unique(y)
        
        if self.num_cluters == 0:
            self.num_cluters = len(types)
        
        # 原型向量
        self.p = np.zeros((self.num_cluters, n))
        # 向量类别标记
        self.t = [i for i in types]
        
        self.lvq(X, y)
        
    def init_vec(self, X, y):
        
        for i, t in enumerate(self.t):
            # 随机选择样本类别为 t 的样例作为原型向量初始值
            idx = np.random.choice(np.where(y == t)[0])
            self.p[i, :] = X[idx, :]
    
    def lvq(self, X, y):
        
        m, n = X.shape
        
        self.init_vec(X, y)
        
        for i in range(self.num_iters):
            
            idx = np.random.choice(m)
            dataVec = X[idx, :]
            
            dis = [np.linalg.norm(self.p[idx, :] - dataVec) for idx in range(self.num_cluters)]
            minDis_idx = np.where(dis == np.min(dis))[0][0]
            
            if y[idx] == self.t[minDis_idx]:
                
                self.p[minDis_idx, :] = self.p[minDis_idx, :] \
                                        + self.learning_rate * (X[idx, :] - self.p[minDis_idx, :])
                
            else:
                
                self.p[minDis_idx, :] = self.p[minDis_idx, :] \
                                        - self.learning_rate * (X[idx, :] - self.p[minDis_idx, :])
            if i % (self.num_iters/10) == 0:
                print("iter {0}/{1}".format(i, self.num_iters))