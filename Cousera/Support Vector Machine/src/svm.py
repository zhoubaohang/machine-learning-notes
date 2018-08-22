# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 20:20:39 2018

@author: 周宝航
"""

import numpy as np

KERNEL_GAUSSIAN = 'gausian'
KERNEL_LINEAR = 'linear'

class SVM(object):
    
    def __init__(self, kernel, C, toler, iter_num, sigma=1.0):
        # 核函数类型
        self.kernel = kernel
        # 松弛变量
        self.C = C
        # 容忍度
        self.toler = toler
        # 高斯核函数中的方差
        self.sigma = sigma
        # 迭代次数
        self.iter_num = iter_num
        
    def kernelFunc(self, x, xi):
        
        if self.kernel == KERNEL_GAUSSIAN:
            res = np.exp(- np.power(np.linalg.norm(x - xi), 2) \
                         / (2 * np.power(self.sigma, 2)))
        elif self.kernel == KERNEL_LINEAR:
            res = xi.T.dot(x)
            assert res.shape == (1,1), "xi and x have wrong shape, please check"
            res = res[0,0]

        return res
    
    def kernelTrans(self, X, x):
        
        n, m = X.shape
        
        assert x.shape == (n, 1), "x shape error, should be {0}, but {1}".format((n,1), x.shape)
        
        K = np.zeros((1, m))
        
        for i in range(m):
            K[0, i] = self.kernelFunc(x, X[:, i].reshape([-1, 1]))
        
        return K
    
    def clipAlpha(self, alpha, L, H):
        
        res = alpha
        
        if alpha > H:
            res = H
        elif alpha < L:
            res = L
        
        return res
    
    def select_J(self, i, m):
        
        l = list(range(m))
        l = l[:i] + l[i+1:]
        
        return np.random.choice(l)
    
    def fx(self, k, alpha, y, b):
        
        m = alpha.shape[1]
        assert alpha.shape == y.shape, "data shape error, please check"
        
        w = alpha * y
        
        assert w.shape == (1, m), "w shape error, please check"
        
        predY = w.dot(k) + b
        
        return predY
    
    def predict(self, X):
        
        n, m = X.shape
        n_, m_ = self.X.shape
        
        K = np.zeros((m_, m))
        for i in range(m):
            K[:, i] = self.kernelTrans(self.X, X[:, i].reshape([-1, 1]))
        
        wx = self.fx(K, self.alpha, self.y, self.b)
        
        predict = np.sign(wx)
        
        return predict
        
    def fit(self, X, y):
        
        n, m = X.shape
        
        self.X = X
        self.y = y
        self.alpha = np.zeros((1, m))
        self.b = 0.0
        self.K = np.zeros((m, m))
        
        for i in range(m):
            self.K[:, i] = self.kernelTrans(X, X[:, i].reshape([-1, 1]))
        
        self.SMO(m, y)
        self.w = (y * X).dot(self.alpha.T)
    
    def SMO(self, m, y):
        cnt = 0
        while cnt < self.iter_num:
            pair_changed = 0
            for i in range(m):
                a_i, x_i, y_i = self.alpha[0, i], self.K[:, i].reshape([-1, 1]), y[0, i]
                fx_i = self.fx(x_i, self.alpha, y, self.b)[0, 0]
                E_i = fx_i - y_i
                if ((y_i * E_i < -self.toler) and (a_i < self.C)) or ((y_i * E_i > self.toler) and (a_i > 0)):
                    j = self.select_J(i, m)
                    a_j, x_j, y_j = self.alpha[0, j], self.K[:, j].reshape([-1,1]), y[0, j]
                    fx_j = self.fx(x_j, self.alpha, y, self.b)[0, 0]
                    E_j = fx_j - y_j
                    k_ii, k_jj, k_ij = self.K[i, i], self.K[j, j], self.K[i, j]
                    eta = k_ii + k_jj - 2*k_ij
                    if (eta <= 0):
                        print("WARNING eta {0}".format(eta))
                        continue
                    a_i_old, a_j_old = a_i, a_j
                    a_j_new = a_j_old + y_j*(E_i - E_j)/eta
                    if y_i != y_j:
                        L = max(0, a_j_old - a_i_old)
                        H = min(self.C, self.C + a_j_old - a_i_old)
                    else:
                        L = max(0, a_i_old + a_j_old - self.C)
                        H = min(self.C, a_j_old + a_i_old)
                    if (L == H):
                        continue
                    a_j_new = self.clipAlpha(a_j_new, L, H)
                    a_i_new = a_i_old + y_i*y_j*(a_j_old - a_j_new)
                    if abs(a_j_new - a_j_old) < 0.00001:
                        continue
                    self.alpha[0, i], self.alpha[0, j] = a_i_new, a_j_new
                    b_i = self.b - E_i - y_i*k_ii*(a_i_new - a_i_old) - y_j*k_ij*(a_j_new - a_j_old)
                    b_j = self.b - E_j - y_i*k_ij*(a_i_new - a_i_old) - y_j*k_jj*(a_j_new - a_j_old)
                    if 0 < a_i_new < self.C:
                        self.b = b_i
                    elif 0 < a_j_new < self.C:
                        self.b = b_j
                    else:
                        self.b = (b_i + b_j)/2
                    pair_changed += 1
                    print('INFO   iteration:{}  i:{}  pair_changed:{}'.format(cnt, i, pair_changed))
            if pair_changed == 0:
                cnt += 1
            else:
                cnt = 0
            print('iteration number: {}'.format(cnt))