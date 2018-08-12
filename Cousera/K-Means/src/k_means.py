# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 09:46:20 2018

@author: 周宝航
"""

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

class KMeans(object):
    
    def __init__(self, K, num_itres):
        # cluster numbers
        self.K = K
        # iteration numbers
        self.num_itres = num_itres
    
    def initCentroids(self, X):
        m, n = X.shape
        rd = np.random.randint(0, m, self.K)
        centroids = X[rd, :]
        return centroids
        
    def findClosestCentroids(self, X, centroids):
        m, n = X.shape
        idx = np.zeros([m, 1])
        for i in range(m):
            dis = [la.norm(X[i,:] - centroids[j,:]) for j in range(self.K)]
            idx[i] = np.where(dis == np.min(dis))[0][0]
        return np.uint8(idx)
    
    def computeCentroids(self, X, idx):
        m, n = X.shape
        centroids = np.zeros([self.K, n])
        for i in range(self.K):
            index = (idx == i)
            counts = np.sum(index)
            centroids[i, :] = X.T.dot(index).T / counts
        return centroids
    
    def plotDataPoints(self, X, idx):
        self.ax.scatter(X[:, 0], X[:, 1], c = idx[:, 0])
    
    def plotProgresskMeans(self, X, centroids, previous, idx, no_i):
        self.plotDataPoints(X, idx)
        self.ax.scatter(centroids[:, 0].tolist(), centroids[:, 1].tolist(), c = 'r', marker = 'x')
        for i in range(self.K):
            data = np.r_[centroids[i, :].reshape([1,-1]), previous[i, :].reshape([1,-1])]
            self.ax.plot(data[:, 0], data[:, 1])
        self.ax.set_title('Iteration number {0}'.format(no_i))
    
    def train_model(self, X, centroids, plot_progress=False):
        m, n = X.shape
        self.centroids = centroids
        previous_centroids = self.centroids
        idx = np.zeros([m, 1])
        if plot_progress:
            fig = plt.figure()
            self.ax = fig.add_subplot(1,1,1)
        for i in range(self.num_itres):
            idx = self.findClosestCentroids(X, self.centroids)
            if plot_progress:
                self.plotProgresskMeans(X, self.centroids, previous_centroids, idx, i)
                previous_centroids = self.centroids
            self.centroids = self.computeCentroids(X, idx)
            plt.pause(0.01)
        if plot_progress:
            plt.show()
        return (self.centroids, idx)
        
