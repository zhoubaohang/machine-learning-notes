# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 10:05:54 2018

@author: 周宝航
"""

from k_means import KMeans
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

# =================== K-Means Clustering ======================
data = sio.loadmat('data\\ex7data2.mat')

K = 3
num_iters = 10
X = data['X']
initial_centroids = np.matrix([[3,3],
                               [6,2],
                               [8,5]])

kmeans = KMeans(K, num_iters)

idx = kmeans.findClosestCentroids(X, initial_centroids)

kmeans.train_model(X, initial_centroids, True)

# ============= K-Means Clustering on Pixels ===============
data = sio.loadmat('data\\bird_small.mat')

A = data['A']

A = A / 255
m, n, _ = A.shape
X = A.reshape([-1, 3])

K = 16
num_iters = 10

kmeans = KMeans(K, num_iters)
initial_centroids = kmeans.initCentroids(X)
centroids, _ = kmeans.train_model(X, initial_centroids)

# ================= Image Compression ======================
idx = kmeans.findClosestCentroids(X, centroids)

X_recoverd = centroids[idx, :]

X_recoverd = X_recoverd.reshape([m, n, 3])

fig = plt.figure()

original = fig.add_subplot(1,2,1)
original.set_title('Original')
original.imshow(A)

compressed = fig.add_subplot(1,2,2)
compressed.set_title('Compressed, with {0} colors.'.format(K))
compressed.imshow(X_recoverd)

plt.show()