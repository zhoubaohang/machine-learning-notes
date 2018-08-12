# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 11:02:43 2018

@author: 周宝航
"""

from anomaly_detect import AnomalyDetector
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np

# ================== Load Example Dataset  ===================
data = sio.loadmat('data\\ex8data1.mat')

X = data['X']

fig = plt.figure()
ax = fig.add_subplot(1,1,1)

ax.plot(X[:, 0], X[:, 1], 'bx')
ax.axis([0, 30, 0, 30])
ax.set_xlabel('Latency (ms)')
ax.set_ylabel('Throughput (mb/s)')

# ================== Estimate the dataset statistics ===================
def visualizeFit(anomalyDetector, X, mu, sigma2):
    rg = np.arange(0,35.5,0.5)
    X1, X2 = np.meshgrid(rg, rg)
    Z = anomalyDetector.multivariateGaussian(np.c_[X1.reshape([-1, 1]), X2.reshape([-1, 1])], mu, sigma2)
    Z = Z.reshape(X1.shape)
    
    plt.contour(X1, X2, Z, 10**np.arange(-20,0,3,dtype=float).T)

anomalyDetector = AnomalyDetector()

mu, sigma2 = anomalyDetector.estimateGaussian(X)

p = anomalyDetector.multivariateGaussian(X, mu, sigma2)

visualizeFit(anomalyDetector, X, mu, sigma2)

# ================== Find Outliers ===================
Xval = data['Xval']
yval = data['yval']

pval = anomalyDetector.multivariateGaussian(Xval, mu, sigma2)

epsilon, F1 = anomalyDetector.selectThreshold(yval, pval)

outliers = np.where(p < epsilon)[0]

ax.scatter(X[outliers, 0], X[outliers, 1], marker='o', color='', edgecolors='r', linewidth=2)
plt.show()
# ================== Multidimensional Outliers ===================
data = sio.loadmat('data\\ex8data2.mat')
X = data['X']
Xval = data['Xval']
yval = data['yval']

mu, sigma2 = anomalyDetector.estimateGaussian(X)

p = anomalyDetector.multivariateGaussian(X, mu, sigma2)

pval = anomalyDetector.multivariateGaussian(Xval, mu, sigma2)

epsilon, F1 = anomalyDetector.selectThreshold(yval, pval)

print('Best epsilon found using cross-validation: %e\n' % epsilon);
print('Best F1 on Cross Validation Set:  %f\n' % F1);
print('   (you should see a value epsilon of about 1.38e-18)\n');
print('   (you should see a Best F1 value of 0.615385)\n');
print('# Outliers found: %d\n\n' % np.sum(p < epsilon));