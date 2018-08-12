# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 16:52:11 2018

@author: 周宝航
"""

import numpy as np
import matplotlib.pyplot as plt
import logging

class LogisticRegression(object):
    
    def __init__(self, num_iters=None, alpha=None, num_params=3):
        # iteration numbers
        self.num_iters = num_iters if num_iters else 1500
        # learning rate
        self.alpha = alpha if alpha else 0.01
        # parameters
        self.theta = np.zeros([num_params,1])
        # training datas
        self.data = None
        # logger
        self.logger = logging.getLogger()
        logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
        logging.root.setLevel(level=logging.INFO)
        
    def read_data(self, file_path=None):
        if file_path:
            self.logger.info("reading the data from %s" % file_path)
            self.data = np.loadtxt(file_path)
            
    def save(self, path=None):
        if path:
            import pickle
            with open(path, "rb") as f:
                pickle.dump(self.theta, f)
                
    def load(self, path=None):
        if path:
            import pickle
            with open(path, "rb") as f:
                self.theta = pickle.load(f)
                
    def sigmoid(self, z):
        return 1 / (1 + np.exp(- z))
    
    def computeCost(self, X, y, theta):
        m = len(y)
        h = self.sigmoid(X.dot(theta))
        J = - (y.T.dot(np.log(h)) + (1.0 - y).T.dot(np.log(1.0 - h))) / m
        return np.sum(J)
    
    def gradientDescent(self, X, y):
        m = len(y)
        for i in range(self.num_iters):
            h = self.sigmoid(X.dot(self.theta))
            self.theta = self.theta - self.alpha / m * X.T.dot(h - y)
            J = self.computeCost(X, y, self.theta)
            yield J
    
    def train_model(self, file_path=None):
        self.read_data(file_path)
        self.logger.info("getting the feature values")
        x = self.data[:,:-1]
        self.logger.info("getting the object values")
        y = self.data[:,-1].reshape([-1, 1])
        # generate the feature matrix
        X = np.c_[np.ones([len(x), 1]), x]
        self.logger.info("start gradient descent")
        fig = plt.figure()
        ax_model = fig.add_subplot(1,2,1)
        for feature,tag in zip(x,y):
            color = 'or' if tag==0 else 'ob'
            ax_model.plot(feature[0], feature[1], color)
        ax_loss = fig.add_subplot(1,2,2)
        J_history = []
        for J in self.gradientDescent(X, y):
            J_history.append(J)
        ax_model.set_title('Logistic regression')
        ax_model.set_xlabel('feature 1')
        ax_model.set_ylabel('feature 2')
        tx = x[:,0]
        ty = (-self.theta[0, 0] - self.theta[1, 0] * tx) / self.theta[2, 0]
        ax_model.plot(tx, ty, color='r')
        
        ax_loss.set_title('Loss')
        ax_loss.set_xlabel('Iteration')
        ax_loss.set_ylabel('Loss')
        ax_loss.set_xlim(0,self.num_iters)
        ax_loss.plot(J_history)
        plt.show()
        self.logger.info("end")