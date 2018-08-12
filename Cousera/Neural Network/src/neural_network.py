# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 21:02:28 2018

@author: 周宝航
"""

import numpy as np
import logging
import matplotlib.pyplot as plt

class NeuralNetwork(object):
    
    def __init__(self, sizes, num_iters=None, alpha=None, lam_bda=None):
        # layers numbers
        self.num_layers = len(sizes)
        # parameter sizes
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        # bias sizes
        self.bias = [np.zeros((y, 1)) for y in sizes[1:]]
        # iteration numbers
        self.num_iters = num_iters if num_iters else 6000
        # learning rate
        self.alpha = alpha if alpha else 1
        # regularization
        self.lam_bda = lam_bda if lam_bda!=None else 0
        # logger
        self.logger = logging.getLogger()
        logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
        logging.root.setLevel(level=logging.INFO)
        
    def __read_data(self, file_path=None):
        data = None
        if file_path:
            self.logger.info("loading train data from {0}".format(file_path))
            data = np.loadtxt(file_path)
        return data
        
    def __sigmoid(self, z, derive=False):
        if derive:
            return self.__sigmoid(z) * (1.0 - self.__sigmoid(z))
        else:
            return 1.0 / (1.0 + np.exp(-z))
    
    def save(self, file_path):
        if file_path:
            import pickle
            with open(file_path, 'wb') as f:
                pickle.dump(self.weights, f)
    
    def load(self, file_path):
        if file_path:
            import pickle
            with open(file_path, 'rb') as f:
                self.weights = pickle.load(f)
    
    def forwardprop(self, X):
        activation = X
        activations = [X]
        zs = []
        for w, b in zip(self.weights, self.bias):
            z = w.dot(activation) + b
            zs.append(z)
            activation = self.__sigmoid(z)
            activations.append(activation)
        return (activations, zs)
    
    def costFunction(self, y, _y):
        m = len(y)
        regularization = sum([np.sum(weight**2) for weight in self.weights])
        return - np.sum(y * np.log(_y) + (1.0 - y) * np.log(1.0 - _y)) / m + self.lam_bda / (2*m) * regularization

    def backprop(self, X, y):
        m = len(y)
        nable_w = [np.zeros(w.shape) for w in self.weights]
        nable_b = [np.zeros(b.shape) for b in self.bias]
        # forward propagation
        activations, zs = self.forwardprop(X)
        # cost
        # delta^(l) = a^(l) - y
        cost = activations[-1] - y
        # calc delta
        delta = cost * self.__sigmoid(zs[-1], derive=True)
        nable_b[-1] = delta
        nable_w[-1] = delta.dot(activations[-2].T)
        # back propagation
        for l in range(2, self.num_layers):
            # delta^(l) = weights^(l)^T delta^(l+1)
            delta = self.weights[-l+1].T.dot(delta) * self.__sigmoid(zs[-l], derive=True)
            nable_b[-l] = delta
            nable_w[-l] = delta.dot(activations[-l-1].T)
        
        # update bias, weights
        self.bias = [b-self.alpha/m*delta_b for b, delta_b in zip(self.bias, nable_b)]
        self.weights = [(1-self.lam_bda*self.alpha/m)*w-self.alpha*delta_w for w, delta_w in zip(self.weights, nable_w)]
        return activations[-1]
    
    def train_model(self, file_path=None):
        train_data = self.__read_data(file_path)
        self.logger.info("getting feature values")
        X = train_data[:, :-1].T
        self.logger.info("getting object values")
        y = train_data[:, -1]
        J_history = []
        for i in range(self.num_iters):
            _y = self.backprop(X, y)
            cost = self.costFunction(y, _y)
            if i % 100 == 0:
                self.logger.info("epoch {0}/{1} cost : {2}".format(i, self.num_iters, cost))
            J_history.append(cost)
        fig = plt.figure()
        ax_loss = fig.add_subplot(1,1,1)
        ax_loss.set_title('Loss')
        ax_loss.set_xlabel('Iteration')
        ax_loss.set_ylabel('Loss')
        ax_loss.set_xlim(0,self.num_iters)
        ax_loss.plot(J_history)
        plt.show()
        
