# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import logging

class LinearRegression(object):
    
    def __init__(self, num_iters=None, alpha=None, num_params=2):
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
            self.data = np.genfromtxt(file_path, delimiter=',', dtype=None)
            
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

    def computeCost(self, X, y, theta):
        m = len(y)
        J = 0
        J = np.sum((X.dot(theta) - y) ** 2) / (2 * m)
        return J
    
    def gradientDescent(self, X, y):
        m = len(y)
        for i in range(self.num_iters):
            self.theta = self.theta - self.alpha / m * X.T.dot(X.dot(self.theta) - y)
            J = self.computeCost(X, y, self.theta)
            yield J
    
    def train_model(self, file_path=None):
        self.read_data(file_path)
        self.logger.info("getting the feature values")
        x = self.data[:,0].reshape([-1, 1])
        self.logger.info("getting the object values")
        y = self.data[:,1].reshape([-1, 1])
        # generate the feature matrix
        X = np.c_[np.ones([len(x), 1]), x]
        self.logger.info("start gradient descent")
        fig = plt.figure()
        ax_model = fig.add_subplot(1,2,1)
        ax_model.scatter(x, y)
        ax_loss = fig.add_subplot(1,2,2)
        J_history = []
        for J in self.gradientDescent(X, y):
            J_history.append(J)
        
            if len(ax_model.lines) > 0:
                    ax_model.lines.pop()
            ax_model.set_title('Linear regression')
            ax_model.plot(x, X.dot(self.theta), color='r')
            
            ax_loss.set_title('Loss')
            ax_loss.set_xlabel('Iteration')
            ax_loss.set_ylabel('Loss')
            ax_loss.set_xlim(0,self.num_iters)
            ax_loss.plot(J_history)
            plt.pause(0.001)
        plt.show()
        self.logger.info("end")
    