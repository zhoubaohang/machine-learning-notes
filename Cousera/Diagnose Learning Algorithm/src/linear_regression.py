# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import logging

class LinearRegression(object):
    
    def __init__(self, num_iters=None, alpha=None, lamb_da=None, num_features=None):
        # iteration numbers
        self.num_iters = num_iters if num_iters else 1500
        # learning rate
        self.alpha = alpha if alpha else 0.01
        # regulation
        self.lamb_da = lamb_da if lamb_da!=None else 0.01
        # feature numbers
        self.num_features = num_features if num_features else 1
        # parameters
        self.theta = np.zeros([self.num_features+1,1])
        # training datas
        self.data = None
        # mu
        self.mu = None
        # sigma
        self.sigma = None
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
                
    def featureNormalize(self, X):
        X_norm = X;
        mu = np.zeros((1, X.shape[1]))
        sigma = np.zeros((1, X.shape[1]))
        for i in range(X.shape[1]):
            mu[0, i] = np.mean(X[:,i])
            sigma[0, i] = np.std(X[:,i])
        X_norm  = (X - mu) / sigma
        return X_norm,mu,sigma

    def computeCost(self, X, y, theta, lamb_da):
        m = len(y)
        J = 0
        J = (np.sum((X.dot(theta) - y) ** 2) + lamb_da * np.sum(theta**2)) / (2 * m)
        return J
    
    def computeError(self, X, y):
        return np.sum((X.dot(self.theta) - y) ** 2) / (2 * len(y))
    
    def gradientDescent(self, X, y):
        m = len(y)
        J_history = []
        for i in range(self.num_iters):
            regulation = self.theta * self.lamb_da
            regulation[0] = 0
            self.theta = self.theta - self.alpha / m * (X.T.dot(X.dot(self.theta) - y) + regulation)
            J = self.computeCost(X, y, self.theta, self.lamb_da)
            J_history.append(J)
        return J_history
    
    def plot_model(self, x, y, X, J_history):
        if self.num_features == 1:
            fig = plt.figure(2)
            ax_loss = fig.add_subplot(1,2,2)
            ax_model = fig.add_subplot(1,2,1)
            ax_model.set_title('Linear regression')
            ax_model.scatter(x, y)
            ax_model.plot(x, X.dot(self.theta), color='r')
        else:
            ax_loss = plt.figure()
            ax_loss = ax_loss.add_subplot(1,1,1)
            ax_model = Axes3D(plt.figure())
            ax_model.set_title('Linear regression')
            ax_model.scatter(x[:,0], x[:,1], y, color='r')
            X = np.arange(min(x[:,0]), max(x[:,0]), 1)
            Y = np.arange(min(x[:,1]), max(x[:,1]), 1)
            X, Y = np.meshgrid(X, Y)
            Z = self.theta[0] + self.theta[1] * X + self.theta[2] * Y
            ax_model.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
        
        ax_loss.set_title('Loss')
        ax_loss.set_xlabel('Iteration')
        ax_loss.set_ylabel('Loss')
        ax_loss.set_xlim(-0.5,self.num_iters)
        ax_loss.plot(J_history)
        plt.show()
    
    def train_model(self, file_path=None):
        self.read_data(file_path)
        
        self.logger.info("getting the feature values")
        x = self.data[:,:self.num_features].reshape([-1, self.num_features])
        x_norm, self.mu, self.sigma = self.featureNormalize(x)
        self.logger.info("getting the object values")
        y = self.data[:,-1].reshape([-1, 1])
        # generate the feature matrix
        X = np.c_[np.ones([len(x), 1]), x_norm]
        self.logger.info("start gradient descent")
        
        J_history = []
        # learning curves
        train_errors = []
        cv_errors = []
        bach_size = 10
        m_sizes = [i for i in range(bach_size, len(y), bach_size)]
        for sizes in m_sizes:
            train_x, train_y, cv_x, cv_y = X[:sizes,:],y[:sizes],X[sizes:,:],y[sizes:]
            self.gradientDescent(train_x, train_y)
            train_errors.append(self.computeError(train_x, train_y))
            cv_errors.append(self.computeError(cv_x, cv_y))
            self.theta = np.zeros([self.num_features+1,1])
        J_history += self.gradientDescent(X, y)
        
        self.logger.info("end")
        
        fig = plt.figure(1)
        ax = fig.add_subplot(1,1,1)
        ax.set_title("learning curves")
        ax.set_xlabel("m (training set size)")
        ax.set_ylabel("error")
        ax.plot(m_sizes, train_errors, color='red', label='train error')
        ax.plot(m_sizes, cv_errors, color='blue', label='cross validation error')
        ax.legend()
        self.plot_model(x_norm, y, X, J_history)
    