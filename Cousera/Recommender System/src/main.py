# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 19:41:20 2018

@author: 周宝航
"""

from recommender import Recommender
import scipy.io as sio
import matplotlib.pyplot as plt

# =============== Loading movie ratings dataset ================
data = sio.loadmat('data\\ex8_movies.mat')
Y = data['Y']

fig = plt.figure()
ax = fig.add_subplot(1,1,1)

ax.imshow(Y)
ax.set_xlabel('Users')
ax.set_ylabel('Movies')










