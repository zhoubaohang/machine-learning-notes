# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 09:49:10 2018

@author: 周宝航
"""

import numpy as np

data = np.loadtxt('data.txt', delimiter=',', dtype=str)

X = data[1:, 1:7]
y = data[1:, -1]

features = []

for i, label in enumerate(data[0, 1:7].tolist()):
    features.append([label, np.unique(X[:, i]).tolist()])

#%%
from decision_tree import DecisionTree, GINI_INDEX, INFOENT

tree = DecisionTree(features, index=INFOENT)

tree.fit(X, y)

tree.plotTree()