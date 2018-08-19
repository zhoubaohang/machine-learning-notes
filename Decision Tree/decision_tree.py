# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 12:17:27 2018

@author: 周宝航
"""

from collections import Counter
import numpy as np
from graphviz import Digraph
import copy


GINI_INDEX = 'gini'

INFOENT = 'info'

class DecisionTree(object):
    
    def __init__(self, features, index=INFOENT):
        # features list
        self.features = features
        # information index
        self.index = index
        
    def calcGini(self, data):
        
        assert type(data) == list, 'data type error, data should be list'
        
        m = len(data)
        gini = 1.0
        cnt = Counter(str(i) for i in data)
        
        for k, v in cnt.items():
            gini -= np.power(v / m, 2)
    
        return gini
    
    def calcGiniIndex(self, X, y):
        
        D = len(X)
        cntX = Counter(str(i) for i in X.flatten())
        y = y.flatten()
        
        gini_index = 0.0
        
        for xfet, xfreq in cntX.items():
             y_fet = y[np.where(X == xfet)[0]]
             Dv = len(y_fet)
             gini_index += Dv / D * self.calcGini(y_fet.tolist())
        
        return gini_index
    
    def calcInfoEnt(self, data):
        
        assert type(data) == list, 'data type error, data should be list'
        
        m = len(data)
        ent = 0.0
        cnt = Counter(str(i) for i in data)
        
        for k, v in cnt.items():
            ent -= (v / m) * np.log2(v / m)
    
        return ent
    
    def calcInfoGain(self, X, y):
        
        D = len(X)
        cntX = Counter(str(i) for i in X.flatten())
        y = y.flatten()
        ent = self.calcInfoEnt(y.tolist())
        
        for xfet, xfeq in cntX.items():
            y_fet = y[np.where(X == xfet)[0]]
            Dv = len(y_fet)
            ent -= Dv / D * self.calcInfoEnt(y_fet.tolist())
    
        return ent
    
    def calcInfoGainRatio(self, X, y):
        
        D = len(X)
        cntX = Counter(str(i) for i in X.flatten())
        y = y.flatten()
        ent = self.calcInfoEnt(y.tolist())
        IV = 0.0
        
        for xfet, xfeq in cntX.items():
            y_fet = y[np.where(X == xfet)[0]]
            Dv = len(y_fet)
            ent -= Dv / D * self.calcInfoEnt(y_fet.tolist())
            IV -= Dv / D * np.log2(Dv / D)
    
        return ent / IV
            
    def chooseBestFeature(self, X, y, features):
        
        if self.index == GINI_INDEX:
            min_gini = 100
            fet = ''
            for i, feature in enumerate(features):
                gini = self.calcGiniIndex(X[:, i], y)
                if gini < min_gini:
                    min_gini = gini
                    fet = feature[0]
            return fet, min_gini
        elif self.index == INFOENT:
            max_Ent = 0.0
            fet = ''
            
            for i, feature in enumerate(features):
                ent = self.calcInfoGain(X[:, i], y)
                if ent > max_Ent:
                    max_Ent = ent
                    fet = feature[0]
            return fet, max_Ent
            
    def sameDataOnFet(self, X, y, sub_features):
        
        flag = True
        for i in range(len(sub_features)):
            if len(np.unique(X[:, i])) != 1:
                flag = False
                break
        return flag
    
    def majorityClass(self, y):
        feature = ''
        flag = 0
        cnt = Counter(str(i) for i in y)
        for k, v in cnt.items():
            if v > flag:
                feature = k
                flag = v
        return feature
    
    def predict(self, X):
        m = len(X)
        y = []
        for i in range(m):
            x = X[i, :]
            node = self.root
            while type(node) != str:
                feature = list(node.keys())[0]
                node = node[feature]
                fetIndex = self.features.index(feature)
                for k in node.keys():
                    if k == x[fetIndex]:
                        node = node[k]
                        break
            y.append(node)
        
        return y
    
    def fit(self, X, y):
        self.root = self.generateTree(X, y, self.features)
    
    def generateTree(self, X, y, features):
    
        cntY = np.unique(y)
        
        if len(cntY) == 1:
            node = str(cntY[0])
            return node
        
        if len(features) == 0 or self.sameDataOnFet(X, y, features):
            node = self.majorityClass(y)
            return node
        
        fet, ent = self.chooseBestFeature(X, y, features)
        
        node = None
        
        if fet == '':
            node = self.majorityClass(y)
        else:
            fetIndex = [features[i][0] for i in range(len(features))].index(fet)
            
            fetValues = features[fetIndex][1]
            
            node = {fet:{}}
            for val in fetValues:
                index = np.where(X[:,fetIndex] == val)[0]
                subX = X[index, :]
                subY = y[index]
                if len(index) == 0:
                    node[fet][val] = self.majorityClass(y)
                else:
                    subX = np.c_[subX[:, :fetIndex], subX[:, fetIndex+1:]]
                    subFeatures = copy.deepcopy(features)
                    subFeatures.pop(fetIndex)
                    node[fet][val] = self.generateTree(subX, subY, subFeatures)
            
        return node
        
    def plotTree(self):
    
        cache = []
        
        dot = Digraph("Tree", node_attr={"fontname":"Microsoft YaHei"},\
                      edge_attr={"fontname":"Microsoft YaHei", "arrowhead":"none"})
        
        cnt = 0
        cache.append([cnt, self.root])
        while len(cache):
            index, node = cache.pop(0)
            if type(node) == dict:
                feature = list(node.keys())[0]
                dot.node(str(index), feature)
                for value in node[feature].keys():
                    cnt += 1
                    dot.node(str(cnt))
                    dot.edge(str(index), str(cnt), value)
                    cache.append([cnt, node[feature][value]])
            elif type(node) == str:
                dot.node(str(index), node)            
        
        return dot
