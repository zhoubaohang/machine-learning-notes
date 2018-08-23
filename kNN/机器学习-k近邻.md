
# kNN(K邻近)

> 《机器学习》-周志华
> 《统计学习方法》-李航

- k近邻（k-Nearest Neighbour）学习是一种常用的监督学习方法。

## 算法流程

1. 给定测试样本，基于某种距离度量找出训练集中与其最靠近的k个训练样本，然后基于这k个邻居的信息来进行预测。
2. 通常，在分类任务中可使用“投票法”，即选择这k个样本中出现最多的类别标记作为预测结果。
3. 在回归任务中可使用“平均法”，即将这k个样本的实值输出标记的平均值作为预测结果。
4. 当然，还可以居于距离的远近进行加权平均或加权投票，距离越近的样本权重越大。

## 特点

- 我们可以发现，k近邻学习与其他如：感知机、决策树等方法有明显的不同之处：它没有显式的训练过程。
- 它其实是**懒惰学习（lazy learning）**的代表之一，此类学习方法在训练阶段仅是把样本保存起来，训练时间开销为0，待收到测试样本后在进行处理。
- 而与之对应的，那些在训练阶段就对样本进行学习处理的方法，称为：**急切学习（eager learning）**。

## 泛化错误率
- 给定测试样本$x$，若其最近邻样本为$z$,则最近邻分类器出错的概率就是$x$与$z$类别标记不同的概率，即：

$$P(err)=1-\sum_{c \in y}P(c | x)P(c|z)$$

- 假设样本独立同分布，且对任意$x$和任意小正数$\delta$，在$x$附近$\delta$距离范围内总能找到一个训练样本；换言之，对任意测试样本，总能在任意近的范围内找到上式中的训练样本$z$.
- 令 <a href="https://www.codecogs.com/eqnedit.php?latex=c^*&space;=&space;\arg&space;\max_{c&space;\in&space;y}P(c|x)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?c^*&space;=&space;\arg&space;\max_{c&space;\in&space;y}P(c|x)" title="c^* = \arg \max_{c \in y}P(c|x)" /></a> 表示贝叶斯最优分类器的结果，有：

<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{align}&space;P(err)&space;&&space;=&space;1-\sum_{c&space;\in&space;y}P(c|x)P(c|z)&space;\\&space;&&space;\simeq&space;1&space;-&space;\sum_{c&space;\in&space;y}P^2(c|x)&space;\\&space;&&space;\leq&space;1&space;-&space;P^2(c^*|x)&space;\\&space;&&space;=&space;(1&plus;P(c^*|x))(1-P(c^*|x))&space;\\&space;&&space;\leq&space;2&space;\times&space;(1&space;-&space;P(c^*|x))&space;\end{align}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{align}&space;P(err)&space;&&space;=&space;1-\sum_{c&space;\in&space;y}P(c|x)P(c|z)&space;\\&space;&&space;\simeq&space;1&space;-&space;\sum_{c&space;\in&space;y}P^2(c|x)&space;\\&space;&&space;\leq&space;1&space;-&space;P^2(c^*|x)&space;\\&space;&&space;=&space;(1&plus;P(c^*|x))(1-P(c^*|x))&space;\\&space;&&space;\leq&space;2&space;\times&space;(1&space;-&space;P(c^*|x))&space;\end{align}" title="\begin{align} P(err) & = 1-\sum_{c \in y}P(c|x)P(c|z) \\ & \simeq 1 - \sum_{c \in y}P^2(c|x) \\ & \leq 1 - P^2(c^*|z) \\ & = (1+P(c^*|z))(1-P(c^*|z)) \\ & \leq 2 \times (1 - P(c^*|x)) \end{align}" /></a>

- 最近邻分类器虽然简单，但其繁华错误率不超过贝叶斯最优分类器的错误率的两倍。

## 实验部分

- 使用 **Jupyter notebook**
- [Github地址](https://github.com/zhoubaohang/machine-learning-notes)

```python
# 导入相关库
from knn import kNN
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np

%matplotlib inline

# sklearn 库生成数据

#关键参数有n_samples（生成样本数）， n_features（样本特征数），centers(簇中心的个数或者自定义的簇中心)和cluster_std（簇数据方差，代表簇的聚合程度）
# X为样本特征，Y为样本簇类别， 共1000个样本，每个样本2个特征，共5个簇
X,y=make_blobs(n_samples=1000,n_features=2,centers=5)
y = y.reshape([-1, 1])

# 可视化数据
plt.scatter(X[:,0], X[:,1],c=y.flatten(),s=3,marker='o')
```




    <matplotlib.collections.PathCollection at 0x28f5c3dd780>




![png](output_1_1.png)



```python
# 分割训练集、测试集
# 700个训练样本
X_train = X[:700, :]
y_train = y[:700, :]

# 300个测试样本
X_test = X[700:, :]
y_test = y[700:, :]
```


```python
# 实例化 kNN 分类器对象
kNN = kNN()

# k : 近邻数
k = 5
# 调用训练方法
kNN.fit(X_train, y_train, k)
```


```python
# 对 测试集 进行分类
predict = kNN.predict(X_test)

#可视化测试结果
plt.scatter(X_train[:,0], X_train[:,1],c=y_train.flatten(),s=3,marker='o')
plt.scatter(X_test[:,0], X_test[:,1],c=predict.flatten(),s=3,marker='s')

# 统计正确率
print("acc : {0}".format(np.sum(y_test == predict) / len(y_test)))
```

    acc : 0.99
    


![png](output_4_1.png)

