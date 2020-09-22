#coding:utf-8  

'''
Created on 2020年1月20日
@author: zfg
'''

import numpy as np
import math as mt
from collections import defaultdict

class DecisionTree(object):
    def __init__(self):
        pass

#   计算概率
    def _getDistribution(self,dataArray):
        # defaultdict的作用是在于，当字典里的key不存在但被查找时，返回的不是keyError而是一个默认值，此处返回float类型的0
        # dict  map
        distribution = defaultdict(float)
        m, n = np.shape(dataArray)
        # 累加每一个分类的概率
        for line in dataArray:
            print(line[-1])
            distribution[line[-1]] += 1.0/m
        # 每一个分类号出现的概率  yes 5/14  no 9/14
        return distribution

#   计算信息熵
    def _entropy(self, dataArray):
        ent = 0.0
        distribution=self._getDistribution(dataArray)

        # H(x) = - sigma p * log p
        for key, prob in distribution.items():
            ent -= prob * mt.log(prob, 2)
        return ent

    def _conditionEntropy(self, dataArray, colIdx):
        valueCnt = defaultdict(int)
        m, n = np.shape(dataArray)
        # 条件熵
        condEnt = 0.0
        # 获得第 colInx 列的去重值
        uniqueValues = np.unique(dataArray[:, colIdx])
        # 求 colInx 列每个值的前提下的熵值
        for oneValue in uniqueValues:
            # 获得某一个值（前提）下的数据
            oneData = dataArray[dataArray[:, colIdx] == oneValue]
            # 求出当前前提(中间节点)下信息熵
            oneEnt = self._entropy(oneData)
            # 满足当前前提(已知条件)的概率
            prob = float(np.shape(oneData)[0]) / m
            # 概率*信息熵
            condEnt += prob * oneEnt
        return condEnt

    def _infoGain(self, dataArray, colIdx, baseEnt):
        # 计算第 colIdx 列的条件熵
        condEnt = self._conditionEntropy(dataArray, colIdx)
        # 信息增益
        return baseEnt-condEnt

    # 求出信息增益最大的一列
    def _chooseBestProp(self,dataArray):
        m, n = np.shape(dataArray)
        bestProp = -1
        bestInfoGain = 0
        # 计算分类号信息熵
        baseEnt = self._entropy(dataArray)
        # [0-4)
        for i in range(n-1):
            # 计算已知条件的信息增益
            infoGain=self._infoGain(dataArray, i, baseEnt)
            if infoGain > bestInfoGain:
                bestProp=i
                bestInfoGain=infoGain
        return bestProp

    def _splitData(self,dataArray,colIdx,splitValue):
        m, n = np.shape(dataArray)

        # 返回类型 array([ True,  True,  True, False,  True])
        cols = np.array(range(n)) != colIdx
        rows = (dataArray[:, colIdx] == splitValue)
        print(rows)

        # data=dataArray[rows,:][:,cols]  得到该spliValue分支下的子树冠
        #ix_  取rows中指定的行，取cols中指定的列   花式索引
        data = dataArray[np.ix_(rows, cols)]
        return data

    def createTree(self, dataArray):
        # 获取集合形状 (m,n)
        m, n = np.shape(dataArray)
        # 对于一维数组或者列表，unique函数去除其中重复的元素，并按元素由大到小返回一个新的无元素重复的元组或者列表
        # 此处判断分类结果是否单一，如若单一，说明此模型为单叶节点，直接返回结果
        if len(np.unique(dataArray[:, -1])) == 1:
            return (dataArray[0, -1], 1.0)
        # 此处判断数据集矩阵是否为2列，如果为真，则对结果分类的影响只有一列（因素），模型只有根节点，不构成树，则直接返回概率较大的叶节点即可
        if n == 2:
            distribution = self._getDistribution(dataArray)
            sortProb = sorted(distribution.items(), key=lambda x: x[1], reverse=True)
            return sortProb
        rootNode = {}
        # 选择分类条件的     信息增益  0
        bestPropIdx = self._chooseBestProp(dataArray)

        # 树
        rootNode[bestPropIdx] = {}
        uniqValues = np.unique(dataArray[:, bestPropIdx])
        # 根据第一列的数据来分类
        for oneValue in uniqValues:
            splitDataArray = self._splitData(dataArray, bestPropIdx, oneValue)
            # 要不要把分类出来的这堆数据  进行再次切割   信息熵判断一下
            rootNode[bestPropIdx][oneValue] = self.createTree(splitDataArray)
        return rootNode
    
def loadData():
    # 矩阵
    dataMat = []                 
    fr = open("decisiontree.txt")
#     readlines他会一次性将decisiontree.txt文件全部加载到内存的列表中
    lines = fr.readlines()
    for line in lines:
        curLine = line.strip().split('\t')
        dataMat.append(curLine)
    return dataMat


if __name__ == '__main__':
    data = loadData()
    dataarray = np.array(data)
    dt = DecisionTree()
    tree = dt.createTree(dataarray)
    print(tree)
