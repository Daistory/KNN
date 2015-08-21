#!/usr/bin/python
#encoding:utf-8
from numpy import *
from array import *
import operator
'''
K近邻算法
参数：
        intX：分类的输入向量
        dataSet：输入的样本训练集
        labels：标签向量
        K：选择的最近邻的数目
        返回值：得到的是一个分类labels里面具体的类别
'''
def classify0(intX,dataSet,labels,k):
    dataSetSize=dataSet.shape[0] #获得训练集的数目
    '''
    diffMat 得到给的判定点intX和样本训练集之间所有点的横纵坐标差
    tile函数的含义：例如：a =[1,2,3],b = tile(a,2)=[1,2,3,1,2,3];c = tile(a,(1,2)) = [[1,2,3,1,2,3]];d= tile(a,(2,1) = [[1,2,3],[1,2,3]]
    '''
    diffMat = tile(intX, (dataSetSize,1)) - dataSet
    sqDiffMat=diffMat**2 #获得横纵坐标的平方值
    sqDistances = sqDiffMat.sum( axis = 1 ) #获得横纵坐标的差的平方的和;
    distance = sqDistances ** 0.5
    sortedDistIndicies = distance .argsort() #将之前得到的距离排序，距离从小到大排序，返回的是列表中的数字所在的下标位置
    classCount={} #创建一个空的字典
    for i in range(k):  #获取得到前面K个点的数据
        voteilabel = labels [sortedDistIndicies[i]] #获取到对应的前k个点的对应的分类信息
        classCount[voteilabel] = classCount.get(voteilabel,0)+1 #将获取到的信息和对应的点  加入到字典,最后得到就是一个key是对应类别 value是读应出现次数的字典
    sortedClassCount = sorted(classCount.items(),key = operator.itemgetter(1),reverse = True)
    '''
    maxCount = 0
    for key ,value in classCount.items():
        if value > maxCount:
            maxCount = value
            maxIndex = key
            '''
    return sortedClassCount[0][0]
'''
归一化特征值函数（避免因特征值本身数据很大而y影响分类结果）
函数名：autoNormtile
参数：datasSet  需要处理的数据集
返回：得到三个list，其中normDataSet就是处理后的[0,1]的数据集
'''
def autoNorm(dataSet):
    minVals = dataSet.min(0)    #获得dataSet里面没一个List里面对应下标最小的元素组成的新的list，min（1）表示将每一个list里面对应最小的元素拿出来组成新的List，第一个list得到地第一个元素，，。。。。
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet)) #得到一个和dataSet 同行同列的全零矩阵
    m = dataSet.shape[0] #get a value of dataSet's  line
    normDataSet = dataSet - tile(minVals,(m,1))     #a =[1,2,3],b = tile(a,2)=[1,2,3,1,2,3];c = tile(a,(1,2)) = [[1,2,3,1,2,3]];d= tile(a,(2,1) = [[1,2,3],[1,2,3]],将dataSet里面没一个list都和得到最小的list相减
    normDataSet = normDataSet/tile(ranges, (m, 1) )
    return normDataSet 