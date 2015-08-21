#/usr/bin/python 
#encoding:utf-8
import operator
from numpy import *
'''
函数功能：
寻找把某个点当做最近邻所有点 
参数：intX，需要需找将它当做最近邻点的点  是一个Array
            dataArray，需要需找的所有范围
返回值：一个List包括输入点的所有的最近邻点 
'''
def serachMostNeiberPoint(intX,dataArray):
    dataArraySize=dataArray.shape[0]
    keyList= []
    a = len(dataArray)
    distanceList = zeros((a,a-1))
    index = 0
    for List in dataArray:
        diffMat = tile(List, (dataArraySize,1)) - dataArray
        sqDiffMat=diffMat**2 #获得横纵坐标的平方值
        sqDistances = sqDiffMat.sum( axis = 1 ) #获得横纵坐标的差的平方的和;
        distance = sqDistances ** 0.5
        sortedDistIndicies = distance.argsort()#将之前得到的距离排序，距离从小到大排序，返回的是列表中的数字所在的下标位置
        distanceList[index,:] = sortedDistIndicies[1:,]
        index = index + 1
    for i in range(a):
        intNUm = distanceList[i][0]
        if intX[0] ==dataArray[intNUm][0] and intX[1]==dataArray[intNUm][1] and intX[2]==dataArray[intNUm][2]:
            keyList.append(i)
    return keyList
'''
函数功能：就是将把彼此视为最近邻点的点聚类到一起，进行相同K处理
返回值：一个List里面是的每一个元素就是一对点，标识下标ij是彼此互为最近邻点的
参数：dataArray，需要寻找的范围
'''
def getNeiberP2P(dataArray):
    index = len(dataArray)
    newList = []
    neiberList = []
    myList = []
    for i in range(index):
        keyList = serachMostNeiberPoint(dataArray[i], dataArray)
        newList.append(keyList)
    for j in range(index):
        for number in newList[j]:
            for k in newList[number]:
                if k == j and j<number:
                    myList.append(j)
                    myList.append(number) 
                    neiberList.append(myList)
                    myList = []                   
    return neiberList

    