#/usr/bin/python
#encoding:utf-8
import createListForData.changeDataToList
import Classify.knn
import numpy as np
from numpy import *
from array import   array
from createListForData.changeDataToList import classifyDataList,\
    classifyDataArray, openFile
from Classify.knn import classify0
'''
根据基础的Knn算法得到k：1-100的准确率
返回值：一个100元素的list，每一个元素代表对应下标加一的K的准确率
'''
def globalAccuracy(dataArray,labelsList):
    globalKAccuracy = []
    for  k in range(1,20):
        intX = 0
        intY = 0
        for i in range(10):
            dataTextArray,dataTrainArray = classifyDataArray(dataArray,i+1)
            labelsTexttList,labelsTrainList = classifyDataList(labelsList, i+1)  
            index = len(dataTextArray)
            for j in range(index):
                if labelsTexttList[j]==Classify.knn.classify0(dataTextArray[j], dataTrainArray, labelsTrainList, k):
                    intX = intX+1
                intY =  intY+1
        knnAccuracy = double(intX)/intY
        globalKAccuracy.append(knnAccuracy)
    return globalKAccuracy
'''
寻找把当前点当作最近邻的或者次近邻的三个点
参数：
寻找的范围dataArray，需要匹配的点 intX
返回 ：
        keyList  (type:list)三个点对应dataSet 的下标
'''
def searchKPoint(dataArray,intX):
    dataArraySize=dataArray.shape[0]
    keyList= []
    myList = []
    a = len(dataArray)
    distanceList = zeros((a,a-1))
    index = 0
    index1 = 0
    for List in dataArray:
        diffMat = tile(List, (dataArraySize,1)) - dataArray
        sqDiffMat=diffMat**2 #获得横纵坐标的平方值
        sqDistances = sqDiffMat.sum( axis = 1 ) #获得横纵坐标的差的平方的和;
        distance = sqDistances ** 0.5
        sortedDistIndicies = distance.argsort()#将之前得到的距离排序，距离从小到大排序，返回的是列表中的数字所在的下标位置
        distanceList[index,:] = sortedDistIndicies[1:,]
        index = index + 1
    for j in range(a-1):
        for i in range(a):
            intNum = distanceList[i][j]
            index2 = 0
            for x in range(len(intX)):
                if intX[x]==dataArray[intNum][x]:
                    index2 = index2+1
            if index2 == len(intX):
                myList = []
                myList.append(i)
                myList.append(j+1)
                keyList.append(myList)
                index1 = index1+1
            if  index1==3:
                return keyList    
def getKForeveryPoint(dataArray,labelsList,globalKAccuracy):
    Klist = []
    aNum = 0
    dataArrayDataSize = len(dataArray)
    for i in range(dataArrayDataSize):
        evelAccuracy = 0
        KeyList = searchKPoint(dataArray,dataArray[i])
        for k in range(1,20):
            aNum = 0
            for num in range(3):
                labelsTrain = labelsList[0:KeyList[num][0]]+labelsList[KeyList[num][0]:len(labelsList)]
                normalTrainData = np.vstack( (  dataArray[0:KeyList[num][0]],dataArray[KeyList[num][0]:,]) )
                if labelsList[KeyList[num][0]] == Classify.knn.classify0(dataArray[KeyList[num][0]],normalTrainData,labelsTrain,k):
                    aNum = double(aNum+1-double(KeyList[num][1])/1000)
            if evelAccuracy < (double(aNum)/3 + globalKAccuracy[k-1]):
                myindex = k
                evelAccuracy = double(aNum)/3 + globalKAccuracy[k-1]
        Klist.append(myindex)
    return Klist
'''
函数功能：
就是计算出在找到每一个K值之后得到的整体的KNN‘算法的准确率
'''
def getAccuracyForTrain(fileName):
    dataArray ,labelsList ,kList = createListForData.changeDataToList.getNewArray(fileName)
    intX = 0
    intY = 0
    for i in range(10):
        dataArrayTest ,dataArrayTrain= createListForData.changeDataToList.classifyDataArray(dataArray, i+1)
        labelsTextList, labelsTrainList = createListForData.changeDataToList.classifyDataList(labelsList, i+1)
        KTextList,KTrainList = createListForData.changeDataToList.classifyDataList(kList, i+1)
        number = len(dataArrayTest)
        for j in range(number):
            if labelsTrainList[j] == Classify.knn.classify0(dataArrayTest[j], dataArrayTrain, labelsTrainList, KTextList[j]):
                intX = intX+1
            intY = intY + 1
    newAccurucy = double(intX)/intY
    return newAccurucy
'''
函数功能：就是将得到一个数据的所有近邻点，依次是最近邻，次近邻............最远端
函数参数：dataArray（就是需要寻找的范围），intX（需要处理的点）
返回值：得到一个list，里面依次是它的近邻点
'''
def anyNeiberPoint(dataArray,intX):
    dataArraySize=dataArray.shape[0]
    keyList= []
    a = len(dataArray)
    distanceList = zeros((a,a-1))
    index = 0
    KList  = []
    for List in dataArray:
        diffMat = tile(List, (dataArraySize,1)) - dataArray
        sqDiffMat=diffMat**2 #获得横纵坐标的平方值
        sqDistances = sqDiffMat.sum( axis = 1 ) #获得横纵坐标的差的平方的和;
        distance = sqDistances ** 0.5
        sortedDistIndicies = distance.argsort()#将之前得到的距离排序，距离从小到大排序，返回的是列表中的数字所在的下标位置
        distanceList[index,:] = sortedDistIndicies[1:,]
        index = index + 1
    for j in range(a-1):
        keyList = []
        for i in range(a):
            intNum = distanceList[i][j]
            index1 = 0
            for x in range(len(intX)):
                if intX[x]==dataArray[intNum[x]]:
                    index1 =  index1 +1
            if index1 == len(intX):
                keyList.append(i)
        KList.append(keyList)
    return KList
#print getAccuracyForTrain("/home/daizhen/myData/habermanTrain.txt")