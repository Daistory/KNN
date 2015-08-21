#!/usr/bin/python 
#encoding:utf-8
from numpy import *
from array import *
'''
函数名：openfile
参数：打开文件的名字（加上路径）
作用 就是将一个一个数字文件保存为一个list文件每行的数据放在一个list里面当作一个元素 建行前三个元素放在一个LIST保存为特征值，最后一个元素保存成分类的类别
filename=/home/daizhen/myData/haberman.data
'''
def openFile(fileName):
    fileHandle = file(fileName,'r')
    myLines = fileHandle.readlines()
    numberOLines = len(myLines)
    classLabels = []
    index = 0
    myList = myLines[0]
    myList = myList.replace('\n','')
    myList = myList.replace('\r','')
    myList = myList.split(',')
    intX = len(myList)
    dataTemp =  zeros((numberOLines,intX-1))
    for fileLine in myLines:
        fileLine = fileLine.replace('\n','') #将一行里面的\n去除
        fileLine = fileLine.replace('\r','') #将一行里面的\r去除
        fileLine = fileLine.split(',')
        for x in range(len(fileLine)-1):
            fileLine[x]=double(fileLine[x])
        dataTemp[index,:]=fileLine[0:intX-1]
        classLabels.append(fileLine[-1])
        index =index+1
    return dataTemp,classLabels
'''
    将一个list分成10个list，方便进行十折交叉验证
'''
def classifyDataList(dataList,intX):
    if len(dataList)<11:
        return 0
    a =double(len(dataList))/11+0.5
    a=int (a)
    if intX*a>len(dataList):
        return 0
    dataTextList = dataList[(intX-1)*a:intX*a]
    dataTrainList = dataList[0:(intX-1)*a] + dataList[intX*a:len(dataList)]
    return dataTextList,dataTrainList
'''
得到一组测试数据，
并将训练数据减去测试数据
n是决定取第几组出来（平均分成了8组，n是一的时候就是前面第一组的所有数据）
'''
def classifyDataArray(dataArray,intN):
    numberHowlong = len(dataArray)
    if numberHowlong<11:
        return 0
    a = double(numberHowlong)/11+0.5
    a = int(a)
    if intN*a>numberHowlong:
        return 0
    b = len(dataArray[0])
    dataTrainArray = zeros((numberHowlong-a,b))
    dataTextArray = dataArray[(intN-1)*a:intN*a]
    dataTrainArray[0:(intN-1)*a] = dataArray[0:(intN-1)*a]
    index = (intN-1)*a
    for Array in dataArray[(intN)*a:,]:
        dataTrainArray[index:,] = Array
        index = index + 1
    return dataTextArray,dataTrainArray
'''
这个函数是将已经得到的dataArray（得到的数据集），labelesList（得到的特征分类），KList（每一个点对应的K取指），全部写进文件
每一行的形式就是  特征的数据，分类类型，k取值
'''
def writeListToTxt(fileName,dataArray,labelsList,kList):
    myFile = file(fileName,'w+')
    numberLong = len(kList)
    for i in range(numberLong):
        for Array in dataArray[i]:
            myFile.write(str(Array))
            myFile.write(",")
        myFile.write(str(labelsList[i]))
        myFile.write(",")
        myFile.write(str(kList[i]))
        myFile.write("\n")
    myFile.close()
    print "已经生成新的文件"
'''
intX:表示打开文件一行的元素个数
'''
def getNewArray(fileName):
    fileHandle = file(fileName,'r')
    myLines = fileHandle.readlines()
    numberOLines = len(myLines)
    myList = myLines[0]
    myList = myList.replace('\n','')
    myList = myList.replace('\r','')
    myList = myList.split(',')
    intX = len(myList)
    dataArray =  zeros((numberOLines,intX-2))
    classLabels = []
    kList = []
    index = 0
    for fileLine in myLines:
        fileLine = fileLine.replace('\n','') #将一行里面的\n去除
        fileLine = fileLine.replace('\r','') #将一行里面的\r去除
        fileLine = fileLine.split(',')
        for i in range(intX-2):
            fileLine[i] = double(fileLine[i])
        fileLine[-1] = int(fileLine[-1])
        dataArray[index,:]=fileLine[0:intX-2]
        classLabels.append(fileLine[-2])
        kList.append(fileLine[-1])
        index =index+1
    return dataArray,classLabels,kList
    