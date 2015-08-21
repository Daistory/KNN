#/usr/bin/python
#encoding:utf-8
import createListForData.changeDataToList
import Classify.eveal
import Classify.knn
import ByStronglyCon.NeiberPoint
from numpy import double, arange
import pylab
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot
from array import array
from numpy import *
dataArray , labelsList = createListForData.changeDataToList.openFile("/home/Daizhen/BreastCancer/breastCancer_train.data")
globalKAccuracyList = Classify.eveal.globalAccuracy(dataArray, labelsList)
kList = Classify.eveal.getKForeveryPoint(dataArray, labelsList, globalKAccuracyList)
createListForData.changeDataToList.writeListToTxt("/home/Daizhen/BreastCancer/breastCancer_NewTrain.txt", dataArray, labelsList, kList)
print Classify.eveal.getAccuracyForTrain("/home/Daizhen/BreastCancer/breastCancer_NewTrain.txt")
'''
normalDataArray = Classify.knn.autoNorm(dataArray)
myList = []
number = len(normalDataArray)
for i in range(number):
    myList.append(Classify.eveal.anyNeiberPoint(normalDataArray, normalDataArray[i]))
def descripPoint(normalDataArray,labelsList,myList):
    print "准备开始画图！"
    number = len(normalDataArray)
    for a in range(number-1):
        index= 0
        index1 = 0
        for b in range(number):
            normalDataTextArray = zeros((len(normalDataArray)-1,3))
            for x in myList[b][a]:
                if x==0:
                    normalDataTextArray = normalDataArray[1:306]
                    labelsTextList = labelsList[1:len(labelsList)]
                else:
                    normalDataTextArray[0:x] = normalDataArray[0:x]
                    normalDataTextArray[x-1:number] = normalDataArray[x:number]
                    labelsTextList = labelsList[0:x+1] + labelsList[x+1:number]
                if Classify.knn.classify0(normalDataArray[x], normalDataTextArray, labelsTextList, kList[x])==labelsList[x]:
                    index = index + 1
                index1 = index1 + 1
        plt.plot(a,double(index)/index1,'r.',linewidth=2)
    plt.show()
    print "结束！！"
    '''