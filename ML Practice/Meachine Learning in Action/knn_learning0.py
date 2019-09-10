from numpy import *
import operator


def classify_KNN(inX, dataSet, labels, K):
    # shape 表述了数据集的维度
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(K):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def file2matrix(fileName,dim2):
    fr = open(fileName)
    arrayOLines = fr.readlines()
    numberOfLines = arrayOLines.__len__()
    returnMat = zeros((numberOfLines, dim2))
    classLabelVector = []
    fr = open(fileName)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:dim2]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector


def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)

    ranges = maxVals - minVals

    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet/tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix("../../../data/data4modle.json", 13)
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify_KNN(normMat[i, :], normMat[numTestVecs: m, :], datingLabels[numTestVecs: m], 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]) : errorCount += 1.0
    print("the total error rate is: %f" % (errorCount/float(numTestVecs)))


datingClassTest()


def classifyQuery():
    resultList = ["not a video", "is a video"]
    datingDataMat, datingLabels = file2matrix("../../../data/data4modle.json", 13)
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([12, 0, 0, 0, 0, 1, 1, 0, 0.0, 12.0, 82, 0, 0])
    classifierResult = classify_KNN((inArr - minVals)/ranges, normMat, datingLabels, 3)
    print("this is a:", resultList[classifierResult - 1])

# classifyQuery()
