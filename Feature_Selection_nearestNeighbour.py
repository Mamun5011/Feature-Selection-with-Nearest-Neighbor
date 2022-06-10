import numpy as np
import copy

def PointandDistance(data,PrevfeatureSet,addedfeature,NNdistance,NNpoint,i,j):
    distance = np.sum((data[i][PrevfeatureSet] - data[j][PrevfeatureSet])**2) + np.sum((data[i][addedfeature] - data[j][addedfeature])**2)
    if distance < NNdistance:
        NNdistance = distance
        NNpoint = j
    return NNdistance,NNpoint

def loadAndNormalizeData():
    fileName= "CS205_SP_2022_SMALLtestdata__52.txt"
    #fileName = "CS205_SP_2022_Largetestdata__66.txt"
    loadedData = np.loadtxt(fileName)
    features = loadedData[:, 1:]  # Reading second column to last column
    label= loadedData[:, 0]  # Reading only first column for label
    #print(features.shape)
    minVal = np.min(features, axis=0),
    maxVal = np.max(features, axis=0)
    #print(maxVal)
    normalizedData = (features - minVal)/(maxVal - minVal)
    return normalizedData,label



def crossValidationForwardSelection(data, label, PrevfeatureSet, addedfeature):
    samplesCount = data.shape[0]
    columnCountforSamples = data.shape[1]
    prediction = 0
    for i in range(samplesCount):
        NNpoint = -1
        NNdistance = np.inf
        for j in range(columnCountforSamples):
            if i != j:
                NNdistance,NNpoint = PointandDistance(data,PrevfeatureSet,addedfeature,NNdistance,NNpoint,i,j)
        if label[i] == label[NNpoint]:
            prediction += 1
    return prediction/samplesCount


def forwardSelection(data, label):
    
    totalSamples = data.shape[0]
    featureSize = data.shape[1]
    classOneCounts = np.where(label == 1)[0].shape[0]
    classTwoCounts = np.where(label == 2)[0].shape[0]
    maxBetweenClass1or2Counts = classOneCounts
    if maxBetweenClass1or2Counts < classTwoCounts:
       maxBetweenClass1or2Counts = classTwoCounts
    emptyFeatureAccuracy = maxBetweenClass1or2Counts/totalSamples
    featuresList = []
    result = []
    result.append((-1, emptyFeatureAccuracy))   # here -1 means {} features
    print("with {} feature accuracy is",emptyFeatureAccuracy)
    
    for i in range(featureSize):
        level = i+1
        print("==========LEVEL ",level,"================")
        feature = -1
        Accuracy = 0
        for j in range(featureSize):
            if j not in featuresList:
                curAccuracy = crossValidationForwardSelection(data, label, featuresList, j)
                print(" Feature ",j," with accuracy ",curAccuracy)
                if curAccuracy > Accuracy:
                    Accuracy = curAccuracy
                    feature = j
        featuresList.append(feature)
        result.append((copy.deepcopy(featuresList), Accuracy))
        print("--------Newly Added Feature ",feature," at level ",level,"has highest accuracy ",Accuracy)
    return result


def PointandDistanceBackward(data,PrevfeatureSet,removedfeature,NNdistance,NNpoint,i,j):
    distance = np.sum((data[i][PrevfeatureSet] - data[j][PrevfeatureSet])**2) 
    if removedfeature in PrevfeatureSet:
       distance = distance - np.sum((data[i][removedfeature] - data[j][removedfeature])**2)
    if distance < NNdistance:
        NNdistance = distance
        NNpoint = j
    return NNdistance,NNpoint


def crossValidationBackwardElimination(data, label, PrevfeatureSet, removedfeature):
    samplesCount = data.shape[0]
    columnCountforSamples = data.shape[1]
    prediction = 0
    for i in range(samplesCount):
        NNpoint = -1
        NNdistance = np.inf
        for j in range(columnCountforSamples):
            if i != j:
                NNdistance,NNpoint = PointandDistanceBackward(data,PrevfeatureSet,removedfeature,NNdistance,NNpoint,i,j)
        if label[i] == label[NNpoint]:
            prediction += 1
    return prediction/samplesCount




def backwardElimination(data, label):

    totalSamples = data.shape[0]
    featureSize = data.shape[1]
    featuresList = []
    for i in range(featureSize):
        featuresList.append(i)
    
    result = []
    accuracy = crossValidationBackwardElimination(data, label, featuresList, -5) # Last parameter negative means no features to remove
    result.append((copy.deepcopy(featuresList), accuracy))
    print("Feature ",featuresList,"has accuracy ",accuracy)
    for i in range(featureSize - 1):
        level = i+1
        print("==========LEVEL ",level,"================")
        feature = -1
        Accuracy = 0
        for j in featuresList:
            curAccuracy = crossValidationBackwardElimination(data, label, featuresList, j)
            print(" Feature ",j," with accuracy ",curAccuracy)
            if curAccuracy > Accuracy:
                Accuracy = curAccuracy
                feature = j
        featuresList.remove(feature)
        print("-------- Removed Feature ",feature," at level ",level)
        result.append((copy.deepcopy(featuresList), Accuracy))
        print("Feature ",featuresList,"has accuracy ",Accuracy)
        

    classOneCounts = np.where(label == 1)[0].shape[0]
    classTwoCounts = np.where(label == 2)[0].shape[0]
    maxBetweenClass1or2Counts = classOneCounts
    if maxBetweenClass1or2Counts < classTwoCounts:
       maxBetweenClass1or2Counts = classTwoCounts
    emptyFeatureAccuracy = maxBetweenClass1or2Counts/totalSamples
    result.append((-1, emptyFeatureAccuracy))
    
    return result


normalizedData,label = loadAndNormalizeData()
print("Forward Selection Starting")
resultOfFeaturesForwardSelection = forwardSelection(normalizedData,label)
print(resultOfFeaturesForwardSelection)

print("Backward Selection Starting")
resultOfFeaturesBackwardElimination = backwardElimination(normalizedData,label)
print(resultOfFeaturesBackwardElimination)
