
from model import VideoModel
import os
import tensorflow as tf
import pickle
import numpy as np


def runModel():
    """
    Run the model with data. 
    """
    categories, thetaValues = loadCategoryData() 
    categoryIndex = 0

    #loadedData, size, frameNumbers = loadData()
    data, frameNumbers, relevanceList = loadDataFromCategory(categories[categoryIndex])
    raggedTensor = tf.RaggedTensor.from_row_lengths(tf.constant(data), row_lengths = frameNumbers)
    relevanceTensor = tf.constant(relevanceList, tf.float32)
    #newRaggedTensor = tf.repeat(raggedTensor, repeats = 5, axis=0)
    #raggedTensor = tf.ragged.constant(tensorList, ragged_rank = 1)
    #print(newRaggedTensor.shape)
    delta = 3
    k = 100
    mu = 1
    d = raggedTensor.shape[-1]
    scale = 1
    temp = 1

    model = VideoModel(delta, k, mu, d, scale, theta = thetaValues[categoryIndex], temp=temp)
    model.compile(optimizer = 'adam', run_eagerly = True)
    #model.numpyFitMethod(raggedTensor, epochs = 10, batch_size = 2)
    model.fit(raggedTensor, relevanceTensor, epochs=10, batch_size = 2)

def loadDataNumpy():
    listArrays = []
    listData = ["Apple0", "Apple1"]
    path = os.getcwd() + "/transformedData"

    for data in listData:
        loadedArray = tf.constant(np.squeeze(np.load(path + "/" + data)), tf.float32)
        print(loadedArray.shape)
        size = loadedArray.shape[-1]
        listArrays.append(loadedArray)
    return listArrays, size
def loadData():
    path = os.getcwd() + "/transformedData"
    array0 = np.squeeze(np.load(path + "/Apple0"))
    print(array0.shape)
    
    array1 = np.squeeze(np.load(path + "/Apple1"))
    print(array1.shape)
    size = array0.shape[-1]
    frameNumbers = [array0.shape[-2], array1.shape[-2]]
    arrayList = [array0, array1]
    listArrays = np.vstack([array0, array1])
    print(listArrays.shape)
    return listArrays, size, frameNumbers
def loadDataFromCategory(category):
    #load the file names. 
    dataPath = os.getcwd() + "/transformedData/"
    path = os.getcwd() + "/videoData/" + category
    with open(path, "r") as f:
        allLines = f.readlines()
        numExamples = int(allLines[0].strip())
        fileNames = [fileName.strip() for fileName in allLines[1:1+numExamples]]
        relevanceScores = [float(score.strip()) for score in allLines[1+numExamples:]]
    dataList = []
    relevanceScoreList = []
    frameNumbers = []
    for i in range(len(fileNames)):
        file = fileNames[i]
        #numTransforms x numFrames x dataSize
        loadedArray = np.load(dataPath + file)
        listSplit = np.vsplit(loadedArray, loadedArray.shape[0])
        relevanceScoreMiniList = [relevanceScores[i] for j in range(len(listSplit))]
        miniFrameNumbers = [loadedArray.shape[1] for j in range(len(listSplit))]
        frameNumbers += miniFrameNumbers
        dataList +=listSplit
        relevanceScoreList += relevanceScoreMiniList
    data = np.vstack(dataList)
    return data, frameNumbers, relevanceScoreList
def loadCategoryData():
    path = os.getcwd() + "/videoData/categoryData"
    with open(path, "r") as f:
        allLines = f.readlines()
        numCategories = int(allLines[0].strip())
        categoryNames = [name.strip() for name in allLines[1:1+numCategories]]
        thetaValues = [float(value.strip()) for value in allLines[1+numCategories:]]
    return categoryNames, thetaValues
runModel()