
from model import VideoModel
import os
import tensorflow as tf
import pickle
import numpy as np
from saveLoad import loadCategoryData, loadNamesCategory

def runModel():
    """
    Run the model with data. 
    """
    categories, thetaValues = loadCategoryData() 
    categoryIndex = 0
    print(categories[categoryIndex])

    #loadedData, size, frameNumbers = loadData()
    data, frameNumbers, relevanceList = loadDataFromCategory(categories[categoryIndex])
    print("got here")
    raggedTensor = tf.RaggedTensor.from_row_lengths(data, row_lengths = frameNumbers)
    print(raggedTensor.shape)
    relevanceTensor = tf.constant(relevanceList, tf.float32)
    delta = 3
    k = 100
    mu = 1
    d = raggedTensor.shape[-1]
    scale = 1
    temp = 1
    print("got before model")
    model = VideoModel(delta, k, mu, d, scale, theta = thetaValues[categoryIndex], temp=temp)
    model.compile(optimizer = 'adam', run_eagerly = True)
    #model.numpyFitMethod(raggedTensor, epochs = 10, batch_size = 2)
    print("ready to fit")
    model.fit(raggedTensor, relevanceTensor, epochs=10, batch_size = 5)

def loadDataFromCategory(category):
    """
    Load data from the np saves. All for a given category. 
    """
    #load the file names. 
    dataPath = os.getcwd() + "/transformedData/"
    
    fileNames, relevanceScores = loadNamesCategory(category)
    dataList = []
    relevanceScoreList = []
    frameNumbers = []
    listDefective = []
    for i in range(100):
        file = fileNames[i]
        #numTransforms x numFrames x dataSize
        print("Link: ", dataPath + file +".npy")
        loadedArray = np.load(dataPath + file + ".npy")
        print("loaded shape: ", loadedArray.shape)
        if(loadedArray.shape[1] == 0):
            listDefective.append(i)
            continue
        listSplit = np.vsplit(loadedArray, loadedArray.shape[0])
        print(listSplit[0].shape)
        relevanceScoreMiniList = [relevanceScores[i] for j in range(len(listSplit))]
        miniFrameNumbers = [loadedArray.shape[1] for j in range(len(listSplit))]
        frameNumbers += miniFrameNumbers
        dataList +=[tf.constant(np.squeeze(element, axis=0)) for element in listSplit]
        relevanceScoreList += relevanceScoreMiniList

    print("data list length: ", len(dataList))
    print(listDefective)
    print(len(listDefective))
    for tensor in dataList:
        print(tensor.shape)
    #essential for the ragged tensor step. 
    data = np.vstack(dataList)
    return data, frameNumbers, relevanceScoreList
runModel()