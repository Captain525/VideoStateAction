
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

    #loadedData, size, frameNumbers = loadData()
    data, frameNumbers, relevanceList = loadDataFromCategory(categories[categoryIndex])
    raggedTensor = tf.RaggedTensor.from_row_lengths(tf.constant(data), row_lengths = frameNumbers)
    relevanceTensor = tf.constant(relevanceList, tf.float32)
    delta = 3
    k = 100
    mu = 1
    d = raggedTensor.shape[-1]
    scale = 1
    temp = 1

    model = VideoModel(delta, k, mu, d, scale, theta = thetaValues[categoryIndex], temp=temp)
    model.compile(optimizer = 'adam', run_eagerly = True)
    #model.numpyFitMethod(raggedTensor, epochs = 10, batch_size = 2)
    model.fit(raggedTensor, relevanceTensor, epochs=10, batch_size = 10)

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
    for i in range(len(fileNames)):
        file = fileNames[i]
        #numTransforms x numFrames x dataSize
        loadedArray = np.load(dataPath + file + ".npy")
        listSplit = np.vsplit(loadedArray, loadedArray.shape[0])
        relevanceScoreMiniList = [relevanceScores[i] for j in range(len(listSplit))]
        miniFrameNumbers = [loadedArray.shape[1] for j in range(len(listSplit))]
        frameNumbers += miniFrameNumbers
        dataList +=listSplit
        relevanceScoreList += relevanceScoreMiniList
    data = np.vstack(dataList)
    return data, frameNumbers, relevanceScoreList
runModel()