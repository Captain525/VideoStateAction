
from model import VideoModel
import os
import tensorflow as tf
import pickle
import numpy as np
from saveLoad import loadCategoryData, loadNamesCategory

def runModel():
    """
    Run the model with data. 
    Hyperparameters obtained from the paper
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
    #delta=2 means 5 examples. 
    delta = 2
    k = 60
    mu = 10
    d = raggedTensor.shape[-1]
    scale = .2
    temp = .001
    print("got before model")
    model = VideoModel(delta, k, mu, d, scale, theta = thetaValues[categoryIndex], temp=temp)
    #add run_eagerly=True
    model.compile(optimizer = 'adam')
    #model.compile(optimizer= 'adam', run_eagerly=True)
    #model.numpyFitMethod(raggedTensor, epochs = 10, batch_size = 2)
    print("ready to fit")
    model.fit(raggedTensor, tf.stack([relevanceTensor, tf.constant(frameNumbers, dtype=tf.float32)], axis=1),  epochs=10, batch_size =10)

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