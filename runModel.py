
from model import VideoModel
import os
import tensorflow as tf
import pickle
import numpy as np


def runModel():
    """
    Run the model with data. 
    """
    loadedData, size, frameNumbers= loadData()
    raggedTensor = tf.RaggedTensor.from_row_lengths(tf.constant(loadedData), row_lengths = frameNumbers)
    #newRaggedTensor = tf.repeat(raggedTensor, repeats = 5, axis=0)
    #raggedTensor = tf.ragged.constant(tensorList, ragged_rank = 1)
    #print(newRaggedTensor.shape)
    delta = 3
    k = 3
    mu = 1
    d = size
    scale = 1

    model = VideoModel(delta, k, mu, d, scale)
    model.compile(optimizer = 'adam', run_eagerly = True)
    model.fit(raggedTensor, epochs = 10, batch_size = 2)
    


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
runModel()