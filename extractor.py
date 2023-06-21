#Copied from source code. 


import os
import pickle
import argparse
import numpy as np
import scipy.ndimage
import torch
import timm
import time

from feature_extraction import ffmpeg_utils, transforms, tsm_model
from featureExtractor import get_ResNet
from saveLoad import loadNamesCategory
def justExtractAndSave(videoList):
    """
    Extracts from video to numpy frame array, then saves them in the Frames directory. 
    Ran out of memory here earlier, not sure if that bug is still present. Probably because of the list of frames i was saving in memory. 
    Avgs about 1-5 seconds per extraction, so overall if 1000 videos this takes 3000 seconds, or 50 minutes. 
    """
    exportPath = "Frames/"

    size = (480, 270)
    crop = None
    for video in videoList:
        checked, out_file = checkSaving(video, exportPath, True)
        if not checked:
            continue
        frames = extractFramesFromVideo(video, fps=1, size =size, crop=crop)
        #if put with open(outfile, "wb") as f:
            #np.save(f, frames)
        # there would be no .npy. Thus, would have to remove the True above. 
        np.save(out_file, frames)

def loadExtract(nameList):
    path = "Frames/"
    listFrames = []
    startLoad = time.time()
    for name in nameList:
        print("loading: ", name)
        frames = np.load(path + name+ ".npy")
        listFrames.append(frames)
    endLoad = time.time()
    loadTime = endLoad - startLoad
    print("Load time for extract was: ", loadTime)
    return listFrames

def massTransform(listFrames, num_transforms):
    #transforming 2266 frames(10 videos) in about 5-10 seconds. 
    numFramesEach = np.array([video.shape[0] for video in listFrames])
    
    cumulativeSums = np.cumsum(numFramesEach)
    print("before stacked frames")
    stackedFrames = np.vstack(listFrames)
    print("stacked frames size: ", stackedFrames.shape)
    listAllPoints = []
    for i in range(num_transforms):
        print("on transform: ", i)
        transform = transforms.get_transform(stackedFrames.shape[1:3][::-1], identity= (i == 0))
        startTransform = time.time()
        #can treat the list of videos as one big video. 
        transformedInputs = transform(stackedFrames, end_size=(224,224))/255.0
        endTransform = time.time()
        print("transform time: ", endTransform - startTransform)
        #hopefully  our feature extractor WANTS inputs in form 0,1. 
        print("transformedShape: ", transformedInputs.shape)
        assert(transformedInputs.shape==(stackedFrames.shape[0], 224, 224, 3))
        listAllPoints.append(transformedInputs)
    print("time for stacking")
    stackedTransforms = np.vstack(listAllPoints)
    #last one is just a nothing array. 
    #listTransforms = np.split(stackedTransforms, cumulativeSums, axis=1)[:-1]
    #listOneArray = [array.reshape((-1, 224, 224, 3)) for array in listTransforms]
    return stackedTransforms, numFramesEach
def featureExtractFrames(transformedFeatures, numTransforms, numFramesEach):
    #time: 45 seconds for 10 videos. Num frames without transforms 2549, *5 for transforms. 12745 ops. 283 frames per second. 
    model = get_ResNet()
    #numFramesEach = [video.shape[0] for video in transformedFeatures]
    #stackedFeatures = np.vstack(transformedFeatures)
    print("extracting frames")
    startModel = time.time()
    modelOutputs = model(transformedFeatures)
    endModel = time.time()
    print("Model time elapsed: ", endModel - startModel)
    cumulativeSums = np.cumsum(numFramesEach)
    amount = cumulativeSums[-1]
    print("amount", amount)
    #dont want 0
    values = amount*np.arange(0, numTransforms)[1:]
    print("output size: ", modelOutputs.shape)
    print("values: ", values)
    videoFeatures = np.split(modelOutputs, values, axis=0)
    for feature in videoFeatures:
        print(feature.shape)
    stacked = np.stack(videoFeatures,axis=0)
    print("stacked shape: ", stacked.shape)
    videoFeaturesReshaped = np.split(stacked, cumulativeSums, axis=1)[:-1]
    for feature in videoFeaturesReshaped:
        print(feature.shape)
    #videoFeatures = np.split(modelOutputs, numFramesEach, axis=0)[:-1]
    #videoFeaturesReshaped = [array.reshape((numTransforms, -1, modelOutputs.shape[-1]))  for array in videoFeatures]
    #note that videoFeatures are shaped by the transformed ones. 
    return videoFeaturesReshaped
def saveFeatures(videoFeatures, names):
    """
    Can we add a way to stop the pipeline earlier if the things we're trying to get already exist? 
    Bc they're in batches it's more difficult, but there should be a way. 
    """
    exportPath = "transformedData/"
    for i in range(0, len(videoFeatures)):
        name = names[i]
        video = videoFeatures[i]
        assert(video.shape[-1] == 2048)
        assert(video.shape[-2]!=0)
        if os.path.exists(exportPath + name + ".npy"):
            print(name, " already exists")
            continue
        np.save(exportPath + name, video)
    return 
def callTransformAndFeatureExtract(category):
    """
    Assumes frames are saved already.
    Need names for hte category we desire
    """
    names, _ = loadNamesCategory(category)
    batchSize = 20
    #need at least 1 for identity. 
    numTransforms = 5
    start = 100
    iterator = iterData(names, batchSize=batchSize, start=start)
    for nameList, batchNumpy in iterator:
        startBatch = time.time()
        transformedFeatures,numFramesEach= massTransform(batchNumpy, numTransforms)
        featureList = featureExtractFrames(transformedFeatures, numTransforms, numFramesEach)
        
        saveFeatures(featureList, nameList)
        endBatch = time.time()
        print("batchTime: ", endBatch-startBatch)
        

def extractFramesFromVideo(videoLink, fps, size, crop):
    """
    Helper in case we want to change how we extract frames from videos. 
    """
    print("video file: ", videoLink)
    startExtract= time.time()
    frames01fps = ffmpeg_utils.extract_frames(videoLink, fps=fps, size=size, crop=crop)
    endExtract= time.time()
    print("time extract: ", endExtract-startExtract)
    return frames01fps
def extractSimple(videoList):
    """
    timeExtract and model take up the most time all things considered. Might want to extract first. Trnasform and model together.   
    """
    exportPath = "transformedData/"
    #how many alterations to add to the dataset. 
    n_augmentations =0
    model = get_ResNet()
    for video_file in videoList:
        print(f"Processing {video_file}.")
        checked, out_file = checkSaving(video_file, exportPath, False)
        if not checked:
            continue
        #maybe add crop if necessary. 
        size = (480, 270)
        crop = None
        frames01fps = extractFramesFromVideo(video_file, fps=1, size = size, crop= crop)
        print("done doing video stuff")
        print("shape is: ", frames01fps.shape)
        feats = []
        for i in range(n_augmentations + 1):
            #hopefully this already works. 
            transform = transforms.get_transform(frames01fps.shape[1:3][::-1], identity= (i == 0))
            startTransform = time.time()
            transformedInputs = transform(frames01fps, end_size=(224,224))
            endTransform = time.time()
            print("transform time: ", endTransform - startTransform)
            #hopefully  our feature extractor WANTS inputs in form 0,1. 
            print("transformedShape: ", transformedInputs.shape)
            #Normalize the inputs to 0 1 if that's what we need to do. 
            startModel = time.time()
            videoFeatures= model(transformedInputs/255.0)
            endModel = time.time()
            print("Model time: ", endModel - startModel)
            feats.append(videoFeatures)
        featArray = np.stack(feats, axis=0)
        #NO .npy in the save comes from this open step. Not sure which to do. 
        with open(out_file,"wb") as f:
            np.save(f, featArray)
    print("all saved")
    return 
def checkSaving(videoFile, exportPath, numpy):
    out_file, _ = os.path.splitext(videoFile)
    if exportPath is not None:
        os.makedirs(exportPath, exist_ok=True)
        out_file = os.path.join(exportPath, os.path.basename(out_file))
    
    #allows for noticing if numpy files are there. 
    if numpy:
        out_file = out_file + ".npy"
    if os.path.exists(out_file):
        print(f"File {out_file} exists, skipping.")
        return False, out_file
    return True, out_file

class iterData:
    def __init__(self, names, batchSize, start):
        self.links = names
        self.batchSize = batchSize
        self.numBatches = len(self.links)//self.batchSize
        self.count = start//self.batchSize
    def __iter__(self):
        return self
    
    def __next__(self):
        if(self.count>self.numBatches):
            raise StopIteration
        print("in next")
        names = self.links[self.batchSize*self.count:self.batchSize*(self.count+1)]
        arrays = loadExtract(names)
        self.count = self.count + 1
        return names, arrays