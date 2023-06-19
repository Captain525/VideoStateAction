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

def massTransform(listFrames):
    num_transforms = 10
    numFramesEach = np.array([video.shape[0] for video in listFrames])
    cumulativeSums = np.cumsum(numFramesEach)
    stackedFrames = np.vstack(listFrames)
    listAllPoints = []
    for i in range(num_transforms):
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
    stackedTransforms = np.stack(listAllPoints, axis=0)
    listTransforms = np.split(stackedTransforms, cumulativeSums, axis=1)
    for array in listTransforms:
        print("array shape: ", array.shape)
    return listTransforms
def featureExtractFrames(transformedFeatures):
    exportPath = "transformedData/"
    model = get_ResNet()
    numFramesEach = [video.shape[0] for video in transformedFeatures]
    stackedFeatures = np.vstack(transformedFeatures)
    startModel = time.time()
    modelOutputs = model(stackedFeatures)
    endModel = time.time()
    print("Model time elapsed: ", endModel - startModel)
    videoFeatures = np.split(modelOutputs, numFramesEach, axis=0)
    #note that videoFeatures are shaped by the transformed ones. 
    return videoFeatures
def callTransformAndFeatureExtract(category):
    """
    Assumes frames are saved already.
    Need names for hte category we desire
    """
    names, _ = loadNamesCategory(category)
    listFrames = loadExtract(names)
    transformedFeatures = massTransform(listFrames)
    

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