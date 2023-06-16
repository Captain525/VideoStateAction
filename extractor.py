#Copied from source code. 


import os
import pickle
import argparse
import numpy as np
import scipy.ndimage
import torch
import timm

from feature_extraction import ffmpeg_utils, transforms, tsm_model, resnext_model

def extractSimple(videoList):
    exportPath = "transformedData/"
    n_augmentations =0
    model = tsm_model.get_model()
    for video_file in videoList:
        print(f"Processing {video_file}.")

        out_file, _ = os.path.splitext(video_file)
        #no pickle. 
        #out_file += ".pickle"
        if exportPath is not None:
            os.makedirs(exportPath, exist_ok=True)
            out_file = os.path.join(exportPath, os.path.basename(out_file))

        if os.path.exists(out_file):
            print(f"File {out_file} exists, skipping.")
            continue

        size = (480, 270)
        crop = None
        #maybe add crop if necessary. 
        print("video file: ", video_file)
        frames01fps = ffmpeg_utils.extract_frames(video_file, fps= 1, size=size, crop=crop)
        print(np.mean(frames01fps))
        print("done doing video stuff")
        print("shape is: ", frames01fps.shape)
        feats = []
        for i in range(n_augmentations + 1):
            #hopefully this already works. 
            transform = transforms.get_transform(frames01fps.shape[1:3][::-1], identity= (i == 0))
            transformedInputs = transform(frames01fps, end_size=(224,224))/255.0
            #numFrames x featureDims. 
            videoFeatures= model(transformedInputs)
            feats.append(videoFeatures)
        featArray = np.stack(feats, axis=0)
        with open(out_file,"wb") as f:
            #obj = {"video": {"model":[m for m in feats],}}
            #pickle.dump(obj, f)
            np.save(f, featArray)
        
    return 
def callExtractor(videoList):
    exportPath = "transformedData/"
    n_augmentations =0
    tsm = tsm_model.get_model()
    resnext_path = os.getcwd() + "/weights/resnext-101"
    resnext = resnext_model.get_resnextTF(resnext_path, 40)

    #resnext.summary()
    print(timm)

    for video_file in videoList:
        print(f"Processing {video_file}.")

        out_file, _ = os.path.splitext(video_file)
        out_file += ".pickle"
        if exportPath is not None:
            os.makedirs(exportPath, exist_ok=True)
            out_file = os.path.join(exportPath, os.path.basename(out_file))

        if os.path.exists(out_file):
            print(f"File {out_file} exists, skipping.")
            continue

        size = (480, 270)
        crop = None
        #if args.eval_center_crop:
           # size = (398, 224)
           # crop = (398 - 224, 0)
        print("video file: ", video_file)
        frames01fps = ffmpeg_utils.extract_frames(video_file, fps= 1, size=size, crop=crop)
        print(np.mean(frames01fps))
        #frames25fps = ffmpeg_utils.extract_frames(video_file, fps=25, size=size, crop=crop)
        print("done doing video stuff")
        print("shape is: ", frames01fps.shape)
        feats = []
        for i in range(n_augmentations + 1):
            print("first transformation")
            #if it's iteration 0, have identity transform
            transform = transforms.get_transform(frames01fps.shape[1:3][::-1], identity= (i == 0))
            numFrames = frames01fps.shape[0]
            listFeats = []
            transformed = (transform(frames01fps, end_size=(224, 224))).astype(np.float16)/255.0
            for j in range(numFrames):
                resnext_feats = resnext(np.expand_dims(transformed[j, : , : , :], 0))
                listFeats.append(resnext_feats)
            resnext_feats = np.squeeze(np.stack(listFeats, axis=0))
            meanRes = np.mean(resnext_feats, axis=1)
            print("mean of res: ", meanRes)
            print(resnext_feats)

            print("resnet shape: ", resnext_feats.shape)
            transformedInputs = transform(frames01fps, end_size=(224,224))/255.0
            print("transformedInput shape: ", transformedInputs.shape)
            tsm_feats = tsm(transformedInputs)
            print("tsm feats: ", tsm_feats)
            print("shape of tsm feats: ", tsm_feats.shape)
            #tsm_feats_resized = scipy.ndimage.zoom(tsm_feats, (len(resnext_feats) / len(tsm_feats), 1), order=1).astype(np.float16)

            feats.append([resnext_feats, tsm_feats])
            #feats.append([None, tsm_feats])
        with open(out_file, "wb") as f:
            obj = {"video": {
                "resnext": [f1 for f1, f2 in feats],
                "tsm": [f2 for f1, f2 in feats]
            }}
            pickle.dump(obj, f)