Implementing paper Look for the Change
Look for the Change: Learning Object States and
State-Modifying Actions from Untrimmed Web Videos
https://openaccess.thecvf.com/content/CVPR2022/papers/Soucek_Look_for_the_Change_Learning_Object_States_and_State-Modifying_Actions_CVPR_2022_paper.pdf


USE ChangeIt dataset
https://github.com/soCzech/ChangeIt

problem with tensorflow was that the cuda toolkit version wasn't valid with tensorflow. ALso wrong tensorflow version.
don't know problem with research env but other env works.



Feature extraction idea: 
Get video file from the dataset. 
Want to extract "features" from the dataset. That is, get a list of T feature vectors for each video, where T is the number of seconds in the video. Each vector represents original resolution of video features into 1 fps. 

First, we go from video -> reduced fps, resized to some common size we desire. This is with the ffmpeg_utils file. 

Then, we want to apply some pretrained feature extractor on it. The paper uses 2D ResNeXT pre-trained and 3D TSM resNet50 pretrained models, then concatenates them together. 