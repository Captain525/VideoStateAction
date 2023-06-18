#library to download videos from youtube
from __future__ import unicode_literals
import yt_dlp
import csv
import os
from extractor import callExtractor, extractSimple
import pandas as pd

def loadVideos(category):
    csvLink = os.getcwd() + "/ChangeIt/ChangeIt-main/videos/" + category + ".csv"
    preset = "https://www.youtube.com/watch?v="
    csvRead = csv.reader(open(csvLink), delimiter=",")
    count = 0
    listValues = []
    listNames = []
    listLinks = []
    saveLink = os.getcwd() + "/videoData"
    for row in list(csvRead):
        print("on row: ", count)
        #make sure row[0] is a string and works. 
        link = preset + row[0]
        value = float(row[1])
        location = 'DownloadedVideos/{cat}{num}.mp4'.format(cat = category, num = count)
        ydl_opts={ 'format': 'best', 'outtmpl':location}
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([link])
        except:
            print("Exception on video num: ", count)
            continue
        #do operations to it now. 
        listValues.append(value)
        listNames.append(category + str(count))
        listLinks.append(location)
        count = count+1
    with open(saveLink + "/" + category, "w") as f:
           f.write(str(count) + "\n")
           f.writelines([name + "\n" for name in listNames])
           f.writelines([str(relevance) + "\n" for relevance in listValues])
    extractSimple(listLinks)
    return count
def loadCenteringParams():
    link = os.getcwd() + "/ChangeIt/ChangeIt-main/categories.csv"
    categoryCSV = pd.read_csv(link)
    #ordered in alphabetical categories.  
    centeringParams = categoryCSV.loc[:, "CENTERING_PARAM"]
    catList = categoryCSV.loc[:, "DIR_NAME"]
    return centeringParams, catList
def loadCategories():
    centeringParams, listCategories = loadCenteringParams()
    saveLink = os.getcwd() + "/videoData"
    listCounts = []
    for category in listCategories: 
        count =  loadVideos(category)
        listCounts.append(count) 
    with open(saveLink + "/categoryData", "w") as f:
        f.write(str(len(listCategories)) + "\n")
        f.writelines([category + "\n" for category in listCategories])
        f.writelines([str(center) + "\n" for center in centeringParams])
    return 
def transformVideo(videoLink):
    extractSimple([videoLink])
def testLoadCategory():
    category = "Apple"
    count = loadVideos(category)
def downloadAllVideos():
    loadCategories()
def tryIt():
    
    link = 'https://www.youtube.com/watch?v=DdSoQsEKXkk'
    category = "Apple"
    count = 0
    ydl_opts={'format':'best', 'outtmpl':'DownloadedVideos/{cat}{num}.mp4'.format(cat = category, num = count)}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([link])
    
    location = os.getcwd() + "\DownloadedVideos\Apple0.mp4"
    print("location:", location)
    transformVideo(location)
downloadAllVideos()
testLoadCategory()