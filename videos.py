#library to download videos from youtube
from __future__ import unicode_literals
import yt_dlp
import csv
import os
from extractor import callExtractor, extractSimple

def loadVideosFromCSVLink(csvLink, category):
    preset = "https://www.youtube.com/watch?v="
    csvRead = csv.reader(open(csvLink), delimiter=",")
    count = 0
    for row in list(csvRead):
        #make sure row[0] is a string and works. 
        link = preset + row[0]
        ydl_opts={ 'outtmpl':'DownloadedVideos/{cat}{num}'.format(cat = category, num = count)}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([link])
        count = count+1

def transformVideo(videoLink):
    extractSimple([videoLink])
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
tryIt()