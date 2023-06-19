from __future__ import unicode_literals
import yt_dlp
import csv
import os
import pandas as pd
def loadCenteringParams():
    """
    Loads centering parameters and list of categories as lists. 
    """
    link = os.getcwd() + "/ChangeIt/ChangeIt-main/categories.csv"
    categoryCSV = pd.read_csv(link)
    #ordered in alphabetical categories.  
    centeringParams = categoryCSV.loc[:, "CENTERING_PARAM"]
    catList = categoryCSV.loc[:, "DIR_NAME"]
    return centeringParams, catList
def loadCategoryData():
    """
    Load the names of categories, and the values for Theta. 
    """
    path = os.getcwd() + "/videoData/categoryData"
    with open(path, "r") as f:
        allLines = f.readlines()
        numCategories = int(allLines[0].strip())
        categoryNames = [name.strip() for name in allLines[1:1+numCategories]]
        thetaValues = [float(value.strip()) for value in allLines[1+numCategories:]]
    return categoryNames, thetaValues

def loadNamesCategory(category):
    """
    Load names of files for a given category, and relevance scores for each video. 
    """
    
    linkNames = os.getcwd() + "/VideoData/" + category
    with open(linkNames, "r") as f:
        allLines = f.readlines()
        #first value is the number of examples for this category
        numExamples = int(allLines[0].strip())
        #second is the file names of the videos and the extracted features. 
        fileNames = [name.strip() for name in allLines[1:1+numExamples]]
        #third is the relevance values. 
        values = [float(num.strip()) for num in allLines[1+numExamples:]]
    return fileNames, values