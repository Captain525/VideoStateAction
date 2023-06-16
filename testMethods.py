import numpy as np
"""
ADD CHANGES HERE TO MODEL. PRETTY SURE IT WORKS NOT 100%


"""
def findMaxPair( gValues, hInitial, hEnd):
        """
        Method to find the max pair. 
        """
        videoLen = gValues.shape[0]
        #s1 x sl x s2
        validTuples = np.ones(shape = (videoLen, videoLen, videoLen))
        #last 2 indices 0. 
        validTuples[videoLen-2:, :, :] = 0
        #the action can't be the first or last. 
        validTuples[:, 0, :] = 0
        validTuples[:, videoLen-1,:] = 0
        #this can't be 0 or 1. 
        validTuples[:, :, 0:2]
        #lower triangular, includes center diagonal. 
        triangleXIndices, triangleYIndices = np.tril_indices(videoLen, k=0)
        validTuples[triangleXIndices, :, triangleYIndices] = 0
        assert(validTuples[0, 1, 2] == 1)
        assert(validTuples[1, 2, 1] == 0)
        assert(validTuples[2, 1, 2]==0)
        assert(validTuples[2,1,1]==0)
        
        validTuples[triangleXIndices, triangleYIndices, :] = 0
        assert(validTuples[0, 1, 2] == 1)
        assert(validTuples[1,1,2] == 0)
        validTuples[:, triangleXIndices, triangleYIndices] = 0
        assert(validTuples[0, 1, 2] == 1)
        assert(validTuples[0, 1, 1] == 0)
        assert(validTuples[0, 2, 1] == 0)
        assert(validTuples[2, 1, 3] == 0)
        assert(validTuples[3, 1, 2] == 0)
        rangeValues= np.arange(start=0, stop=videoLen, step=1)  
        
        productArray = np.tensordot(hInitial, gValues, axes=0)
        productTensor = np.tensordot(productArray, hEnd, axes=0)
        print(productTensor[0,1,2])
        print(hInitial[0]*gValues[1]*hEnd[2])

        assert(productTensor[0, 1, 2] ==hInitial[0]*gValues[1]*hEnd[2])
        #if it works, can multiply by the triangular matrix to get zeros. 
        weightedArray = np.multiply(validTuples, productTensor)
        finalIndices = np.unravel_index(np.argmax(weightedArray), shape = (videoLen, videoLen, videoLen))
        print(finalIndices)
        print(weightedArray[finalIndices])
        return finalIndices
"""
OLD CODE FOR MAX PAIR WITH MAXES. 
 rangeValues= np.arange(start=0, stop=videoLen, step=1)
        validInitial = rangeValues<videoLen-2
        #can't be the last frame or first frame. 
        validAction = np.logical_and(0<rangeValues, rangeValues<videoLen-1)
        validEnd = rangeValues>1
        #note: the optimal initial will never be greater than the maximum, optimal end never lower than the maximum
        #we KNOW hInitial != hEnd unless .5, in which case all states are equal. pick anything. 
        maxInitialIndex = np.argmax(hInitial*validInitial)
        maxEndIndex = np.argmax(hEnd*validEnd)
        maxActionIndex = np.argmax(gValues*validAction)
        #set them all to 0. 
        validInitial[maxInitialIndex+1:] = 0
        validEnd[0:maxEndIndex] = 0
        if(maxInitialIndex==maxEndIndex):
            #in this case all are random, pick some random values which work. 
            firstIndex = videoLen//3
            secondIndex = videoLen//2
            thirdIndex = firstIndex*2
            assert(firstIndex<secondIndex and secondIndex<thirdIndex and thirdIndex < videoLen)
            return np.array([firstIndex, secondIndex, thirdIndex])
        #abnormal behavior
        if(maxEndIndex<maxInitialIndex):
            return 
        else:
            #max initialIndex<maxEndIndex
            #perfect, max of the three are the three maximums. 
            if(maxActionIndex>maxInitialIndex and maxActionIndex<maxEndIndex):
                return np.array([maxInitialIndex, maxActionIndex, maxEndIndex])
            #in this case, initial<end but action on either edge, above or below. Note action can equal either. 
            maxBetween = np.argmax(gValues*np.logical_and(maxInitialIndex<rangeValues, rangeValues<maxEndIndex))
            #all other between values irrelevant. 
            validAction[maxInitialIndex+1:maxBetween] = 0
            validAction[maxBetween+1:maxEndIndex] = 0
            if(maxActionIndex<maxInitialIndex):
                 return 
            return

            
"""
def testCalcMax():
        gValues = np.array([.4, .2, .1, .5, .6])
        hInitial = np.array([.2, .6, .7, .3, .4])
        hEnd = np.array([.8, .4, .3, .7, .6])

        print(findMaxPair(gValues, hInitial, hEnd))
testCalcMax()