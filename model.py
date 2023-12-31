import tensorflow as tf
import numpy as np
class VideoModel(tf.keras.Model):
    def __init__(self, delta, k, mu, d, scale):
        super().__init__(self)
        self.g = ActionClassifier(delta, k, mu, d)
        self.h = StateClassifier(delta, d)
        self.l = scale
        
    def train_step(self, videos):
        """
        Called on each step of the training algorithm. 
        batch of videos: batch_size x d x numberFrames 
        each frame is a d dim vector of features representing a frame. 
        One specific category. Call train step on a batch. 

        Need videos to somehow deal with having varying sizes. 
        """
        #assume we have a list of videos. 
        #step 1: find labels for each category. 
        
        print(tf.executing_eagerly())
        print(videos.shape)
        print(videos.values.numpy())
        with tf.GradientTape() as tape:
            sumGLoss = 0
            sumHLoss = 0
            #hopefully summing the losses of each video works well. 
            """=
            for video in videos:
                videoLabels = self.generate_new_labels(video)
                gLoss, hLoss = self.train_step_classifiers((video, videoLabels))
                sumGLoss+=gLoss
                sumHLoss+=hLoss
            """
            sumGLoss, sumHLoss = self.doMap(videos)
            totalLoss = sumHLoss + self.l*sumGLoss
        """
        gradG = tape.gradient(sumGLoss, self.g.trainable_weights)
        gradH = tape.gradient(sumHLoss, self.h.trainable_weights)
        self.optimizer.apply_gradients(zip(gradG, self.g.trainable_weights))
        self.optimizer.apply_gradients(zip(gradH, self.h.trainable_weights))
        """
        grad = tape.gradient(totalLoss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grad, self.trainable_weights))
        return totalLoss, sumGLoss, sumHLoss
    def doMap(self, videos):
        listLabels = tf.map_fn(self.generate_new_labels, videos, dtype=tf.int32)
        print(listLabels)
        #list of tuples
        print(videos)
        listlosses = tf.map_fn(self.train_step_classifiers, (videos,  listLabels))
        sumG, sumH = np.sum(np.array(listlosses), axis=0)
        return sumG, sumH
    def train_step_classifiers(self, videoLabelPair):
        """
        treat labels as fixed.
        """
        video = videoLabelPair[0]
        labels = videoLabelPair[1]
        print("video: ", video)
        print("labels: ", labels)
        gLoss = self.g.compute_loss(video, labels[1])
        hLoss = self.h.compute_loss(video, np.array([labels[0], labels[2]]))
        return (gLoss, hLoss)
    def generate_new_labels(self, video):
        """
        Assume fixed g and h. We want to find the most likely location of the action, inital state, and end state. 
        THe video contains action with high probability. 
        l(v) = argmax(l in Dv) h1(xls1)*g(xla)*h2(xls2)
        """
        gValues = self.g(video)
        hValues = self.h(video)
        hInitial = hValues[:, 0]
        hEnd = hValues[:, 1]
        #this method finds the max pair according to the causal ordering constraint. 
        labels = self.findMaxPair(hInitial, gValues, hEnd)
        print(labels)
        #assert(labels[0]<labels[1] and labels[1]<labels[2])
        return labels
   
    def findMaxPair(self, hInitial, gValues, hEnd):
       
        """
        Method to find the max pair. 
        """
        videoLen = gValues.shape[0]
        print("video len: ", videoLen)
        #s1 x sl x s2
        validTuples = np.ones(shape = (videoLen, videoLen, videoLen), dtype = bool)
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
        
        print(hInitial.shape)
        print(gValues.shape)
        print(hEnd.shape)
        productArray = tf.tensordot(hInitial, gValues, axes=0)
        productTensor = tf.tensordot(productArray, hEnd, axes=0)
        #print(productTensor[0,1,2])
        #print(hInitial[0]*gValues[1]*hEnd[2])

        #assert(productTensor[0, 1, 2] ==hInitial[0]*gValues[1]*hEnd[2])
        #if it works, can multiply by the triangular matrix to get zeros. 
        print(productTensor.shape)
        print(validTuples.shape)
        weightedArray =  productTensor*validTuples
        print(weightedArray.shape)
        #changed shape to dims when going to tf from np
        print(tf.math.argmax(tf.reshape(weightedArray, (-1,)), axis=None))
        finalIndices = tf.cast(tf.unravel_index(tf.argmax(tf.reshape(weightedArray, (-1,))), dims = (videoLen, videoLen, videoLen)), tf.int32)
        #print(finalIndices)
        #print(weightedArray[finalIndices])
        return finalIndices
    def testCalcMax(self):
        gValues = np.array([.4, .2, .1, .5, .6])
        hInitial = np.array([.2, .6, .7, .3, .1])
        hEnd = np.array([.8, .4, .3, .7, .9])
        self.findMaxPair(hInitial, gValues, hEnd)

    def call(self, video):
        """
        What happens when you call the model. 
        """
        actionProbabilities = self.g(video)
        labelProbabilities = self.h(video)
        return actionProbabilities, labelProbabilities
class ActionClassifier(tf.keras.Model):
    """
    g in the apper. It's a classifier which takes a visual feature as an input, 
    then outputs a conifdence score 
    that the feature depicts the action of interest. In range [0,1]. Uses labels. 
    TAKES IN A FRAME feature vector. 
    """
    def __init__(self, delta, k, mu, d):
        super().__init__()
        self.delta = delta
        self.k = k
        self.mu = mu
        self.dense1 = tf.keras.layers.Dense(512, input_dim = (None, d), activation="relu")
        #Sigmoid to put in range 0,1 
        self.dense2 = tf.keras.layers.Dense(1, activation = "sigmoid")
    def call(self, video):
        intermediate = self.dense1(video)
        assert(intermediate.shape == (video.shape[0], 512))
        output = tf.squeeze(self.dense2(intermediate))
        assert(output.shape == (video.shape[0],))
        return output

    def compute_loss(self, video, label):
        """
        THe loss function for the action classifier is a cross entropy loss, but with a twist. 
        -mu sum(over APv) log(g(xt)) -sum(over ANv) log(1-g(xt))
        APv = sets of positive examples deduced from l(v) where the model is expected to predict the action. 
        ANv - no action label. 
        labels: action label: position of the action in the video. 

        Maybe add batches somehow. 
        """

        #code here designed for ONE video, ie a batch size of 1, so that the length of the video is uniform, because that's needed. 
        #video is numFrames x featureVector. 
        videoLen = video.shape[-2]
        #works for both batch size and no batch size. Doesnt work if make numFrames the last thing. 
        positive, negative = self.computePositiveAndNegativeExamples(videoLen, label)
        #call the model. Want results of size videoLen x 1. 
        gPos = self(video[positive, :])
        gNeg = self(video[negative,:])

        #assuming  numpy but might have to change. 
        positiveTerm = -1*tf.sum(tf.log(gPos))
        negativeTerm = -1*tf.sum(1-tf.log(gNeg))
        loss = self.mu*positiveTerm + negativeTerm
        return loss
    def computePositiveAndNegativeExamples(self, videoLen, label):
        
        #calculate positive frames from the label scalar(would be vec of size batchSize if not). 
        #Indexing with 0. Include start not end. 
        start = label-self.delta
        end = label + self.delta
        #if label=self.delta already 0. 
        if label<self.delta:
            start = 0
        #if label+self.delta = videoLen then end already included. 
        if label+self.delta>videoLen:
            end = videoLen
        #don't include the end index. 
        #np.arange -> tf.range
        #stop -> limit
        #step -> delta
        positiveExampleIndices = tf.range(start = start, limit = end, delta = 1, dtype = tf.int32)

        negativeExamplesPlus = positiveExampleIndices + self.k
        negativeExamplesMinus = positiveExampleIndices - self.k
        validNegPlus = tf.cast(negativeExamplesPlus < videoLen, tf.int32)
        validNegMinus= tf.cast(negativeExamplesMinus >=0,tf.int32)
        #the actual value where it's valid, a -1 where it isn't. 
        examplesPlus =(negativeExamplesPlus*validNegPlus - negativeExamplesPlus*tf.cast(tf.math.logical_not(tf.cast(validNegPlus, bool)), tf.int32))
        examplesMinus = (negativeExamplesMinus*validNegMinus - negativeExamplesMinus*tf.cast(tf.math.logical_not(tf.cast(validNegMinus, bool)), tf.int32))
        examples = tf.concatenate([examplesPlus, examplesMinus], axis=None)
        #need to get rid of the -1 as well. 
        negativeExamples = tf.unique(examples)
        #should be sorted in ascending order. 
        if negativeExamples[0]==-1:
            negativeExamples = negativeExamples[1:]
        #now have a numpy array of positive examples and one of negative examples. 
        return positiveExampleIndices, negativeExamples


class StateClassifier(tf.keras.Model):
    """
    h in the paper. Classifier which takes as an input the fisual feature of a given frame 
    then gives an estimate of probability that the feature corresponds to the initial and end state. 
    uses labels. Outputs 2 scores, one for prob of initial, other for prob of end state. 
    """
    def __init__(self, delta, d):
        super().__init__()
        self.delta = delta
        self.dense1 = tf.keras.layers.Dense(512, input_dim = ( None, d), activation="relu")
        #Softmax acgivation because it has to be either the initial state or the end state.  
        self.dense2 = tf.keras.layers.Dense(2, activation = "softmax")
    def call(self, video):
        #assumes just one video. 
        intermediate = self.dense1(video)
        assert(intermediate.shape == (video.shape[0], 512))
        output = self.dense2(intermediate)
        assert(output.shape == (video.shape[0], 2))
        return output
    def compute_loss(self, video, labels):
        """
        loss for h. Cross entropy with a twist. 
        -sum(over S1v) logh1(xt) - sum(over S2v) logh2(vt)
        S1v is the set of positive exammples deduced from l(v) where model is expected to predicct the initial state. 
        S2v - set of pos examples deduced from l(v) where model expected to predict the end state. 
        So each is the frames where the initial/end states are predicted. 
        Derived from the set of labels. 
        Label input: a number between 1 and teh length of the video, 
        representing the initial state, and a number between 1 and the length of
        the video representing the end state. 
        so it's batch_size x 2
        maybe make video batch_size x numFeatures x length? Are we going to pad? 
        video is batch_size x length x numFeatures
        """
        videoLen = video.shape[-2]
        initLabelPos, endLabelPos = self.samplePosExamples(videoLen, labels)
        hPos = self(video[initLabelPos, :])
        hNeg = self(video[endLabelPos, :])
        hPosLoss = -1*np.sum(np.log(hPos))
        hNegLoss = -1*np.sum(np.log(hNeg))
        loss = hPosLoss + hNegLoss
        return loss

    def samplePosExamples(self, videoLen, labels):
        """
        Labels: size 2. 
        """
        initLabel = labels[0]
        endLabel = labels[1]
        #np arrays of the examples. 
        initLabelPos = self.calcWindow(videoLen, initLabel)
        endLabelPos = self.calcWindow(videoLen, endLabel)
        return initLabelPos, endLabelPos

    def calcWindow(self, videoLen, label):
        start = label-self.delta
        end = label + self.delta
        #if label=self.delta already 0. 
        if label<self.delta:
            start = 0
        #if label+self.delta = videoLen then end already included. 
        if label+self.delta>videoLen:
            end = videoLen
        #don't include the end index. 
        positiveExampleIndices = np.arange(start = start, stop = end, step = 1, dtype = int)
        return positiveExampleIndices
