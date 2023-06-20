 
import tensorflow as tf
import numpy as np
def get_ResNet():
    #avg pooling gets it to a reasonable size. Dont want to include top bc not a classifier. 
    model = tf.keras.applications.resnet50.ResNet50(include_top = False, weights='imagenet', input_shape = (224,224, 3),pooling='avg')
    #model = tf.keras.Sequential([hub.KerasLayer("https://tfhub.dev/deepmind/mmv/tsm-resnet50/1", signature='video',signature_outputs_as_dict=False, output_key='before_head')])
    #model.build([None, None, 200, 200, 3])  # Batch input shape.
    def runModel(inputImages):
        """this edition with batch sizes is much faster than doing one frame at a time, but less memory demanding than doing ALL at once. """
        numFrames = inputImages.shape[0]
        #can't do them all at once because the memory required is too large. Maybe split into batches of a given size. 
        batchSize = 100
        #extra small batch at the end. 
        numBatches = numFrames//batchSize+1
        listFrames = []
        for i in range(numBatches):
            #last still works without indexing errors. 
            batchImage = inputImages[i*batchSize:(i+1)*batchSize, :, : , :]
            visionOutput = model(batchImage)
            listFrames.append(visionOutput)
        visionArray = np.vstack(listFrames)
        print("vision array shape: ", visionArray.shape)
        reshapedFeatures = np.reshape(visionArray, newshape = (numFrames, -1))
        return reshapedFeatures
    return runModel