import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


def get_model():
    #T is the number of steps or whatever. 
    model = tf.keras.applications.resnet50.ResNet50(include_top = False, weights='imagenet', input_shape = (224,224, 3),pooling='avg')
    #model = tf.keras.Sequential([hub.KerasLayer("https://tfhub.dev/deepmind/mmv/tsm-resnet50/1", signature='video',signature_outputs_as_dict=False, output_key='before_head')])
    #model.build([None, None, 200, 200, 3])  # Batch input shape.
    def runModel(inputImages):
        numFrames = inputImages.shape[0]
        listFrames = []
        for i in range(numFrames):
            vision_output = model(np.reshape(inputImages[i,:, :, :], newshape=(1, 224, 224, 3)))
            listFrames.append(vision_output)
        visionArray = np.stack(listFrames, axis=0)
        reshapedFeatures = np.reshape(visionArray, newshape = (numFrames, -1))
        return reshapedFeatures
    return runModel