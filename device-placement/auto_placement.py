from __future__ import division
import tensorflow as tf
from dnn_model import DnnModel

if __name__ == "__main__":

    imgHeight = 224
    imgWidth = 224
    numChannels = 3
    numClasses = 1000
    batchSize = 64
    opt = 'SGD'

    features = tf.placeholder(tf.float32, [None, imgHeight, imgWidth, numChannels])
    labels = tf.placeholder(tf.int64, [None, numClasses])

    dm1 = DnnModel('scn', str(1), 1, imgHeight, imgWidth, numChannels, numClasses, batchSize, opt, 0.0001, 'relu')
    dm1_entity = dm1.getModelEntity()

    with tf.deivce('GPU:0'):
        dm1_conv = dm1_entity.build_conv(features)

    with tf.device('/device:CPU:0'):
        dm1_pool = dm1_entity.build_pool(dm1_conv)

    with tf.deivce('GPU:0'):
        dm1_conv = dm1_entity.build_conv(dm1_pool)

    with tf.device('/device:CPU:0'):
        dm1_pool = dm1_entity.build_pool(dm1_conv)

    with tf.deivce('GPU:0'):
        dm1_flatten = dm1_entity.build_flatten(dm1_pool)
        dm1_fc = dm1_entity.build_fc(dm1_flatten)
        dm1_logit = dm1_entity.build_logit(dm1_fc)

    dm1_train = dm1_entity.train(dm1_logit, labels)
    print(dm1_train.graph)

