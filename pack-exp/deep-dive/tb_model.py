from __future__ import division
import tensorflow as tf
from tensorflow.python.client import timeline
from dnn_model import DnnModel
from img_utils import *

def execTrain(unit, num_epoch, X_train, Y_train):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        num_batch = Y_train.shape[0] // batchSize
        num_batch_list = np.arange(num_batch)
        for e in range(num_epoch):
            for i in range(num_batch):
                print('epoch %d / %d, step %d / %d' % (e + 1, num_epoch, i + 1, num_batch))
                batch_offset = i * batchSize
                batch_end = (i + 1) * batchSize
                X_mini_batch_feed = X_train[batch_offset:batch_end, :, :, :]
                Y_mini_batch_feed = Y_train[batch_offset:batch_end, :]
                sess.run(unit, feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})


if __name__ == '__main__':
    image_path = '/home/user/Development/dataset/imagenet1k'
    bin_path = '/home/user/Development/dataset/imagenet1k.bin'
    label_path = '/home/user/Development/dataset/imagenet1k-label.txt'
    profile_dir = '/home/user/Development/mtml-tf/deep-dive/profile_dir'

    imgHeight = 224
    imgWidth = 224
    numChannels = 3
    numClasses = 1000
    batchSize = 32
    opt = 'SGD'

    features = tf.placeholder(tf.float32, [None, imgHeight, imgWidth, numChannels])
    labels = tf.placeholder(tf.int64, [None, numClasses])

    dm1 = DnnModel('scn', str(1), 1, imgHeight, imgWidth, numChannels, numClasses, batchSize, opt, 0.0001, 'relu')
    dm1_entity = dm1.getModelEntity()
    dm1_logit = dm1_entity.build(features)
    dm1_train = dm1_entity.train(dm1_logit, labels)

    dm2 = DnnModel('scn', str(2), 1, imgHeight, imgWidth, numChannels, numClasses, batchSize, opt, 0.0001, 'relu')
    dm2_entity = dm2.getModelEntity()
    dm2_logit = dm2_entity.build(features)
    dm2_train = dm2_entity.train(dm2_logit, labels)

    labels_data = load_labels_onehot(label_path, numClasses)
    execTrain([dm1_train, dm2_train], 1, image_path, labels_data)