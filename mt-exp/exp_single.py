from __future__ import division
import tensorflow as tf
from tensorflow.python.client import timeline
import numpy as np
from multiprocessing import Process
import argparse
from timeit import default_timer as timer

from img_utils import *
from dnn_model import DnnModel

#########################
# Command Parameters
#########################

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, default='resnet', help='choose a model to run')
parser.add_argument('-p', '--preproc', action='store_false', default=True, help='use preproc to transform the data before training or not')
parser.add_argument('-s', '--trainstep', action='store_false', default=True, help='measure the single step or the whole model')
parser.add_argument('-o', '--optimizer', type=str, default='Adam', help='identify an optimizer for training')
parser.add_argument('-bs', '--batchsize', type=int, default=32, help='identify the batch size')
parser.add_argument('-l', '--layersize', type=int, default=1, help='identify the layer of deep learning model')
args = parser.parse_args()

#########################
# Global Variables
#########################

#image_dir = '/home/ruiliu/Development/mtml-tf/dataset/imagenet10k'
image_dir = '/tank/local/ruiliu/dataset/imagenet10k'
#image_dir = '/local/ruiliu/dataset/imagenet10k'

#bin_dir = '/home/ruiliu/Development/mtml-tf/dataset/imagenet1k.bin'
bin_dir = '/tank/local/ruiliu/dataset/imagenet10k.bin'
#bin_dir = '/local/ruiliu/dataset/imagenet10k.bin'

#label_path = '/home/ruiliu/Development/mtml-tf/dataset/imagenet1k-label.txt'
label_path = '/tank/local/ruiliu/dataset/imagenet10k-label.txt'
#label_path = '/local/ruiliu/dataset/imagenet10k-label.txt'

#profile_dir = '/home/ruiliu/Development/mtml-tf/mt-perf/profile_dir'
#profile_dir = '/tank/local/ruiliu/mtml-tf/mt-perf/profile_dir'

imgWidth = 224
imgHeight = 224
numClasses = 1000
numChannels = 3
numEpochs = 1

marker = 3

trainModel = args.model
trainBatchSize = args.batchsize
preproc = args.preproc
trainOptimizer = args.optimizer
trainModelLayer = args.layersize
trainStepMeasure = args.trainstep

features = tf.placeholder(tf.float32, [None, imgWidth, imgHeight, numChannels])
labels = tf.placeholder(tf.int64, [None, numClasses])

def showExpConfig():
    print("run the model:", trainModel)
    print("Using the preprocess:", preproc)
    print("Using the optimizer:", trainOptimizer)
    print("Using the batch size:", trainBatchSize)

def prepareModel():
    model_name_abbr = np.random.choice(100000, 1, replace=False).tolist()
    dm = DnnModel(trainModel, str(model_name_abbr.pop()), model_layer=1, 
                  input_w=imgWidth, input_h=imgHeight, num_classes=numClasses, 
                  batch_size=trainBatchSize, optimizer=trainOptimizer)
    return dm

def printModelDes(model):
    print(model.getInstanceName())
    print(model.getModelName())
    print(model.getModelLayer())
    print(model.getBatchSize())
    print("===============")

def buildModel(model):
    modelEntity = model.getModelEntity()
    modelLogit = modelEntity.build(features)
    _, _, trainStep = modelEntity.train_step(modelLogit, labels)
    return trainStep

def execTrainStep(trainStep, num_epoch, X_train, Y_train):
    step_time = 0
    step_count = 0
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        num_batch = Y_train.shape[0] // trainBatchSize
        for e in range(num_epoch):
            for i in range(num_batch):
                print('epoch %d / %d, step %d / %d' %(e+1, num_epoch, i+1, num_batch))
                if (i+1) % marker == 0:
                    start_time = timer()
                    batch_offset = i * trainBatchSize
                    batch_end = (i+1) * trainBatchSize
                    X_mini_batch_feed = X_train[batch_offset:batch_end,:,:,:]
                    Y_mini_batch_feed = Y_train[batch_offset:batch_end,:]
                    sess.run(trainStep, feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})
                    end_time = timer()
                    dur_time = end_time - start_time
                    print("step time:",dur_time)
                    step_time += dur_time
                    step_count += 1
                else:
                    batch_offset = i * trainBatchSize
                    batch_end = (i+1) * trainBatchSize
                    X_mini_batch_feed = X_train[batch_offset:batch_end,:,:,:]
                    Y_mini_batch_feed = Y_train[batch_offset:batch_end,:]
                    sess.run(trainStep, feed_dict={features:X_mini_batch_feed, labels:Y_mini_batch_feed})
        
        print(step_time)
        print(step_count)
        print("average step time:", step_time / step_count * 1000)

def execTrainStepPreproc(trainStep, num_epoch, X_train_path, Y_train):
    step_time = 0
    step_count = 0
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    image_list = sorted(os.listdir(X_train_path))
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        num_batch = Y_train.shape[0] // trainBatchSize
        for e in range(num_epoch):
            for i in range(num_batch):
                print('epoch %d / %d, step %d / %d' %(e+1, num_epoch, i+1, num_batch))
                if (i+1) % marker == 0:
                    start_time = timer()
                    batch_offset = i * trainBatchSize
                    batch_end = (i+1) * trainBatchSize
                    batch_list = image_list[batch_offset:batch_end]  
                    X_mini_batch_feed = load_image_dir(X_train_path, batch_list, imgHeight, imgWidth)
                    Y_mini_batch_feed = Y_train[batch_offset:batch_end,:]
                    sess.run(trainStep, feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})
                    end_time = timer()
                    dur_time = end_time - start_time
                    print("step time:",dur_time)
                    step_time += dur_time
                    step_count += 1
                else:
                    batch_offset = i * trainBatchSize
                    batch_end = (i+1) * trainBatchSize
                    batch_list = image_list[batch_offset:batch_end] 
                    X_mini_batch_feed = load_image_dir(X_train_path, batch_list, imgHeight, imgWidth)
                    Y_mini_batch_feed = Y_train[batch_offset:batch_end,:]
                    sess.run(trainStep, feed_dict={features:X_mini_batch_feed, labels:Y_mini_batch_feed})
        print(step_time)
        print(step_count)
        print("average step time:", step_time / step_count * 1000)


def execTrainModel(trainStep, num_epoch, X_train, Y_train):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        num_batch = Y_train.shape[0] // trainBatchSize
        #start_time = timer()
        for e in range(num_epoch):
            for i in range(num_batch):
                print('epoch %d / %d, step %d / %d' %(e+1, num_epoch, i+1, num_batch))
                batch_offset = i * trainBatchSize
                batch_end = (i+1) * trainBatchSize
                X_mini_batch_feed = X_train[batch_offset:batch_end,:,:,:]
                Y_mini_batch_feed = Y_train[batch_offset:batch_end,:]
                sess.run(trainStep, feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})
        #end_time = timer()
        #dur_time = end_time - start_time
        #print("model training time:",dur_time)

def execTrainModelPreproc(trainStep, num_epoch, X_train_path, Y_train):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    image_list = sorted(os.listdir(X_train_path))
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        num_batch = Y_train.shape[0] // trainBatchSize
        #start_time = timer()
        for e in range(num_epoch):
            for i in range(num_batch):
                print('epoch %d / %d, step %d / %d' %(e+1, num_epoch, i+1, num_batch))
                batch_offset = i * trainBatchSize
                batch_end = (i+1) * trainBatchSize
                batch_list = image_list[batch_offset:batch_end]  
                X_mini_batch_feed = load_image_dir(X_train_path, batch_list, imgHeight, imgWidth)
                Y_mini_batch_feed = Y_train[batch_offset:batch_end,:]
                sess.run(trainStep, feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})
        #end_time = timer()
        #dur_time = end_time - start_time
        #print("model training time:",dur_time)

if __name__ == '__main__':
    trainModel = prepareModel()
    printModelDes(trainModel)
    trainStep = buildModel(trainModel)

    if preproc:
        Y_data = load_labels_onehot(label_path, numClasses)
        if trainStepMeasure:
            execTrainStepPreproc(trainStep, numEpochs, image_dir, Y_data)
        else:
            start_time_model = timer()
            execTrainModelPreproc(trainStep, numEpochs, image_dir, Y_data)
            end_time_model = timer()
            dur_time_model = end_time_model - start_time_model
            print("model training time:",dur_time_model)
    else:
        X_data = load_images_bin(bin_dir, numChannels, imgWidth, imgHeight)
        Y_data = load_labels_onehot(label_path, numClasses)
        if trainStepMeasure:
            execTrainStep(trainStep, numEpochs, X_data, Y_data)
        else:
            start_time_model = timer()
            execTrainModel(trainStep, numEpochs, X_data, Y_data)
            end_time_model = timer()
            dur_time_model = end_time_model - start_time_model
            print("model training time:",dur_time_model)
