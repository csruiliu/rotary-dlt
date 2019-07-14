# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from multiprocessing import Process

from img_utils import *
from dnn_model import DnnModel
from timeit import default_timer as timer
import datetime

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gpuid', type=int, default=0, help='identify a GPU to run')
parser.add_argument('-bs', '--batchsize', type=int, default=10, help='identify the training batch size')
parser.add_argument('-iw', '--imgw', type=int, default=224, help='identify the weight of img')
parser.add_argument('-ih', '--imgh', type=int, default=224, help='identify the height of img')
parser.add_argument('-cls', '--clazz', type=int, default=1000, help='predication classes')
parser.add_argument('-ch', '--channel', type=int, default=3, help='identify the channel of input images')
parser.add_argument('-e', '--epoch', type=int, default=1, help='identify the epoch numbers')
parser.add_argument('-s', '--shuffle', action='store_true', default=False, help='use shuffle the batch input or not, default is sequential, if use different batch size, then this config will be ignored')
parser.add_argument('-d', '--diff', action='store_true', default=False, help='use different batch input for each model in the packed api, default is all the models in packed use all input batch')
args = parser.parse_args()

gpuId = args.gpuid
batchSize = args.batchsize
imgWidth = args.imgw
imgHeight = args.imgh
numClasses = args.clazz
numChannels = args.channel
numEpochs = args.epoch
isShuffle = args.shuffle
isDiffernetBatch = args.diff

modelCollection = []
modelEntityCollection = []
trainCollection = []
scheduleCollection = []
batchCollection = []

input_model_num = 5

#bin_dir = '/home/ruiliu/Development/mtml-tf/dataset/imagenet1k.bin'
#label_path = '/home/ruiliu/Development/mtml-tf/dataset/imagenet1k-label.txt'
bin_dir = '/tank/local/ruiliu/dataset/imagenet10k.bin'
label_path = '/tank/local/ruiliu/dataset/imagenet10k-label.txt'
X_data = load_images_bin(bin_dir, numChannels, imgWidth, imgHeight)
Y_data = load_labels_onehot(label_path, numClasses)

if isDiffernetBatch:
    names = locals()
    input_dict = {}
    for idx in range(input_model_num):
        names['features' + str(idx)] = tf.placeholder(tf.float32, [None, imgWidth, imgHeight, numChannels])
        names['labels' + str(idx)] = tf.placeholder(tf.int64, [None, numClasses])
else:
    features = tf.placeholder(tf.float32, [None, imgWidth, imgHeight, numChannels])
    labels = tf.placeholder(tf.int64, [None, numClasses])


def prepareModelsMan():
    #Generate all same models 
    model_class_num = [input_model_num]
    model_class = ["mobilenet"]
    all_batch_list = np.repeat(batchSize,input_model_num).tolist()
    layer_list = np.repeat(1,input_model_num).tolist()
    #layer_list = np.random.choice(np.arange(3,10), 9).tolist()
    #layer_list = [5, 2, 8, 4, 9, 10, 3, 7, 1, 4,2,8,4,3,11]
    model_name_abbr = np.random.choice(100000, sum(model_class_num), replace=False).tolist()
    for idx, mls in enumerate(model_class):
        for _ in range(model_class_num[idx]):
            dm = DnnModel(mls, str(model_name_abbr.pop()), model_layer=layer_list.pop(), input_w=imgWidth, input_h=imgHeight, num_classes=numClasses, batch_size=all_batch_list.pop(), desired_accuracy=0.9)
            modelCollection.append(dm)

def printAllModels():
    for idm in modelCollection:
        print(idm.getInstanceName())
        print(idm.getModelName())
        print(idm.getModelLayer())
        print(idm.getBatchSize())
        print("===============")

def saveModelDes():
    with open('models_des.txt', 'w') as file:
        file.write("Workload:\n")
        for idx in modelCollection:
            modelName = idx.getModelName()
            modelLayer = idx.getModelLayer()
            instanceName = idx.getInstanceName()
            batchSize = idx.getBatchSize()
            file.write('model type:' + modelName + '\n')
            file.write('instance name:' + instanceName + '\n')
            file.write('layer num: ' + str(modelLayer) + '\n')
            file.write('input batch size: '+ str(batchSize)+ '\n')
            file.write('=======================\n')

def buildPackedModels():
    for midx, mc in enumerate(modelCollection):
        modelEntity = mc.getModelEntity()
        modelEntityCollection.append(modelEntity)
        if isDiffernetBatch:
            modelLogit = modelEntity.build(names['features' + str(midx)])
            modelTrain = modelEntity.cost(modelLogit, names['labels' + str(midx)])
        else:
            modelLogit = modelEntity.build(features)
            modelTrain = modelEntity.cost(modelLogit, labels)
        trainCollection.append(modelTrain)

def buildPackedModelsCombine():
    combineTrain = 0
    for midx, mc in enumerate(modelCollection):
        modelEntity = mc.getModelEntity()
        modelLogit = modelEntity.build(features)
        modelTrain = modelEntity.getCost(modelLogit, labels)
        combineTrain += modelTrain

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(combineTrain)

    return train_step


def executePack(train_collection, num_epoch, X_train, Y_train):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        num_batch = Y_train.shape[0] // batchSize
        num_batch_list = np.arange(num_batch)
        for e in range(num_epoch):
            for i in range(num_batch):
                print('epoch %d / %d, step %d / %d' %(e+1, num_epoch, i+1, num_batch))
                if isDiffernetBatch:
                    for ridx in range(input_model_num):
                        rand_idx = int(np.random.choice(num_batch_list, 1))
                        names['X_mini_batch_feed' + str(ridx)] = X_train[rand_idx:rand_idx+batchSize,:,:,:]
                        names['Y_mini_batch_feed' + str(ridx)] = Y_train[rand_idx:rand_idx+batchSize,:]
                        input_dict[names['features' + str(ridx)]] = names['X_mini_batch_feed' + str(ridx)]
                        input_dict[names['labels' + str(ridx)]] = names['Y_mini_batch_feed' + str(ridx)]
                    sess.run(train_collection, feed_dict=input_dict)
                else:
                    if isShuffle:
                        rand_idx = int(np.random.choice(num_batch_list, 1, replace=False))
                        X_mini_batch_feed = X_train[rand_idx:rand_idx+batchSize,:,:,:]
                        Y_mini_batch_feed = Y_train[rand_idx:rand_idx+batchSize,:]
                    else:
                        X_mini_batch_feed = X_train[i:i+batchSize,:,:,:]
                        Y_mini_batch_feed = Y_train[i:i+batchSize,:]
                    sess.run(train_collection, feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})


def executePackCombine(train_collection_combine, num_epoch, X_train, Y_train):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        num_batch = Y_train.shape[0] // batchSize
        num_batch_list = np.arange(num_batch)
        for e in range(num_epoch):
            for i in range(num_batch):
                print('epoch %d / %d, step %d / %d' %(e+1, num_epoch, i+1, num_batch))
                if isShuffle:
                    rand_idx = int(np.random.choice(num_batch_list, 1, replace=False))
                    X_mini_batch_feed = X_train[rand_idx:rand_idx+batchSize,:,:,:]
                    Y_mini_batch_feed = Y_train[rand_idx:rand_idx+batchSize,:]
                else:
                    X_mini_batch_feed = X_train[i:i+batchSize,:,:,:]
                    Y_mini_batch_feed = Y_train[i:i+batchSize,:]
                sess.run(train_collection_combine, feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})


if __name__ == '__main__':
    deviceId = '/device:GPU:' + str(gpuId)
    print("run gpu:", deviceId)
    print("training batch size", batchSize)
    print("training epochs:", numEpochs)
    print("is shuffle input:", isShuffle)
    print("is different batch input:", isDiffernetBatch)
    print("input image width:", imgWidth)
    print("input image height", imgHeight)
    print("prediction classes:", numClasses)
    print("channel of input images:", numChannels)
    with tf.device(deviceId):
        prepareModelsMan()
        printAllModels()
        buildPackedModels()
        executePack(trainCollection, numEpochs, X_data, Y_data)
        #trainStep=buildPackedModelsCombine()
        #executePackCombine(trainStep, numEpochs, X_data, Y_data)    
