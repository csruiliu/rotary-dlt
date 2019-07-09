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
parser.add_argument('-iw', '--imgw', type=int, default=224, help='identify the weight of img')
parser.add_argument('-ih', '--imgh', type=int, default=224, help='identify the height of img')
parser.add_argument('-cls', '--clazz', type=int, default=1000, help='predication classes')
parser.add_argument('-ch', '--channel', type=int, default=3, help='identify the channel of input images')
parser.add_argument('-e', '--epoch', type=int, default=1, help='identify the epoch numbers')
parser.add_argument('-s', '--shuffle', action='store_true', default=False, help='use shuffle batch input or not')
args = parser.parse_args()

gpuId = args.gpuid
imgWidth = args.imgw
imgHeight = args.imgh
numClasses = args.clazz
numChannels = args.channel
numEpochs = args.epoch
isShuffle = args.shuffle

modelCollection = []
modelEntityCollection = []
trainCollection = []
scheduleCollection = []
batchCollection = []

input_model_num = 3

#features1 = tf.placeholder(tf.float32, [None, imgWidth, imgHeight, numChannels])
#labels1 = tf.placeholder(tf.int64, [None, numClasses])
#features2 = tf.placeholder(tf.float32, [None, imgWidth, imgHeight, numChannels])
#labels2 = tf.placeholder(tf.int64, [None, numClasses])

#bin_dir = '/home/ruiliu/Development/mtml-tf/dataset/imagenet1k.bin'
#label_path = '/home/ruiliu/Development/mtml-tf/dataset/imagenet1k-label.txt'
bin_dir = '/tank/local/ruiliu/imagenet1k.bin'
label_path = '/tank/local/ruiliu/imagenet1k-label.txt'
X_data = load_images_bin(bin_dir, numChannels, imgWidth, imgHeight)
Y_data = load_labels_onehot(label_path, numClasses)

names = locals()
input_dict = {} 

for idx in range(input_model_num):
    names['features' + str(idx)] = tf.placeholder(tf.float32, [None, imgWidth, imgHeight, numChannels])
    names['labels' + str(idx)] = tf.placeholder(tf.int64, [None, numClasses])
    
    #name='features'+str(idx)
    #locals()['features'+str(idx)] = tf.placeholder(tf.float32, [None, imgWidth, imgHeight, numChannels])
    #name='labels'+str(idx)
    #locals()['labels'+str(idx)] = tf.placeholder(tf.int64, [None, numClasses])

def prepareModelsMan():
    #Generate all same models 
    model_class_num = [input_model_num]
    model_class = ["mobilenet"]
    all_batch_list = np.repeat(10,input_model_num).tolist()
    #all_batch_list = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
    layer_list = np.repeat(1,input_model_num).tolist()
    #layer_list = np.random.choice(np.arange(3,10), 9).tolist()
    #layer_list = [5, 2, 8, 4, 9, 10, 3, 7, 1, 4,2,8,4,3,11]
    model_name_abbr = np.random.choice(100000, sum(model_class_num), replace=False).tolist()
    for idx, mls in enumerate(model_class):
        for _ in range(model_class_num[idx]):
            dm = DnnModel(mls, str(model_name_abbr.pop()), model_layer=layer_list.pop(), input_w=imgWidth, input_h=imgHeight, num_classes=numClasses, batch_size=all_batch_list.pop(), desired_accuracy=0.9)
            modelCollection.append(dm)

    #modelClass = ["resnet", "mobilenet", "perceptron", "convnet"]
    #modelNum = [1,1,1,1]
    #all_batch_list = [20, 20, 20, 20]
    #layer_list = [4, 8, 1, 1]
    #modelNameAddr = np.random.choice(100000, 4, replace=False).tolist()
    #dm1 = DnnModel("mobilenet", str(modelNameAddr.pop()), model_layer=1, input_w=imgWidth, input_h=imgHeight, num_classes=numClasses, batch_size=10, desired_accuracy=0.9)
    #dm2 = DnnModel("resnet",str(modelNameAddr.pop()), model_layer=1, input_w=imgWidth, input_h=imgHeight, num_classes=numClasses, batch_size=10, desired_accuracy=0.9)
    #modelCollection.append(dm1)
    #modelCollection.append(dm2)

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

def buildModels():
    for midx, mc in enumerate(modelCollection):
        modelEntity = mc.getModelEntity()
        modelEntityCollection.append(modelEntity)
        modelLogit = modelEntity.build(names['features' + str(midx)])
        modelTrain = modelEntity.cost(modelLogit, names['labels' + str(midx)])
        trainUnit = []
        trainUnit.append(modelTrain)
        trainUnit.append(mc.getBatchSize())
        trainCollection.append(trainUnit)

def schedulePack():
    schUnit = []
    trainUnit = []
    batchUnit = []
    for tit in trainCollection:
        trainUnit.append(tit[0])
    batchUnit.append(trainCollection[0][1])
    schUnit.append(trainUnit)
    schUnit.append(batchUnit)
    scheduleCollection.append(schUnit)

def executeSch(sch_unit, batch_unit, num_epoch, X_train, Y_train):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    if len(batch_unit) == 1:
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            mini_batches = batch_unit[0]
            num_batch = Y_train.shape[0] // mini_batches
            num_batch_list = np.arange(num_batch)
            for e in range(num_epoch):
                for i in range(num_batch):
                    print('epoch %d / %d, step %d / %d' %(e+1, num_epoch, i+1, num_batch))
                    if isShuffle:
                        rand_idx = int(np.random.choice(num_batch_list, 1, replace=False))
                        #print(rand_idx)
                        X_mini_batch_feed = X_train[rand_idx:rand_idx+mini_batches,:,:,:]
                        Y_mini_batch_feed = Y_train[rand_idx:rand_idx+mini_batches,:]
                    else:
                        #print(i)
                        #X_mini_batch_feed1 = X_train[i:i+mini_batches,:,:,:]
                        #Y_mini_batch_feed1 = Y_train[i:i+mini_batches,:]
                        
                        for ridx in range(input_model_num):
                            rand_idx = int(np.random.choice(num_batch_list, 1, replace=False))
                            names['X_mini_batch_feed' + str(ridx)] = X_train[rand_idx:rand_idx+mini_batches,:,:,:]
                            names['Y_mini_batch_feed' + str(ridx)] = Y_train[rand_idx:rand_idx+mini_batches,:]
                            input_dict[names['features' + str(ridx)]] = names['X_mini_batch_feed' + str(ridx)]
                            input_dict[names['labels' + str(ridx)]] = names['Y_mini_batch_feed' + str(ridx)]
                        sess.run(sch_unit, feed_dict=input_dict)
                        
                        #sess.run(sch_unit, feed_dict={features0: X_mini_batch_feed1, labels0: Y_mini_batch_feed1, features1: X_mini_batch_feed2, labels1: Y_mini_batch_feed2})
    else:
        for idx, batch in enumerate(batch_unit):
            with tf.Session(config=config) as sess:
                sess.run(tf.global_variables_initializer())
                mini_batches = batch
                num_batch = Y_train.shape[0] // mini_batches
                num_batch_list = np.arange(num_batch)
                for e in range(num_epoch):
                    for i in range(num_batch):
                        print('epoch %d / %d, step %d / %d' %(e+1, num_epoch, i+1, num_batch))
                        if isShuffle:
                            rand_idx = np.random.choice(num_batch_list, 1, replace=False)
                            X_mini_batch_feed = X_train[rand_idx:rand_idx+mini_batches,:,:,:]
                            Y_mini_batch_feed = Y_train[rand_idx:rand_idx+mini_batches,:]
                        else:
                            X_mini_batch_feed = X_train[i:i+mini_batches,:,:,:]
                            Y_mini_batch_feed = Y_train[i:i+mini_batches,:]
                        #sess.run(sch_unit[idx], feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})


if __name__ == '__main__':
    deviceId = '/device:GPU:' + str(gpuId)
    print("run gpu:", deviceId)
    print("training epochs:", numEpochs)
    print("shuffle input:", isShuffle)
    print("input image width:", imgWidth)
    print("input image height", imgHeight)
    print("prediction classes:", numClasses)
    print("channel of input images:", numChannels)
    with tf.device(deviceId):
        prepareModelsMan()

        printAllModels()
        buildModels()
        schedulePack()
        for idx, sit in enumerate(scheduleCollection):
            p = Process(target=executeSch, args=(sit[0], sit[1], numEpochs, X_data, Y_data,))
            p.start()
            print(p.pid)
            p.join()

