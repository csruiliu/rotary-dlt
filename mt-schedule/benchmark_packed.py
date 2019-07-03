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
logitCollection = []
scheduleCollection = []
batchCollection = []

features = tf.placeholder(tf.float32, [None, imgWidth, imgHeight, numChannels])
labels = tf.placeholder(tf.int64, [None, numClasses])

bin_dir = '/home/ruiliu/Development/mtml-tf/dataset/imagenet10k.bin'
label_path = '/home/ruiliu/Development/mtml-tf/dataset/imagenet10k-label.txt'
X_data = load_images_bin(bin_dir, numChannels, imgWidth, imgHeight)
Y_data = load_labels_onehot(label_path, numClasses)

def prepareModelsMan():
    #Generate all same models 
    model_class_num = [3]
    model_class = ["mobilenet"]
    all_batch_list = np.repeat(20,3).tolist()
    #all_batch_list = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
    layer_list = np.repeat(1,3).tolist()
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
    for midx in modelCollection:
        modelEntity = midx.getModelEntity()
        modelEntityCollection.append(modelEntity)
        modelLogit = modelEntity.build(features)
        logitUnit = []
        logitUnit.append(modelLogit)
        logitUnit.append(midx.getBatchSize())
        logitCollection.append(logitUnit)

def scheduleNo():
    for lit in logitCollection:
        schUnit = []
        logitUnit = []
        batchUnit = []
        logitUnit.append(lit[0])
        batchUnit.append(lit[1])
        schUnit.append(logitUnit)
        schUnit.append(batchUnit)
        scheduleCollection.append(schUnit)

def schedulePack():
    schUnit = []
    logitUnit = []
    batchUnit = []
    for lit in logitCollection:
        logitUnit.append(lit[0])
    batchUnit.append(logitCollection[0][1])
    schUnit.append(logitUnit)
    schUnit.append(batchUnit)
    scheduleCollection.append(schUnit)

def scheduleMan():
    schUnit1 = []
    logitUnit1 = []
    batchUnit1 = []
    logitUnit1.append(logitCollection[0][0])
    logitUnit1.append(logitCollection[2][0])
    logitUnit1.append(logitCollection[7][0])
    batchUnit1.append(logitCollection[0][1])
    schUnit1.append(logitUnit1)
    schUnit1.append(batchUnit1)
    scheduleCollection.append(schUnit1)

    schUnit2 = []
    logitUnit2 = []
    batchUnit2 = []
    logitUnit2.append(logitCollection[1][0])
    logitUnit2.append(logitCollection[6][0])
    logitUnit2.append(logitCollection[9][0])
    batchUnit2.append(logitCollection[1][1])
    schUnit2.append(logitUnit2)
    schUnit2.append(batchUnit2)
    scheduleCollection.append(schUnit2)

    schUnit3 = []
    logitUnit3 = []
    batchUnit3 = []
    logitUnit3.append(logitCollection[3][0])
    logitUnit3.append(logitCollection[4][0])
    batchUnit3.append(logitCollection[3][1])
    batchUnit3.append(logitCollection[4][1])
    schUnit3.append(logitUnit3)
    schUnit3.append(batchUnit3)
    scheduleCollection.append(schUnit3)

    schUnit4 = []
    logitUnit4 = []
    batchUnit4 = []
    logitUnit4.append(logitCollection[5][0])
    logitUnit4.append(logitCollection[8][0])
    batchUnit4.append(logitCollection[5][1])
    batchUnit4.append(logitCollection[8][1])
    schUnit4.append(logitUnit4)
    schUnit4.append(batchUnit4)
    scheduleCollection.append(schUnit4)

def executeSch(sch_unit, batch_unit, num_epoch, X_train, Y_train):
    total_time = 0
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    if len(batch_unit) == 1:
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            mini_batches = batch_unit[0]
            num_batch = Y_train.shape[0] // mini_batches
            num_batch_list = np.arange(num_batch)
            for e in range(num_epoch):
                for i in range(num_batch):
                    print('epoch %d / %d, step %d / %d' %(e+1, num_epoch, i+1, num_batch))
                    #print(datetime.datetime.now())
                    if isShuffle:
                        rand_idx = int(np.random.choice(num_batch_list, 1, replace=False))
                        #print(rand_idx)
                        X_mini_batch_feed = X_train[rand_idx:rand_idx + mini_batches,:,:,:]
                        Y_mini_batch_feed = Y_train[rand_idx:rand_idx + mini_batches,:]
                    else:
                        #print(i)
                        X_mini_batch_feed = X_train[num_batch:num_batch + mini_batches,:,:,:]
                        Y_mini_batch_feed = Y_train[num_batch:num_batch + mini_batches,:]
                    #print(datetime.datetime.now())
                    #start_time = timer()
                    sess.run(sch_unit, feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})
                    #print(datetime.datetime.now())
                    #end_time = timer()
                    #total_time += end_time - start_time
                    #print("training time for 1 epoch:", total_time)

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
                        print(timer())
                        if isShuffle:
                            rand_idx = np.random.choice(num_batch_list, 1, replace=False)
                            X_mini_batch_feed = X_train[rand_idx:rand_idx + mini_batches,:,:,:]
                            Y_mini_batch_feed = Y_train[rand_idx:rand_idx + mini_batches,:]
                        else:
                            X_mini_batch_feed = X_train[num_batch:num_batch + mini_batches,:,:,:]
                            Y_mini_batch_feed = Y_train[num_batch:num_batch + mini_batches,:]
                        print(timer())
                        #start_time = timer()
                        sess.run(sch_unit[idx], feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})
                        print(timer())
                        #end_time = timer()
                        #total_time += end_time - start_time
                        #print("training time for 1 epoch:", total_time)


if __name__ == '__main__':
    print("run gpu:", gpuId)
    print("training epochs:", numEpochs)
    print("shuffle input:", isShuffle)
    print("input image width:", imgWidth)
    print("input image height", imgHeight)
    print("prediction classes:", numClasses)
    print("channel of input images:", numChannels)
    deviceId = '/device:GPU:' + str(gpuId)
    with tf.device(deviceId):
        #print(isShuffle)
        prepareModelsMan()
        printAllModels()
        buildModels()
        schedulePack()
        #scheduleMan()
        for idx, sit in enumerate(scheduleCollection):
            p = Process(target=executeSch, args=(sit[0], sit[1], numEpochs, X_data, Y_data,))
            p.start()
            print(p.pid)
            p.join()






