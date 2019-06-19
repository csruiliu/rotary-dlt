import tensorflow as tf
from dnn_model import DnnModel

from img_utils import *
from cifar_utils import *
from tfrecord_utils import *
from img2bin_utils import *

import numpy as np
from datetime import datetime
from multiprocessing import Process
from timeit import default_timer as timer

import os

img_w = 224
img_h = 224
num_channel = 3
num_classes = 1000

modelCollection = []
modelEntityCollection = []
logitCollection = []
scheduleCollection = []
batchCollection = []

features = tf.placeholder(tf.float32, [None, img_w, img_h, num_channel])
labels = tf.placeholder(tf.int64, [None, num_classes])

#cifar10_dir = '/home/ruiliu/Development/mtml-tf/dataset/cifar-10'
#cifar = cifar_utils(cifar10_dir)
#X_data, _, Y_data = cifar.load_evaluation_data()

#data_dir = '/home/ruiliu/Development/mtml-tf/dataset/test'
#label_path = '/home/ruiliu/Development/mtml-tf/dataset/test-label.txt'
#X_data = load_images(data_dir)
#Y_data = load_labels_onehot(label_path)

bin_dir = '/home/ruiliu/Development/mtml-tf/dataset/imagenet10k.bin'
label_path = '/home/ruiliu/Development/mtml-tf/dataset/imagenet10k-label.txt'
X_data = unpickle_load_images(bin_dir, num_channel, img_w, img_h)
Y_data = load_labels_onehot(label_path)


def prepareModelsFix():   
    model_name_abbr = np.random.choice(100000, 4, replace=False).tolist()

    dm1 = DnnModel("mobilenet", str(model_name_abbr.pop()), model_layer=1, input_w=img_w, input_h=img_h,  
                            num_classes=num_classes, batch_size=40, desired_accuracy=0.9)
    dm2 = DnnModel("resnet", str(model_name_abbr.pop()), model_layer=1, input_w=img_w, input_h=img_h,  
                            num_classes=num_classes, batch_size=20, desired_accuracy=0.9)
    
    modelCollection.append(dm1)
    modelCollection.append(dm2)
    
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


def executeSch(sch_unit, batch_unit, X_train, Y_train):
    total_time = 0
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.log_device_placement = True
    if len(batch_unit) == 1:
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            mini_batches = batch_unit[0]    
            num_batch = Y_train.shape[0] // mini_batches            
            for i in range(num_batch):
                print('step %d / %d' %(i+1, num_batch))
                X_mini_batch_feed = X_train[num_batch:num_batch + mini_batches,:,:,:]
                Y_mini_batch_feed = Y_train[num_batch:num_batch + mini_batches,:]
                start_time = timer()
                sess.run(sch_unit, feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})
                end_time = timer()
                total_time += end_time - start_time
                print("training time for 1 epoch:", total_time)  

    else:
        for idx, batch in enumerate(batch_unit):
            with tf.Session(config=config) as sess:
                sess.run(tf.global_variables_initializer())
                mini_batches = batch    
                num_batch = Y_train.shape[0] // mini_batches            
                for i in range(num_batch):
                    print('step %d / %d' %(i+1, num_batch))
                    X_mini_batch_feed = X_train[num_batch:num_batch + mini_batches,:,:,:]
                    Y_mini_batch_feed = Y_train[num_batch:num_batch + mini_batches,:]
                    start_time = timer()
                    sess.run(sch_unit[idx], feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})
                    end_time = timer()
                    total_time += end_time - start_time
                    print("training time for 1 epoch:", total_time)  


if __name__ == '__main__':
    prepareModelsFix()
    printAllModels()
    #saveModelDes()
    buildModels()
    scheduleNo()
    #scheduleGreedy()

    for sit in scheduleCollection:
        p = Process(target=executeSch, args=(sit[0], sit[1], X_data, Y_data,))
        p.start()
        print(p.pid)
        p.join()
