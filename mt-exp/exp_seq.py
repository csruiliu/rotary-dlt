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
parser.add_argument('-m', '--samemodel', action='store_false', default=True, help='pack same model to train or not')
parser.add_argument('-p', '--preproc', action='store_false', default=True, help='use preproc to transform the data before training or not')
parser.add_argument('-d', '--samedata', action='store_false', default=True, help='use same training batch data or not')
parser.add_argument('-o', '--sameoptimizer', action='store_false', default=True, help='use same optimizer or not')
parser.add_argument('-b', '--samebatchsize', action='store_false', default=True, help='use same batch size or not')
parser.add_argument('-t', '--trainstep', action='store_true', default=False, help='use same compute and apply to update gradient or not')
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

remark = 3

input_model_num = 2

sameBatchSize = args.samebatchsize
sameModel = args.samemodel
preproc = args.preproc
sameTrainData = args.samedata
sameOptimizer = args.sameoptimizer
trainStep = args.trainstep

if sameBatchSize:
    maxBatchSize = 32
else:
    maxBatchSize = 50

if sameTrainData:
    features = tf.placeholder(tf.float32, [None, imgWidth, imgHeight, numChannels])
    labels = tf.placeholder(tf.int64, [None, numClasses])
else:
    names = locals()
    input_dict = {}
    for idx in range(input_model_num):
        names['features' + str(idx)] = tf.placeholder(tf.float32, [None, imgWidth, imgHeight, numChannels])
        names['labels' + str(idx)] = tf.placeholder(tf.int64, [None, numClasses])

def showExpConfig():
    print("Packing the same model:", sameModel)
    print("Using the same train data:", sameModel)
    print("Using the preprocess:", preproc)
    print("Using the same optimizer:", sameOptimizer)
    print("Using the same batch size:", sameBatchSize)

def prepareModels():
    modelCollection = []
    
    if sameModel:
        model_class_num = [1,1]
        model_class = ["resnet","resnet"]       
    else:
        model_class_num = [1,1]
        model_class = ["resnet","mobilenet"]
    
    if sameBatchSize:
        batch_list = [32,32]
    else:
        batch_list = [32,50]

    if sameOptimizer:
        opt_list = ["Adam","Adam"]
    else:
        opt_list = ["Adam","SGD"]

    model_name_abbr = np.random.choice(100000, sum(model_class_num), replace=False).tolist()

    for idx, mls in enumerate(model_class):
        dm = DnnModel(mls, str(model_name_abbr.pop()), model_layer=1, 
                        input_w=imgWidth, input_h=imgHeight, num_classes=numClasses, 
                        batch_size=batch_list[idx], optimizer=opt_list[idx])
        modelCollection.append(dm)
    return modelCollection

def printAllModels(model_collection):
    print("Max Batch Size:",maxBatchSize)
    for idm in model_collection:
        print(idm.getInstanceName())
        print(idm.getModelName())
        print(idm.getModelLayer())
        print(idm.getBatchSize())
        print("===============")

def buildModels(model_collection):
    modelEntityCollection = []
    trainCollection = []
    for midx, mc in enumerate(model_collection):
        modelEntity = mc.getModelEntity()
        modelEntityCollection.append(modelEntity)
        if sameTrainData:
            modelLogit = modelEntity.build(features)
            if trainStep:
                trainOptimizer, trainGradsVars, trainOps = modelEntity.train_step(modelLogit, labels)
            else:
                trainOps = modelEntity.train(modelLogit, labels)
        else:
            modelLogit = modelEntity.build(names['features' + str(midx)])
            if trainStep:
                trainOptimizer, trainGradsVars, trainOps = modelEntity.train_step(modelLogit, names['labels' + str(midx)])
            else:
                trainOps = modelEntity.train(modelLogit, labels)
        trainCollection.append(trainOps)
    return trainCollection

def execTrain(unit, num_epoch, X_train, Y_train):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        num_batch = Y_train.shape[0] // maxBatchSize
        num_batch_list = np.arange(num_batch)
        for e in range(num_epoch):
            for i in range(num_batch):
                print('epoch %d / %d, step %d / %d' %(e+1, num_epoch, i+1, num_batch))
                if sameTrainData:    
                    batch_offset = i * maxBatchSize
                    batch_end = (i+1) * maxBatchSize
                    X_mini_batch_feed = X_train[batch_offset:batch_end,:,:,:]
                    Y_mini_batch_feed = Y_train[batch_offset:batch_end,:]
                    sess.run(unit, feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})
                else:
                    for ridx in range(input_model_num):
                        rand_idx = int(np.random.choice(num_batch_list, 1))
                        batch_offset = rand_idx * maxBatchSize
                        batch_end = (rand_idx+1) * maxBatchSize
                        names['X_mini_batch_feed' + str(ridx)] = X_train[batch_offset:batch_end,:,:,:]
                        names['Y_mini_batch_feed' + str(ridx)] = Y_train[batch_offset:batch_end,:]
                        input_dict[names['features' + str(ridx)]] = names['X_mini_batch_feed' + str(ridx)]
                        input_dict[names['labels' + str(ridx)]] = names['Y_mini_batch_feed' + str(ridx)]
                    sess.run(unit, feed_dict=input_dict)

def execTrainPreproc(unit, num_epoch, X_train_path, Y_train):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    image_list = sorted(os.listdir(X_train_path))
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        num_batch = Y_train.shape[0] // maxBatchSize
        num_batch_list = np.arange(num_batch)
        for e in range(num_epoch):
            for i in range(num_batch):
                print('epoch %d / %d, step %d / %d' %(e+1, num_epoch, i+1, num_batch))
                if sameTrainData:       
                    batch_offset = i * maxBatchSize
                    batch_end = (i+1) * maxBatchSize
                    batch_list = image_list[batch_offset:batch_end]   
                    X_mini_batch_feed = load_image_dir(X_train_path, batch_list, imgHeight, imgWidth)
                    Y_mini_batch_feed = Y_train[batch_offset:batch_end,:]
                    sess.run(unit, feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})
                else:
                    for ridx in range(input_model_num):
                        rand_idx = int(np.random.choice(num_batch_list, 1))
                        batch_offset = rand_idx * maxBatchSize
                        batch_end = (rand_idx+1) * maxBatchSize
                        batch_list = image_list[batch_offset:batch_end]
                        names['X_mini_batch_feed' + str(ridx)] = load_image_dir(X_train_path, batch_list, imgHeight, imgWidth)
                        names['Y_mini_batch_feed' + str(ridx)] = Y_train[batch_offset:batch_end,:]
                        input_dict[names['features' + str(ridx)]] = names['X_mini_batch_feed' + str(ridx)]
                        input_dict[names['labels' + str(ridx)]] = names['Y_mini_batch_feed' + str(ridx)]
                    sess.run(unit, feed_dict=input_dict)     

if __name__ == '__main__':
    showExpConfig()    
    modelCollection = prepareModels()
    printAllModels(modelCollection)
    trainCollection = buildModels(modelCollection)
    if preproc:
        X_data = load_images_bin(bin_dir, numChannels, imgWidth, imgHeight)
        Y_data = load_labels_onehot(label_path, numClasses)
        start_time = timer()
        for tidx in trainCollection:
            p = Process(target=execTrain, args=(tidx, numEpochs, X_data, Y_data,))
            p.start()
            print(p.pid)
            p.join()
        end_time = timer()
        dur_time = end_time - start_time
        print("overall training time", dur_time)
    else:
        Y_data = load_labels_onehot(label_path, numClasses)
        for tidx in trainCollection:
            p = Process(target=execTrainPreproc, args=(tidx, numEpochs, image_dir, Y_data,))
            p.start()
            print(p.pid)
            p.join()
        end_time = timer()
        dur_time = end_time - start_time
        print("overall training time", dur_time)
