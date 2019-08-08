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
parser.add_argument('-bs', '--batchsize', type=int, default=32, help='identify the training batch size')
parser.add_argument('-e', '--epoch', type=int, default=1, help='identify the epoch numbers')
args = parser.parse_args()

#########################
# Global Variables
#########################

#image_dir = '/home/ruiliu/Development/mtml-tf/dataset/imagenet10k'
image_dir = '/tank/local/ruiliu/dataset/imagenet10k'

#bin_dir = '/home/ruiliu/Development/mtml-tf/dataset/imagenet1k.bin'
bin_dir = '/tank/local/ruiliu/dataset/imagenet10k.bin'

#label_path = '/home/ruiliu/Development/mtml-tf/dataset/imagenet1k-label.txt'
label_path = '/tank/local/ruiliu/dataset/imagenet10k-label.txt'

#profile_dir = '/home/ruiliu/Development/mtml-tf/mt-perf/profile_dir'
#profile_dir = '/tank/local/ruiliu/mtml-tf/mt-perf/profile_dir'

imgWidth = 224
imgHeight = 224
numClasses = 1000
numChannels = 3
maxBatchSize = args.batchsize
numEpochs = args.epoch

input_model_num = 1

features = tf.placeholder(tf.float32, [None, imgWidth, imgHeight, numChannels])
labels = tf.placeholder(tf.int64, [None, numClasses])

def prepareModelsMan():
    modelCollection = []
    model_class_num = [2]  
    model_class = ["resnet"]
    all_batch_list = [32,32]
    layer_list = np.repeat(1, sum(model_class_num)).tolist()
    model_name_abbr = np.random.choice(100000, sum(model_class_num), replace=False).tolist()
    for idx, mls in enumerate(model_class):
        for _ in range(model_class_num[idx]):
            dm = DnnModel(mls, str(model_name_abbr.pop()), model_layer=layer_list.pop(), input_w=imgWidth, input_h=imgHeight, num_classes=numClasses, batch_size=all_batch_list.pop(),optimizer="Adam")
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
    trainGradsCollection = []
    trainOpsCollection = []
    for mc in model_collection:
        modelEntity = mc.getModelEntity()
        modelLogit = modelEntity.build(features)
        trainOps = modelEntity.train(modelLogit, labels)
        #trainOptimizer, trainGradsVars, trainOps = modelEntity.train_step(modelLogit, labels)
        #trainCollection.append(trainGradsVars)
        #train_vars_with_grad = [v for g, v in trainGradsVars if g is not None]
        #trainGradsCollection.append(train_vars_with_grad)
        #trainGradsCollection.append(trainGradsVars)
        #with tf.control_dependencies([trainOps]):
        #    dummy = tf.constant(0)
        trainOpsCollection.append(trainOps)
    return trainOpsCollection
    #return trainGradsCollection, trainOpsCollection

def execTrain(trainOpsUnit, num_epoch, X_train_path, Y_train):
    step_time = 0
    step_count = 0
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    image_list = sorted(os.listdir(X_train_path))
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        num_batch = Y_train.shape[0] // maxBatchSize
        for e in range(num_epoch):
            for i in range(num_batch):
                print('epoch %d / %d, step %d / %d' %(e+1, num_epoch, i+1, num_batch))
                #h = sess.partial_run_setup([trainGradsUnit,trainOpsUnit],[features, labels])
                if (i+1) % 10 == 0:
                    start_time = timer()
                    batch_offset = i * maxBatchSize
                    batch_end = (i+1) * maxBatchSize
                    batch_list = image_list[batch_offset:batch_end]  
                    X_mini_batch_feed = load_image_dir(X_train_path, batch_list, imgHeight, imgWidth)
                    #X_mini_batch_feed = X_train[batch_offset:batch_end,:,:,:]
                    Y_mini_batch_feed = Y_train[batch_offset:batch_end,:]
                    sess.run(trainOpsUnit,feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})
                    #h = sess.partial_run_setup([trainGradsUnit,trainOpsUnit],[features, labels]) 
                    #sess.partial_run(h,trainGradsUnit,feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})
                    #sess.partial_run(h,trainOpsUnit)
                    
                    #sess.partial_run(h,[trainGradsUnit[0],trainGradsUnit[1]],feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})
                    #sess.partial_run(h,trainGradsUnit,feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})
                    #sess.partial_run(h,trainOpsUnit)
                    end_time = timer()
                    dur_time = end_time - start_time
                    print("step time:",dur_time)
                    step_time += dur_time
                    step_count += 1
                else:
                    batch_offset = i * maxBatchSize
                    batch_end = (i+1) * maxBatchSize
                    batch_list = image_list[batch_offset:batch_end] 
                    X_mini_batch_feed = load_image_dir(X_train_path, batch_list, imgHeight, imgWidth)
                    #X_mini_batch_feed = X_train[batch_offset:batch_end,:,:,:]
                    Y_mini_batch_feed = Y_train[batch_offset:batch_end,:]
                    #h = sess.partial_run_setup([trainGradsUnit,trainOpsUnit],[features, labels]) 
                    #sess.partial_run(h,trainGradsUnit,feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})
                    #sess.partial_run(h,trainOpsUnit)                  
                    #sess.partial_run(h,[trainGradsUnit[0],trainGradsUnit[1]],feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})
                    #sess.partial_run(h,trainGradsUnit,feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})
                    #sess.partial_run(h,trainOpsUnit)
                    sess.run(trainOpsUnit, feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})
                
        print(step_time)
        print(step_count)
        print("average step time:", step_time / step_count * 1000)


if __name__ == '__main__':
    #X_data = load_images_bin(bin_dir, numChannels, imgWidth, imgHeight)
    Y_data = load_labels_onehot(label_path, numClasses)
    modelCollection = prepareModelsMan()
    printAllModels(modelCollection)
    trainOpsCollection = buildModels(modelCollection)
    execTrain(trainOpsCollection, numEpochs, image_dir, Y_data)
