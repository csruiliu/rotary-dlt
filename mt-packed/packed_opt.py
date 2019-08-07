import tensorflow as tf
#from tensorflow.python.profiler import model_analyzer
#from tensorflow.python.profiler import option_builder
from tensorflow.python.client import timeline
import numpy as np
import argparse
from timeit import default_timer as timer

from img_utils import *
from dnn_model import DnnModel


#########################
# Command Parameters
#########################

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
parser.add_argument('-p', '--profile', action='store_true', default=False, help='use tf.profiler or not')
args = parser.parse_args()

#########################
# Global Variables
#########################

#image_dir = '/home/ruiliu/Development/mtml-tf/dataset/imagenet10k'
#image_dir = '/tank/local/ruiliu/dataset/imagenet10k'
#bin_dir = '/home/ruiliu/Development/mtml-tf/dataset/imagenet1k.bin'
bin_dir = '/tank/local/ruiliu/dataset/imagenet10k.bin'
#label_path = '/home/ruiliu/Development/mtml-tf/dataset/imagenet1k-label.txt'
label_path = '/tank/local/ruiliu/dataset/imagenet10k-label.txt'
#profile_dir = '/home/ruiliu/Development/mtml-tf/mt-perf/profile_dir'
profile_dir = '/tank/local/ruiliu/mtml-tf/mt-perf/profile_dir'

gpuId = args.gpuid
maxBatchSize = args.batchsize
imgWidth = args.imgw
imgHeight = args.imgh
numClasses = args.clazz
numChannels = args.channel
numEpochs = args.epoch
isShuffle = args.shuffle
isDiffernetBatch = args.diff
isProfile = args.profile

features = tf.placeholder(tf.float32, [None, imgWidth, imgHeight, numChannels])
labels = tf.placeholder(tf.int64, [None, numClasses])

def prepareModelsMan():
    modelCollection = []
    model_class_num = [1,1]
    model_class = ["mobilenet"]
    all_batch_list = [50,50]
    #all_batch_list = np.random.choice([10,20,40,50], input_model_num, replace=False).tolist()
    #all_batch_list = np.repeat(batchSize, input_model_num).tolist()
    layer_list = np.repeat(1, sum(model_class_num)).tolist()
    model_name_abbr = np.random.choice(100000, sum(model_class_num), replace=False).tolist()
    for idx, mls in enumerate(model_class):
        for _ in range(model_class_num[idx]):
            dm = DnnModel(mls, str(model_name_abbr.pop()), model_layer=layer_list.pop(), input_w=imgWidth, input_h=imgHeight, num_classes=numClasses, batch_size=all_batch_list.pop(), desired_accuracy=0.9)
            modelCollection.append(dm)
    return modelCollection

def printAllModels(model_collection):
    for idm in model_collection:
        print(idm.getInstanceName())
        print(idm.getModelName())
        print(idm.getModelLayer())
        print(idm.getBatchSize())
        print("===============")

def buildPackedModels(model_collection):
    modelEntityCollection = []
    trainCollection = []
    for midx, mc in enumerate(model_collection):
        modelEntity = mc.getModelEntity()
        modelEntityCollection.append(modelEntity)
        modelLogit = modelEntity.build(features)
        #trainOps = modelEntity.train(modelLogit, labels)
        trainOptimizer, trainGradsVars, trainOps = modelEntity.train_step(modelLogit, labels)
        trainCollection.append(trainOps)
    return trainCollection

def execPack(train_collection, num_epoch, X_train, Y_train):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    with tf.Session(config=config) as sess:
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        sess.run(tf.global_variables_initializer())
        num_batch = Y_train.shape[0] // maxBatchSize
        for e in range(num_epoch):
            for i in range(num_batch):
                print('epoch %d / %d, step %d / %d' %(e+1, num_epoch, i+1, num_batch))
                batch_offset = i * maxBatchSize
                batch_end = (i+1) * maxBatchSize
                X_mini_batch_feed = X_train[batch_offset:batch_end,:,:,:]
                Y_mini_batch_feed = Y_train[batch_offset:batch_end,:]
                sess.run(train_collection, feed_dict={features:X_mini_batch_feed, labels:Y_mini_batch_feed})
                    
if __name__ == '__main__':
    X_data = load_images_bin(bin_dir, numChannels, imgWidth, imgHeight)
    Y_data = load_labels_onehot(label_path, numClasses)
    modelCollection = prepareModelsMan()
    printAllModels(modelCollection)
    trainCollection = buildPackedModels(modelCollection)
    start_time = timer()
    execPack(trainCollection, numEpochs, X_data, Y_data)
    end_time = timer()
    train_time = end_time - start_time
    print("training time:",train_time)