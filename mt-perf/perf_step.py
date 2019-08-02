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
parser.add_argument('-bs', '--batchsize', type=int, default=10, help='identify the training batch size')
parser.add_argument('-e', '--epoch', type=int, default=1, help='identify the epoch numbers')
parser.add_argument('-p', '--profile', action='store_true', default=False, help='use tf.profiler or not')
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
profile_dir = '/tank/local/ruiliu/mtml-tf/mt-perf/profile_dir'

imgWidth = 224
imgHeight = 224
numClasses = 1000
numChannels = 3
maxBatchSize = args.batchsize
numEpochs = args.epoch
isProfile = args.profile

#input_model_num = 4

features = tf.placeholder(tf.float32, [None, imgWidth, imgHeight, numChannels])
labels = tf.placeholder(tf.int64, [None, numClasses])

def prepareModelsMan():
    modelCollection = []
    #model_class_num = [1,1]
    model_class_num = [2]  
    model_class = ["convnet"]
    #all_batch_list = [40]
    #model_class = ["mobilenet","mobilenet_padding"]
    all_batch_list = [10,10]
    #all_batch_list = [40,20,10,20,20,20,40,40,40,100]
    #all_batch_list = [20,10,40,50,20,10,40,20,40,20,10,10,40,20,40,10,20,40,50,100]
    #all_batch_list = np.random.choice([10,20,40,50], input_model_num, replace=False).tolist()
    #all_batch_list = np.repeat(batchSize, input_model_num).tolist()
    layer_list = np.repeat(1, sum(model_class_num)).tolist()
    #layer_list = [2,2,2,2,2,3,3,3,3,3,1,1,1,1,1,1,1,1,1,1]
    #layer_list = [2,2,3,3,1,1,1,1,1,1]
    model_name_abbr = np.random.choice(100000, sum(model_class_num), replace=False).tolist()
    for idx, mls in enumerate(model_class):
        for _ in range(model_class_num[idx]):
            dm = DnnModel(mls, str(model_name_abbr.pop()), model_layer=layer_list.pop(), input_w=imgWidth, input_h=imgHeight, num_classes=numClasses, batch_size=all_batch_list.pop(), desired_accuracy=0.9)
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
    trainCollection = []
    for midx,mc in enumerate(model_collection):
        workerDevice = '/job:localhost/task:'+ str(midx) +'/gpu:0'
        print("workerDevice:", workerDevice)
        #with tf.device(tf.train.replica_device_setter(worker_device=workerDevice)):
        modelEntity = mc.getModelEntity()
        modelLogit = modelEntity.build(features)
        modelTrain = modelEntity.cost(modelLogit, labels)
        trainCollection.append(modelTrain)
    return trainCollection

def execTrain(unit, num_epoch, X_train, Y_train):
    step_time = 0
    step_count = 0
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    #print("model name:",train_unit[2])
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
                if (i+1) % 10 == 0:
                    if isProfile:
                        start_time = timer()
                        sess.run(unit, feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed}, options=run_options, run_metadata=run_metadata)
                        end_time = timer()
                        step_time += end_time - start_time
                        step_count += 1
                        trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                        trace_file = open('/tank/local/ruiliu/mtml-tf/mt-perf/profile_dir/m10-m10-task/tf_packed_step'+str(i)+'.json', 'w')
                        #trace_file = open('/home/ruiliu/Development/mtml-tf/mt-perf/exp-result/test/tf_packed_task'+str(i)+'.json', 'w')
                        trace_file.write(trace.generate_chrome_trace_format())
                    else:
                        start_time = timer()
                        sess.run(unit, feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})
                        end_time = timer()
                        step_time += end_time - start_time
                        step_count += 1
                else:
                    sess.run(unit, feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})
        print(step_time)
        print(step_count)
        print("average step time:", step_time / step_count * 1000)

if __name__ == '__main__':
    X_data = load_images_bin(bin_dir, numChannels, imgWidth, imgHeight)
    Y_data = load_labels_onehot(label_path, numClasses)
    modelCollection = prepareModelsMan()
    printAllModels(modelCollection)
    trainCollection = buildModels(modelCollection)
    execTrain(trainCollection, numEpochs, X_data, Y_data)
       
