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
#parser.add_argument('-m', '--samemodel', action='store_false', default=True, help='pack same model to train or not')
parser.add_argument('-p', '--preproc', action='store_false', default=True, help='use preproc to transform the data before training or not')
#parser.add_argument('-d', '--samedata', action='store_false', default=True, help='use same training batch data or not')
#parser.add_argument('-o', '--sameoptimizer', action='store_false', default=True, help='use same optimizer or not')
#parser.add_argument('-b', '--samebatchsize', action='store_false', default=True, help='use same batch size or not')
#parser.add_argument('-t', '--trainstep', action='store_true', default=False, help='use same compute and apply to update gradient or not')
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
numEpochs = 1
input_model_num = 2
remark = 3

#model_class_num = [1,1,1,1,1]
#model_class = ["mobilenet","mobilenet","mobilenet","mobilenet","mobilenet"]       
#batch_list = [10,10,10,10,10]
#opt_list = ["Adam","Adam","Adam","Adam","Adam"]
#maxBatchSize = 10

model_class_num = [1,1]
model_class = ["densenet","densenet"]       
batch_list = [32,32]
opt_list = ["Adam","Adam"]
maxBatchSize = 32



preproc = args.preproc

features = tf.placeholder(tf.float32, [None, imgWidth, imgHeight, numChannels])
labels = tf.placeholder(tf.int64, [None, numClasses])

def showExpConfig():
    #print("Packing the same model:", sameModel)
    #print("Using the same train data:", sameTrainData)
    print("Using the preprocess:", preproc)
    #print("Using the same optimizer:", sameOptimizer)
    #print("Using the same batch size:", sameBatchSize)

def prepareModels():
    modelCollection = []
    
    model_name_abbr = np.random.choice(100000, sum(model_class_num), replace=False).tolist()

    for idx, mls in enumerate(model_class):
        dm = DnnModel(mls, str(model_name_abbr.pop()), model_layer=1, 
                      input_w=imgWidth, input_h=imgHeight, num_classes=numClasses, 
                      batch_size=batch_list[idx], optimizer=opt_list[idx])
        modelCollection.append(dm)
    return modelCollection

def printAllModels(model_collection):
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
        modelLogit = modelEntity.build(features)
        #trainOps, _ = modelEntity.compute_grads(modelLogit, labels)
        #_, _, trainOps = modelEntity.train_step(modelLogit, labels)
        trainOps = modelEntity.train(modelLogit, labels)
        trainCollection.append(trainOps)
    return trainCollection

def execTrain(unit, num_epoch, X_train, Y_train):
    step_time = 0
    step_count = 0
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    with tf.Session(config=config) as sess:
        #run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        #run_metadata = tf.RunMetadata()
        writer = tf.summary.FileWriter("logs/", sess.graph)
        sess.run(tf.global_variables_initializer())
        num_batch = Y_train.shape[0] // maxBatchSize
        for e in range(num_epoch):
            for i in range(num_batch):
                print('epoch %d / %d, step %d / %d' %(e+1, num_epoch, i+1, num_batch))
                batch_offset = i * maxBatchSize
                batch_end = (i+1) * maxBatchSize
                X_mini_batch_feed = X_train[batch_offset:batch_end,:,:,:]
                Y_mini_batch_feed = Y_train[batch_offset:batch_end,:]
                sess.run(unit, feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})
                        
def execTrainPreprocProfile(unit, num_epoch, X_train_path, Y_train):
    step_count = 0
    step_time = 0
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    #config.inter_op_parallelism_threads = 9
    #config.intra_op_parallelism_threads = 9
    image_list = sorted(os.listdir(X_train_path))
    with tf.Session(config=config) as sess:
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        trainUnit = unit 
        sess.run(tf.global_variables_initializer())
        num_batch = Y_train.shape[0] // maxBatchSize
        for e in range(num_epoch):
            for i in range(num_batch):
                print('epoch %d / %d, step %d / %d' %(e+1, num_epoch, i+1, num_batch))
                batch_offset = i * maxBatchSize
                batch_end = (i+1) * maxBatchSize
                batch_list = image_list[batch_offset:batch_end]   
                X_mini_batch_feed = load_image_dir(X_train_path, batch_list, imgHeight, imgWidth)
                Y_mini_batch_feed = Y_train[batch_offset:batch_end,:]
                if i % 20 == 0:
                    start_time = timer()
                    sess.run(trainUnit, feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed}, options=run_options, run_metadata=run_metadata)
                    end_time = timer()
                    #dur_time = end_time - start_time
                    #step_count += 1
                    #step_time += dur_time
                    #sess.run(unit, feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed}, options=run_options, run_metadata=run_metadata)
                    trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                    trace_file = open('/tank/local/ruiliu/mtml-tf/mt-exp/profile_dir/tf_single_comp_'+str(i)+'.json', 'w')
                    #trace_file = open('/home/ruiliu/Development/mtml-tf/mt-exp/profile_dir/tf_packed_'+str(i)+'.json', 'w')
                    trace_file.write(trace.generate_chrome_trace_format(show_dataflow=True))
                else:
                    sess.run(trainUnit, feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})
        #print(step_time)
        #print(step_count)
        #print("average step time:", step_time / step_count * 1000)



def execTrainPreproc(unit, num_epoch, X_train_path, Y_train):
    step_count = 0
    step_time = 0
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
                batch_offset = i * maxBatchSize
                batch_end = (i+1) * maxBatchSize
                batch_list = image_list[batch_offset:batch_end]   
                X_mini_batch_feed = load_image_dir(X_train_path, batch_list, imgHeight, imgWidth)
                Y_mini_batch_feed = Y_train[batch_offset:batch_end,:]
                sess.run(unit, feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})


if __name__ == '__main__':
    showExpConfig()    
    modelCollection = prepareModels()
    printAllModels(modelCollection)
    trainCollection = buildModels(modelCollection) 
    if preproc:
        Y_data = load_labels_onehot(label_path, numClasses)
        start_time = timer()
        execTrainPreproc(trainCollection, numEpochs, image_dir, Y_data)
        end_time = timer()
        dur_time = end_time - start_time
        print("overall time:",dur_time)
    else:
        X_data = load_images_bin(bin_dir, numChannels, imgWidth, imgHeight)
        Y_data = load_labels_onehot(label_path, numClasses)
        execTrain(trainCollection, numEpochs, X_data, Y_data)
    
