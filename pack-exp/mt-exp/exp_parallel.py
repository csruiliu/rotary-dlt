from __future__ import division
import tensorflow as tf
from tensorflow.python.client import timeline
import numpy as np
from multiprocessing import Process
from multiprocessing import Pool
import argparse

from timeit import default_timer as timer
import time

import contextlib
from img_utils import *
from dnn_model import DnnModel

from mobilenet import mobilenet

#########################
# Command Parameters
#########################

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--preproc', action='store_false', default=True, help='use preproc to transform the data before training or not')
parser.add_argument('-o', '--sameoptimizer', action='store_false', default=True, help='use same optimizer or not')
parser.add_argument('-b', '--samebatchsize', action='store_false', default=True, help='use same batch size or not')
parser.add_argument('-t', '--trainstep', action='store_true', default=False, help='use same compute and apply to update gradient or not')
parser.add_argument('-c', '--usecpu', action='store_true', default=False, help='use same compute and apply to update gradient or not')
args = parser.parse_args()

#########################
# Global Variables
#########################

image_dir = '/home/ruiliu/Development/mtml-tf/dataset/imagenet10k'
#image_dir = '/tank/local/ruiliu/dataset/imagenet10k'
#image_dir = '/local/ruiliu/dataset/imagenet10k'

bin_dir = '/home/ruiliu/Development/mtml-tf/dataset/imagenet1k.bin'
#bin_dir = '/tank/local/ruiliu/dataset/imagenet10k.bin'
#bin_dir = '/local/ruiliu/dataset/imagenet10k.bin'

label_path = '/home/ruiliu/Development/mtml-tf/dataset/imagenet1k-label.txt'
#label_path = '/tank/local/ruiliu/dataset/imagenet10k-label.txt'
#label_path = '/local/ruiliu/dataset/imagenet10k-label.txt'

#profile_dir = '/home/ruiliu/Development/mtml-tf/mt-perf/profile_dir'
#profile_dir = '/tank/local/ruiliu/mtml-tf/mt-perf/profile_dir'

imgWidth = 224
imgHeight = 224
numClasses = 1000
numChannels = 3
numEpochs = 10

remark = 3

sameBatchSize = args.samebatchsize
preproc = args.preproc
sameOptimizer = args.sameoptimizer
trainStep = args.trainstep
useCPU = args.usecpu

if useCPU:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

model_list = ["resnet","resnet"]
batch_list = [32,32]
opt_list = ["Adam","Adam"]

features = tf.placeholder(tf.float32, [None, imgWidth, imgHeight, numChannels])
labels = tf.placeholder(tf.int64, [None, numClasses])

def showExpConfig():
    print("Running the models:")
    for midx,mc in enumerate(model_list):
        print(mc,":",str(batch_list[midx]),",",str(opt_list[midx]))
    print("Using the preprocess:", preproc)

def prepareModels():
    modelCollection = []
    model_name_abbr = np.random.choice(100000, len(model_list), replace=False).tolist()
    for idx, mls in enumerate(model_list):
        dm = DnnModel(mls, str(model_name_abbr.pop()), model_layer=1, 
                        input_w=imgWidth, input_h=imgHeight, num_classes=numClasses, 
                        batch_size=batch_list[idx], optimizer=opt_list[idx])
        modelCollection.append(dm)
    return modelCollection

def buildModels(model_collection):
    modelEntityCollection = []
    trainCollection = []
    for midx, mc in enumerate(model_collection):
        modelEntity = mc.getModelEntity()
        modelEntityCollection.append(modelEntity)
        modelLogit = modelEntity.build(features)
        if trainStep:
            trainOptimizer, trainGradsVars, trainOps = modelEntity.train_step(modelLogit, labels)
        else:
            trainOps = modelEntity.train(modelLogit, labels)
        trainCollection.append(trainOps)
    return trainCollection

def execTrain(unit, num_epoch, batch_size, X_train, Y_train):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        num_batch = Y_train.shape[0] // batch_size
        num_batch_list = np.arange(num_batch)
        for e in range(num_epoch):
            for i in range(num_batch):
                print('epoch %d / %d, step %d / %d' %(e+1, num_epoch, i+1, num_batch)) 
                batch_offset = i * batch_size
                batch_end = (i+1) * batch_size
                X_mini_batch_feed = X_train[batch_offset:batch_end,:,:,:]
                Y_mini_batch_feed = Y_train[batch_offset:batch_end,:]
                sess.run(unit, feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})

def execTrainPreproc1(unit, num_epoch, batch_size, X_train_path, Y_train):
    print("sss")

def execTrainPreproc(unit, num_epoch, batch_size, X_train_path, Y_train):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    image_list = sorted(os.listdir(X_train_path))
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        num_batch = Y_train.shape[0] // batch_size
        num_batch_list = np.arange(num_batch)
        for e in range(num_epoch):
            for i in range(num_batch):
                start_time = timer()
                print('epoch %d / %d, step %d / %d' %(e+1, num_epoch, i+1, num_batch))      
                batch_offset = i * batch_size
                batch_end = (i+1) * batch_size
                batch_list = image_list[batch_offset:batch_end]   
                X_mini_batch_feed = load_image_dir(X_train_path, batch_list, imgHeight, imgWidth)
                Y_mini_batch_feed = Y_train[batch_offset:batch_end,:]
                sess.run(unit, feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})

    return num_epoch

def time_wrap(f):
    def wrap(*args, **kvargs):
        start = timer()
        result = f(*args, **kvargs)
        end = timer()
        print("the run time for function %s with params %s is %s" %(f.__name__, args[1],  end-start))
        return result
    return wrap

if __name__ == '__main__':
    start_time = timer()
    
    model_layer = 1
    batch_size = 32
    opt = 'Adam'
    modelEntity1 = mobilenet("mlp_"+str(1), model_layer, imgHeight, imgWidth, batch_size, numClasses, opt)
    modelEntity2 = mobilenet("mlp_"+str(2), model_layer, imgHeight, imgWidth, batch_size, numClasses, opt)

    features1 = tf.placeholder(tf.float32, [None, imgWidth, imgHeight, numChannels])
    features2 = tf.placeholder(tf.float32, [None, imgWidth, imgHeight, numChannels])
    labels1 = tf.placeholder(tf.int64, [None, numClasses])
    labels2 = tf.placeholder(tf.int64, [None, numClasses])

    modelLogit1 = modelEntity1.build(features1)
    modelLogit2 = modelEntity2.build(features2)

    trainOps1 = modelEntity1.train(modelLogit1, labels1)
    trainOps2 = modelEntity2.train(modelLogit2, labels2)

    Y_data = load_labels_onehot(label_path, numClasses)

    para_list1 = [trainOps1, 1, 32, image_dir, Y_data]
    para_list2 = [trainOps2, 2, 32, image_dir, Y_data]

    pls = list()
    pls.append(para_list1)
    pls.append(para_list2)

    end_time = timer()
    dur_time = end_time - start_time
    print(dur_time)
    #pool = Pool(processes=2) 
    jobs = []
    start_time = timer()
    pool = Pool(processes=2)
    for pidx in pls:
        #print(pidx[1])
        res = pool.map_async(execTrainPreproc, (pidx[0],pidx[1],pidx[2],pidx[3],pidx[4],))
        jobs.append(res)
    pool.close()
    pool.join()
    for j in jobs:
        print(j)
    #pool_results = pool.map_async(execTrainPreproc, ((unit, num_epoch, batch_size, X_train_path, Y_train) for unit, num_epoch, batch_size, X_train_path, Y_train in pls))
    #pool_results.wait()
        #pool.close()
        #pool.join()
    #if pool_results.ready():
    #    if pool_results.successful():
    #        print("successful")

    end_time = timer()
    dur_time = end_time - start_time
    print(dur_time)
        #print(pool_results)
    #pool.terminate()
    '''
    showExpConfig()
    modelCollection = prepareModels()
    trainCollection = buildModels(modelCollection)
    
    Y_data = load_labels_onehot(label_path, numClasses)
    if preproc:
        start_time = timer()
        for tidx, tc in enumerate(trainCollection):
            p = Process(target=execTrainPreproc, args=(tc, numEpochs, batch_list[tidx], image_dir, Y_data,))
            p.start()
            print(p.pid)
            p.join()
        end_time = timer()
        dur_time = end_time - start_time
        print("total training time:", dur_time)
        
    else:
        X_data = load_images_bin(bin_dir, numChannels, imgWidth, imgHeight)
        start_time = timer()
        for tidx, tc in enumerate(trainCollectionn):
            p = Process(target=execTrain, args=(tc,  numEpochs, batch_list[tidx], X_data, Y_data,))
            p.start()
            print(p.pid)
            p.join()
        end_time = timer()
        dur_time = end_time - start_time
        print("total training time:", dur_time)
    '''
