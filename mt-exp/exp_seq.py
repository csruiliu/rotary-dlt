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
parser.add_argument('-p', '--preproc', action='store_false', default=True, help='use preproc to transform the data before training or not')
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

sameBatchSize = args.samebatchsize
preproc = args.preproc
sameOptimizer = args.sameoptimizer
trainStep = args.trainstep

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

def execTrainPreproc(model, num_epoch, X_train_path, Y_train):
    modelEntity = model.getModelEntity()
    modelLogit = modelEntity.build(features)
    trainStep = modelEntity.train(modelLogit, labels)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    image_list = sorted(os.listdir(X_train_path))
    batchSize = model.getBatchSize()
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        num_batch = Y_train.shape[0] // batchSize
        num_batch_list = np.arange(num_batch)
        for e in range(num_epoch):
            for i in range(num_batch):
                print('epoch %d / %d, step %d / %d' %(e+1, num_epoch, i+1, num_batch))     
                batch_offset = i * batchSize
                batch_end = (i+1) * batchSize
                batch_list = image_list[batch_offset:batch_end]   
                X_mini_batch_feed = load_image_dir(X_train_path, batch_list, imgHeight, imgWidth)
                Y_mini_batch_feed = Y_train[batch_offset:batch_end,:]
                sess.run(trainStep, feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})

def execTrain(model, num_epoch, X_train, Y_train):
    modelEntity = model.getModelEntity()
    modelLogit = modelEntity.build(features)
    trainStep = modelEntity.train(modelLogit, labels)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    batchSize = model.getBatchSize()
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        num_batch = Y_train.shape[0] // batchSize 
        num_batch_list = np.arange(num_batch)
        for e in range(num_epoch):
            for i in range(num_batch):
                print('epoch %d / %d, step %d / %d' %(e+1, num_epoch, i+1, num_batch))     
                batch_offset = i * batchSize
                batch_end = (i+1) * batchSize
                X_mini_batch_feed = X_train[batch_offset:batch_end,:,:,:]
                Y_mini_batch_feed = Y_train[batch_offset:batch_end,:]
                sess.run(trainStep, feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})

if __name__ == '__main__':
    showExpConfig()
    modelCollection = prepareModels()
    if preproc:
        start_time = timer()
        for midx in modelCollection:
            Y_data = load_labels_onehot(label_path, numClasses)
            p = Process(target=execTrainPreproc, args=(midx, numEpochs, image_dir, Y_data,))
            p.start()
            print(p.pid)
            p.join()
        end_time = timer()
        dur_time = end_time - start_time
        print("total training time:", dur_time)
        
    else:
        start_time = timer()
        for midx in modelCollection:
            X_data = load_images_bin(bin_dir, numChannels, imgWidth, imgHeight)
            Y_data = load_labels_onehot(label_path, numClasses)
            p = Process(target=execTrain, args=(midx, numEpochs, image_dir, Y_data,))
            p.start()
            print(p.pid)
            p.join()
        end_time = timer()
        dur_time = end_time - start_time
        print("total training time:", dur_time)
