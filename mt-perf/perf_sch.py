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
batchSize = 0
numEpochs = args.epoch
isProfile = args.profile

#input_model_num = 4

features = tf.placeholder(tf.float32, [None, imgWidth, imgHeight, numChannels])
labels = tf.placeholder(tf.int64, [None, numClasses])

def prepareModelsMan():
    modelCollection = []
    #model_class_num = [5,5,5,5]
    model_class_num = [3,3,2,2]
    #model_class_num = [1]  
    #model_class = ["resnet_padding"]
    #all_batch_list = [40]
    model_class = ["mobilenet_padding","resnet_padding","perceptron_padding","convnet_padding"]
    all_batch_list = [40,20,10,20,20,20,40,40,40,100]
    #all_batch_list = [20,10,40,50,20,10,40,20,40,20,10,10,40,20,40,10,20,40,50,100]
    #all_batch_list = np.random.choice([10,20,40,50], input_model_num, replace=False).tolist()
    #all_batch_list = np.repeat(batchSize, input_model_num).tolist()
    #layer_list = np.repeat(1, input_model_num).tolist()
    #layer_list = [2,2,2,2,2,3,3,3,3,3,1,1,1,1,1,1,1,1,1,1]
    layer_list = [2,2,3,3,1,1,1,1,1,1]
    model_name_abbr = np.random.choice(100000, sum(model_class_num), replace=False).tolist()
    for idx, mls in enumerate(model_class):
        for _ in range(model_class_num[idx]):
            global batchSize
            batchSize = all_batch_list.pop()
            #print(batchSize)
            dm = DnnModel(mls, str(model_name_abbr.pop()), model_layer=layer_list.pop(), input_w=imgWidth, input_h=imgHeight, num_classes=numClasses, batch_size=batchSize, desired_accuracy=0.9)
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
    #modelEntityCollection = []
    trainCollection = []
    for midx in model_collection:
        trainUnit = []
        modelEntity = midx.getModelEntity()
        modelBatchSize = midx.getBatchSize()
        modelName = midx.getModelName()
        modelLogit = modelEntity.build(features)
        modelTrain = modelEntity.cost(modelLogit, labels)
        trainUnit.append(modelTrain)
        trainUnit.append(modelBatchSize)
        trainUnit.append(modelName)
        trainCollection.append(trainUnit)
    return trainCollection

def scheduleModels(train_collection):
    sch_unit1 = []
    sch_unit1_train = []
    sch_unit1_train.append(train_collection[0][0])
    sch_unit1_train.append(train_collection[1][0])
    sch_unit1_train.append(train_collection[4][0])
    sch_unit1_train.append(train_collection[3][0])
    sch_unit1_train.append(train_collection[8][0])
    sch_unit1.append(sch_unit1_train)
    sch_unit1.append(100)

    sch_unit2 = []
    sch_unit2_train = []
    sch_unit2_train.append(train_collection[2][0])
    sch_unit2_train.append(train_collection[5][0])
    sch_unit2_train.append(train_collection[7][0])
    sch_unit2_train.append(train_collection[6][0])
    sch_unit2_train.append(train_collection[9][0])
    
    sch_unit2.append(sch_unit2_train)
    sch_unit2.append(40)

    sch_unit3 = []
    sch_unit3_train = []
    sch_unit3_train.append(train_collection[3][0])
    sch_unit3_train.append(train_collection[6][0])
    sch_unit3_train.append(train_collection[8][0])
    sch_unit3_train.append(train_collection[9][0])
    sch_unit3.append(sch_unit3_train)
    sch_unit3.append(40)

    sch_collection = []
    sch_collection.append(sch_unit1)
    sch_collection.append(sch_unit2)
    #sch_collection.append(sch_unit3)
    return sch_collection

def execSch(sch_unit, num_epoch, X_train, Y_train):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    #print("model name:",train_unit[2])
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        num_batch = Y_train.shape[0] // sch_unit[1]
        for e in range(num_epoch):
            for i in range(num_batch):
                print('epoch %d / %d, step %d / %d' %(e+1, num_epoch, i+1, num_batch))
                batch_offset = i * batchSize
                batch_end = (i+1) * batchSize
                X_mini_batch_feed = X_train[batch_offset:batch_end,:,:,:]
                Y_mini_batch_feed = Y_train[batch_offset:batch_end,:]
                sess.run(sch_unit[0], feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})
    
if __name__ == '__main__':
    X_data = load_images_bin(bin_dir, numChannels, imgWidth, imgHeight)
    Y_data = load_labels_onehot(label_path, numClasses)
    modelCollection = prepareModelsMan()
    printAllModels(modelCollection)
    trainCollection = buildModels(modelCollection)
    schCollection = scheduleModels(trainCollection)
    for sidx in schCollection:
        p = Process(target=execSch, args=(sidx, numEpochs, X_data, Y_data,))
        p.start()
        print(p.pid)
        p.join()