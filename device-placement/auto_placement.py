from __future__ import division
import tensorflow as tf
import numpy as np
from operator import itemgetter

from dnn_model import DnnModel
from img_utils import *
import config as cfg_yml

def generateWorkload():
    trainOpCollection = list()

    np.random.seed(randSeed)

    model_name_abbr = np.random.choice(randSeed, workloadNum, replace=False).tolist()

    workload_model_index_list = np.random.choice(len(workloadModel), workloadNum, replace=True).tolist()
    workload_model_list = list(itemgetter(*workload_model_index_list)(workloadModel))

    workload_batchsize_index_list = np.random.choice(len(workloadBatchSize), workloadNum, replace=True).tolist()
    workload_batchsize_list = list(itemgetter(*workload_batchsize_index_list)(workloadBatchSize))

    workload_activation_index_list = np.random.choice(len(workloadActivation), workloadNum, replace=True).tolist()
    workload_activation_list = list(itemgetter(*workload_activation_index_list)(workloadActivation))

    workload_opt_index_list = np.random.choice(len(workloadOptimizer), workloadNum, replace=True).tolist()
    workload_opt_list = list(itemgetter(*workload_opt_index_list)(workloadOptimizer))

    for idx, mt in enumerate(workload_model_list):
        dm = DnnModel(mt, str(model_name_abbr.pop()), workloadNumLayer[0], imgHeight, imgWidth, numChannels, numClasses,
                      workload_batchsize_list[idx], workload_opt_list[idx], workloadLearnRate[0], workload_activation_list[idx], True)
        modelEntity = dm.getModelEntity()
        modelLogit = modelEntity.build(features)
        trainStep = modelEntity.train(modelLogit, labels)
        trainOpCollection.append(trainStep)

    return trainOpCollection

if __name__ == "__main__":

    #########################
    # Parameters
    #########################

    imgWidth = cfg_yml.img_width
    imgHeight = cfg_yml.img_height
    numChannels = cfg_yml.num_channels
    numClasses = cfg_yml.num_classes
    randSeed = cfg_yml.rand_seed

    workloadNum = cfg_yml.workload_num
    workloadModel = cfg_yml.workload_model_type
    workloadBatchSize = cfg_yml.workload_batch_size
    workloadActivation = cfg_yml.workload_activation
    workloadOptimizer = cfg_yml.workload_opt
    workloadLearnRate = cfg_yml.workload_learning_rate
    workloadNumLayer = cfg_yml.workload_num_layer

    #########################
    # Build workload
    #########################

    features = tf.placeholder(tf.float32, [None, imgWidth, imgHeight, numChannels])
    labels = tf.placeholder(tf.int64, [None, numClasses])

    workload = generateWorkload()
    
    #########################
    # Handle workload
    #########################

    image_path_raw = cfg_yml.imagenet_t1k_img_path
    image_path_bin = cfg_yml.imagenet_t1k_bin_path
    label_path = cfg_yml.imagenet_t1k_label_path



