from __future__ import division
import tensorflow as tf

from dnn_model import DnnModel
from img_utils import *
import config as cfg_yml

if __name__ == "__main__":
    imgWidth = cfg_yml.pack_img_width
    imgHeight = cfg_yml.pack_img_height
    numChannels = cfg_yml.pack_num_channels
    numClasses = cfg_yml.pack_num_classes
    numEpochs = cfg_yml.pack_num_epoch
    trainModel = cfg_yml.pack_model_type
    trainBatchSize = cfg_yml.pack_batch_size
    trainOptimizer = cfg_yml.pack_opt
    randSeed = cfg_yml.pack_rand_seed
    trainNumLayer = cfg_yml.pack_num_layer
    trainLearnRate = cfg_yml.pack_learning_rate
    trainActivation = cfg_yml.pack_activation


    image_path_raw = cfg_yml.imagenet_t1k_img_path
    image_path_bin = cfg_yml.imagenet_t1k_bin_path
    label_path = cfg_yml.imagenet_t1k_label_path

    numClasses = cfg_yml.pack_num_classes
