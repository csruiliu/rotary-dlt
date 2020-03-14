from __future__ import division
import tensorflow as tf
from tensorflow.python.client import timeline
from img_utils import *
from dnn_model import DnnModel

import config as cfg_yml


if __name__ == "__main__":
    image_path_raw = cfg_yml.imagenet_t1k_img_path
    image_path_bin = cfg_yml.imagenet_t1k_bin_path
    label_path = cfg_yml.imagenet_t1k_label_path

    numClasses = cfg_yml.pack_num_classes
