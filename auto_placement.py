from __future__ import division
import tensorflow as tf
from tensorflow.python.client import timeline
from utils.img_utils import *





if __name__ == "__main__":
    label_path = '/home/ruiliu/Development/dataset/imagenet1k-label.txt'



    numClasses = 1000
    Y_data = load_imagenet_labels_onehot(label_path, numClasses)