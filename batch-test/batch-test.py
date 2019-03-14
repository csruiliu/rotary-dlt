import sys
sys.path.append('/home/ruiliu/Development/tf-exp/models/research/slim')
sys.path.append('/home/ruiliu/Development/tf-exp/models/official')

import tensorflow as tf
from nets import resnet_v2
from datasets import imagenet
import timeit

g = tf.Graph()
with g.as_default():
    

