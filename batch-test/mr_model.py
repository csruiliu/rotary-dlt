import sys
sys.path.append('/home/rui/Development/tf-exp/models/research/slim')
sys.path.append('/home/rui/Development/tf-exp/models/official')

import tensorflow as tf
from tensorflow.python.framework import meta_graph
from nets.mobilenet import mobilenet_v2
from nets import resnet_v2

img_w = 224
img_h = 224
img_size = img_w * img_h 
img_channel = 3

def filter_net(input):
    idx = input[0, 0]
    if idx == 1:
        print("mobilenet")
    elif idx == 2:
        print("resnet")
    else:
        print("cannot identify the idx for next setp")

#def mr_model(input):
#    filter_net = tf.Graph()
#    with filter_net.as_default():   