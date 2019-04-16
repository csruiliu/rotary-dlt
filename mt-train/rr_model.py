import sys
sys.path.append('/home/ruiliu/Development/mtml-tf/models/research/slim')
sys.path.append('/home/ruiliu/Development/mtml-tf/models/official')

import tensorflow as tf
from nets import resnet_v2

def rr_net(images1,images2,isTraining):
    with tf.contrib.slim.arg_scope(resnet_v2.resnet_arg_scope()):
        net1, endpoints1 = resnet_v2.resnet_v2_50(images1, 1001, is_training=isTraining, reuse=tf.AUTO_REUSE)
        print("resnet1")
        print(net1.name)

        net2, endpoints2 = resnet_v2.resnet_v2_50(images2, 1001, is_training=isTraining, reuse=tf.AUTO_REUSE)
        print("resnet2")
        print(net2.name)
     
        return net1, net2

