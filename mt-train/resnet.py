import tensorflow as tf
from img_utils import *

img_h = 224
img_w = 224

class ResNet(object):
    def __init__(self, net_name):
        self.net_name = net_name
        pass

    def conv_block(self, X_input, kernel_size, in_filter, out_filters, stage, block, training, stride):
        block_name = 'resnet' + stage + block
        f1, f2, f3 = out_filters
        with tf.variable_scope(block_name):
            x_shortcut = X_input

            W_conv1 = self.weight_variable([1, 1, in_filter, f1])
            X = tf.nn.conv2d(X_input, W_conv1,strides=[1, stride, stride, 1],padding='VALID')
            X = tf.layers.batch_normalization(X, axis=3, training=training)
            X = tf.nn.relu(X)

            W_conv2 = self.weight_variable([kernel_size, kernel_size, f1, f2])
            X = tf.nn.conv2d(X, W_conv2, strides=[1,1,1,1], padding='SAME')
            X = tf.layers.batch_normalization(X, axis=3, training=training)
            X = tf.nn.relu(X)

            W_conv3 = self.weight_variable([1,1,f2,f3])
            X = tf.nn.conv2d(X, W_conv3, strides=[1, 1, 1, 1], padding='VALID')
            X = tf.layers.batch_normalization(X, axis=3, training=training)

            W_shortcut = self.weight_variable([1, 1, in_filter, f3])
            x_shortcut = tf.nn.conv2d(x_shortcut, W_shortcut, strides=[1, stride, stride, 1], padding='VALID')

            add = tf.add(x_shortcut, X)
            add_result = tf.nn.relu(add)

        return add_result

    def identity_block(self, X_input, kernel_size, in_filter, out_filters, stage, block, training):
        block_name = 'resnet_' + stage + '_' + block
        f1, f2, f3 = out_filters
        with tf.variable_scope(block_name):
            X_shortcut = X_input

            W_conv1 = self.weight_variable([1, 1, in_filter, f1])
            X = tf.nn.conv2d(X_input, W_conv1, strides=[1, 1, 1, 1], padding='SAME')
            X = tf.layers.batch_normalization(X, axis=3, training=training)
            X = tf.nn.relu(X)

            W_conv2 = self.weight_variable([kernel_size, kernel_size, f1, f2])
            X = tf.nn.conv2d(X, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
            X = tf.layers.batch_normalization(X, axis=3, training=training)
            X = tf.nn.relu(X)

            W_conv3 = self.weight_variable([1, 1, f2, f3])
            X = tf.nn.conv2d(X, W_conv3, strides=[1, 1, 1, 1], padding='VALID')
            X = tf.layers.batch_normalization(X, axis=3, training=training)

            add = tf.add(X, X_shortcut)
            add_result = tf.nn.relu(add)

        return add_result


    def build(self, input, training=True, keep_prob=0.5):
        print("building resnet...")

        #assert(x.shape == (x.shape[0],70,70,3))
        with tf.variable_scope(self.net_name + '_instance'):
            x = tf.pad(input, tf.constant([[0, 0], [3, 3, ], [3, 3], [0, 0]]), "CONSTANT")
            #training = tf.placeholder(tf.bool, name='training')
            w_conv1 = self.weight_variable([7, 7, 3, 64])
            x = tf.nn.conv2d(x, w_conv1, strides=[1, 2, 2, 1], padding='VALID')
            x = tf.layers.batch_normalization(x, axis=3, training=training)
            x = tf.nn.relu(x)
            x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
            #assert (x.get_shape() == (x.get_shape()[0], 15, 15, 64))

            #stage 64
            x = self.conv_block(x, 3, 64, [64, 64, 256], stage='stage64', block='conv_block', training=training, stride=1)
            x = self.identity_block(x, 3, 256, [64, 64, 256], stage='stage64', block='identity_block1', training=training)
            x = self.identity_block(x, 3, 256, [64, 64, 256], stage='stage64', block='identity_block2', training=training)

            #stage 128
            x = self.conv_block(x, 3, 256, [128, 128, 512], stage='stage128', block='conv_block', training=training, stride=2)
            x = self.identity_block(x, 3, 512, [128, 128, 512], stage='stage128', block='identity_block1', training=training)
            x = self.identity_block(x, 3, 512, [128, 128, 512], stage='stage128', block='identity_block2', training=training)
            x = self.identity_block(x, 3, 512, [128, 128, 512], stage='stage128', block='identity_block3', training=training)

            #stage 256
            x = self.conv_block(x, 3, 512, [256, 256, 1024], stage='stage256', block='conv_block', training=training, stride=2)
            x = self.identity_block(x, 3, 1024, [256, 256, 1024], stage='stage256', block='identity_block1', training=training)
            x = self.identity_block(x, 3, 1024, [256, 256, 1024], stage='stage256', block='identity_block2', training=training)
            x = self.identity_block(x, 3, 1024, [256, 256, 1024], stage='stage256', block='identity_block3', training=training)
            x = self.identity_block(x, 3, 1024, [256, 256, 1024], stage='stage256', block='identity_block4', training=training)
            x = self.identity_block(x, 3, 1024, [256, 256, 1024], stage='stage256', block='identity_block5', training=training)

            #stage 512
            x = self.conv_block(x, 3, 1024, [512, 512, 2048], stage='stage512', block='conv_block', training=training, stride=2)
            x = self.identity_block(x, 3, 2048, [512, 512, 2048], stage='stage512', block='identity_block1', training=training)
            x = self.identity_block(x, 3, 2048, [512, 512, 2048], stage='stage512', block='identity_block2', training=training)

            x = tf.nn.avg_pool(x, [1, 2, 2, 1], strides=[1,1,1,1], padding='VALID')
            flatten = tf.layers.flatten(x)
            x = tf.layers.dense(flatten, units=50, activation=tf.nn.relu)

            with tf.name_scope('dropout'):
                #keep_prob = tf.placeholder(tf.float32)
                x = tf.nn.dropout(x, keep_prob)

            logits = tf.layers.dense(x, units=1000, activation=tf.nn.softmax)

        print("build resnet-50 successufully")
        return logits


    def cost(self, logits, labels):
        with tf.name_scope('loss'):
            cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
        cross_entropy_cost = tf.reduce_mean(cross_entropy)
        return cross_entropy_cost


    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)
