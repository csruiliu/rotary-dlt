
import tensorflow as tf
import tensorflow.contrib as tc
from img_utils import *

import numpy as np
from timeit import default_timer as timer


mini_batches=10
img_h=224
img_w=224

#MobileNetV2
class MobileNet(object):
    def __init__(self, is_training=True, input_size=224):
        self.is_training = is_training
        self.normalizer = tc.layers.batch_norm
        self.bn_params = {'is_training': self.is_training}

    def build_model(self, input):
        with tf.variable_scope('init_conv'):
            self.i = 0
            output = tc.layers.conv2d(input, 32, 3, 2,normalizer_fn=self.normalizer, normalizer_params=self.bn_params)

            net = self._inverted_bottleneck(output, 1, 16, 0)
            net = self._inverted_bottleneck(net, 6, 24, 1)
            net = self._inverted_bottleneck(net, 6, 24, 0)
            net = self._inverted_bottleneck(net, 6, 32, 1)
            net = self._inverted_bottleneck(net, 6, 32, 0)
            net = self._inverted_bottleneck(net, 6, 32, 0)
            net = self._inverted_bottleneck(net, 6, 64, 1)
            net = self._inverted_bottleneck(net, 6, 64, 0)
            net = self._inverted_bottleneck(net, 6, 64, 0)
            net = self._inverted_bottleneck(net, 6, 64, 0)
            net = self._inverted_bottleneck(net, 6, 96, 0)
            net = self._inverted_bottleneck(net, 6, 96, 0)
            net = self._inverted_bottleneck(net, 6, 96, 0)
            net = self._inverted_bottleneck(net, 6, 160, 1)
            net = self._inverted_bottleneck(net, 6, 160, 0)
            net = self._inverted_bottleneck(net, 6, 160, 0)
            net = self._inverted_bottleneck(net, 6, 320, 0)

            net = tc.layers.conv2d(net, 1280, 1, normalizer_fn=self.normalizer, normalizer_params=self.bn_params)
            net = tc.layers.avg_pool2d(net, 7)
            net = tc.layers.conv2d(net, 1000, 1, activation_fn=None)
            logits = tf.squeeze(net)
        return logits

    def _inverted_bottleneck(self, input, up_sample_rate, channels, subsample):
        with tf.variable_scope('inverted_bottleneck{}_{}_{}'.format(self.i, up_sample_rate, subsample)):
            self.i += 1
            stride = 2 if subsample else 1
            output = tc.layers.conv2d(input, up_sample_rate*input.get_shape().as_list()[-1], 1,
                                      activation_fn=tf.nn.relu6,
                                      normalizer_fn=self.normalizer, normalizer_params=self.bn_params)
            output = tc.layers.separable_conv2d(output, None, 3, 1, stride=stride,
                                                activation_fn=tf.nn.relu6,
                                                normalizer_fn=self.normalizer, normalizer_params=self.bn_params)
            output = tc.layers.conv2d(output, channels, 1, activation_fn=None,
                                      normalizer_fn=self.normalizer, normalizer_params=self.bn_params)
            if input.get_shape().as_list()[-1] == channels:
                output = tf.add(input, output)
            return output

    def cost(self, logits, labels):
        with tf.name_scope('loss'):
            cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
        cross_entropy_cost = tf.reduce_mean(cross_entropy)
        return cross_entropy_cost

    def train(self, X_train, Y_train):
        features = tf.placeholder(tf.float32, [None, img_h, img_w, 3])
        labels = tf.placeholder(tf.int64, [None, 1000])

        logits = self.build_model(features)
        cross_entropy = self.cost(logits, labels)
        with tf.name_scope('optimizer'):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            #num_batch = Y_train.get_shape().as_list()[0] // mini_batches
            num_batch = Y_train.shape[0] // mini_batches
            total_time = 0
            for i in range(num_batch):
                start_time = timer()
                print('step %d / %d' %(i+1, num_batch))
                X_mini_batch_feed = X_train[num_batch:num_batch + mini_batches,:,:,:]
                Y_mini_batch_feed = Y_train[num_batch:num_batch + mini_batches,:]
                #X_mini_batch_feed = X_mini_batch.eval()
                #Y_mini_batch_feed = Y_mini_batch.eval()
                train_step.run(feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})
                end_time = timer()
                total_time += end_time - start_time
            print("training time for 1 epoch:",total_time)

def main(_):
    data_dir = '/home/rui/Development/mtml-tf/dataset/test'
    label_path = '/home/rui/Development/mtml-tf/dataset/test.txt'
    X_data = load_images(data_dir)
    Y_data = load_labels_onehot(label_path)
    #print(Y_data.shape[0])
    mobilenet = MobileNet()
    mobilenet.train(X_data, Y_data)

if __name__ == '__main__':
    tf.app.run(main=main)
