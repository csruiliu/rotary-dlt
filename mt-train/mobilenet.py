
import tensorflow as tf
import tensorflow.contrib as tc

import numpy as np
import time

#MobileNetV2
class MobileNet(object):
    def __init__(self, is_training=True, input_size=224):
        self.input_size = input_size
        self.is_training = is_training
        self.normalizer = tc.layers.batch_norm
        self.bn_params = {'is_training': self.is_training}

        with tf.variable_scope('MobileNetV2'):
            self._create_placeholders()
            self._build_model()

    def _create_placeholders(self):
        self.input = tf.placeholder(dtype=tf.float32, shape=[None, self.input_size, self.input_size, 3])


    def _build_model(self):
        self.i = 0
        with tf.variable_scope('init_conv'):
            output = tc.layers.conv2d(self.input, 32, 3, 2,normalizer_fn=self.normalizer, normalizer_params=self.bn_params)

        self.output = self._inverted_bottleneck(output, 1, 16, 0)
        self.output = self._inverted_bottleneck(self.output, 6, 24, 1)
        self.output = self._inverted_bottleneck(self.output, 6, 24, 0)
        self.output = self._inverted_bottleneck(self.output, 6, 32, 1)
        self.output = self._inverted_bottleneck(self.output, 6, 32, 0)
        self.output = self._inverted_bottleneck(self.output, 6, 32, 0)
        self.output = self._inverted_bottleneck(self.output, 6, 64, 1)
        self.output = self._inverted_bottleneck(self.output, 6, 64, 0)
        self.output = self._inverted_bottleneck(self.output, 6, 64, 0)
        self.output = self._inverted_bottleneck(self.output, 6, 64, 0)
        self.output = self._inverted_bottleneck(self.output, 6, 96, 0)
        self.output = self._inverted_bottleneck(self.output, 6, 96, 0)
        self.output = self._inverted_bottleneck(self.output, 6, 96, 0)
        self.output = self._inverted_bottleneck(self.output, 6, 160, 1)
        self.output = self._inverted_bottleneck(self.output, 6, 160, 0)
        self.output = self._inverted_bottleneck(self.output, 6, 160, 0)
        self.output = self._inverted_bottleneck(self.output, 6, 320, 0)

        self.output = tc.layers.conv2d(self.output, 1280, 1, normalizer_fn=self.normalizer, normalizer_params=self.bn_params)
        self.output = tc.layers.avg_pool2d(self.output, 7)
        self.output = tc.layers.conv2d(self.output, 1000, 1, activation_fn=None)


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

    def train():
        features = tf.placeholder(tf.float32, [None, img_h, img_w, 3])
        labels = tf.placeholder(tf.int64, [None, 6])
        cross_entropy = self.cost(logits, labels)

if __name__ == '__main__':
    mobilenet = MobileNet()
    #print(model.output.get_shape())
    #board_writer = tf.summary.FileWriter(logdir='./', graph=tf.get_default_graph())

    fake_data = np.ones(shape=(2, 224, 224, 3))

    #sess_config = tf.ConfigProto(device_count={'GPU':0})
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        cnt = 0
        for i in range(101):
            t1 = time.time()
            output = sess.run(mobilenet.output, feed_dict={mobilenet.input: fake_data})
            if i != 0:
                cnt += time.time() - t1
        print(cnt / 100)
