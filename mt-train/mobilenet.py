import tensorflow as tf
import tensorflow.contrib as tc

img_h=224
img_w=224

#MobileNetV2
class MobileNet(object):
    def __init__(self, net_name, is_training=True, input_size=224):
        self.net_name = net_name
        self.is_training = is_training
        self.normalizer = tc.layers.batch_norm
        self.bn_params = {'is_training': self.is_training}

    def build(self, input):
        with tf.variable_scope(self.net_name + '_instance'):
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
