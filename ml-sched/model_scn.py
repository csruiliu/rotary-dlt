import tensorflow as tf
from utils_model_func import activation_function


class scn(object):
    def __init__(self, net_name, num_layer, input_h, input_w, num_channel, num_classes, batch_size, opt,
                 learning_rate=0.0001, activation='relu', batch_padding=False):
        self.net_name = net_name
        self.num_layer = num_layer
        self.img_h = input_h
        self.img_w = input_w
        self.channel_num = num_channel
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.opt = opt
        self.learning_rate = learning_rate
        self.activation = activation
        self.batch_padding = batch_padding
        self.model_logit = None
        self.train_op = None
        self.eval_op = None
        self.num_conv_layer = 0
        self.num_pool_layer = 0
        self.num_total_layer = 0

    def add_layer_num(self, layer_type, layer_num):
        if layer_type == 'pool':
            self.num_pool_layer += layer_num
            self.num_total_layer += layer_num
        elif layer_type == 'conv':
            self.num_conv_layer += layer_num
            self.num_total_layer += layer_num
        elif layer_type == 'total':
            self.num_total_layer += layer_num

    def weight_variable(self, shape):
        init_w = tf.truncated_normal_initializer(stddev=0.01)
        return tf.get_variable(name='W', dtype=tf.float32, shape=shape, initializer=init_w)

    def bias_variable(self, shape):
        init_b = tf.constant(0., shape=shape, dtype=tf.float32)
        return tf.get_variable('b', dtype=tf.float32, initializer=init_b)

    def max_pool(self, x, ksize, stride, name):
        self.add_layer_num('pool', 1)
        return tf.nn.max_pool(x, ksize=[1, ksize, ksize, 1], strides=[1, stride, stride, 1], padding="SAME", name=name)

    def conv_layer(self, x, filter_size, num_filters, stride, name):
        with tf.variable_scope(name):
            filter_shape = [filter_size, filter_size, self.channel_num, num_filters]
            W = self.weight_variable(shape=filter_shape)
            b = self.bias_variable(shape=[num_filters])
            layer = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")
            self.add_layer_num('conv', 1)
            layer += b

            return activation_function(layer, self.activation)

    def fc_layer(self, x, num_units, name, use_activation=True):
        self.add_layer_num('total', 1)
        with tf.variable_scope(name):
            in_dim = x.get_shape()[1]
            W = self.weight_variable(shape=[in_dim, num_units])
            b = self.bias_variable(shape=[num_units])
            layer = tf.matmul(x, W) + b
            if use_activation:
                layer = activation_function(layer, self.activation)
            return layer

    def build(self, input):
        if self.batch_padding == True:
            input = input[0:self.batch_size, :, :, :]

        with tf.variable_scope(self.net_name + '_instance'):
            conv = self.conv_layer(input, filter_size=5, num_filters=12, stride=1, name='conv0')
            pool = self.max_pool(conv, ksize=2, stride=2, name='pool0')
            if self.num_layer >= 1:
                for midx in range(self.num_layer - 1):
                    conv = self.conv_layer(conv, filter_size=5, num_filters=12, stride=1, name='conv'+str(midx+1))
                    pool = self.max_pool(conv, ksize=2, stride=2, name='pool'+str(midx+1))

            layer_flat = tf.layers.flatten(pool, name='flat')
            self.add_layer_num('total', 1)
            layer_fc = self.fc_layer(layer_flat, num_units=128, name='fc', use_activation=True)
            self.add_layer_num('total', 1)
            self.model_logit = self.fc_layer(layer_fc, num_units=self.num_classes, name='logit', use_activation=False)
        return self.model_logit

    def train(self, logits, labels):
        if self.batch_padding == True:
            labels = labels[0:self.batch_size, :]

        with tf.name_scope('loss_' + self.net_name):
            cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
            cross_entropy_cost = tf.reduce_mean(cross_entropy)

        with tf.name_scope(self.opt + '_' + self.net_name):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.net_name + '_instance')
            with tf.control_dependencies(update_ops):
                if self.opt == 'Adam':
                    self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(cross_entropy_cost)
                elif self.opt == 'SGD':
                    self.train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(cross_entropy_cost)
                elif self.opt == 'Adagrad':
                    self.train_op = tf.train.AdagradOptimizer(self.learning_rate).minimize(cross_entropy_cost)
                elif self.opt == 'Momentum':
                    self.train_op = tf.train.MomentumOptimizer(self.learning_rate, 0.9).minimize(cross_entropy_cost)

        return self.train_op

    def evaluate(self, logits, labels):
        with tf.name_scope('eval_' + self.net_name):
            pred = tf.nn.softmax(logits)
            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(labels, 1))
            self.eval_op = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        return self.eval_op

    def get_layer_info(self):
        return self.num_conv_layer, self.num_pool_layer, self.num_total_layer

    def print_model_info(self):
        print('=====================================================================')
        print('number of conv layer: {}, number of pooling layer: {}, total layer: {}'.format(self.num_conv_layer, self.num_pool_layer, self.num_total_layer))
        print('=====================================================================')
