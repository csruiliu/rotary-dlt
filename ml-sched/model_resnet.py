import tensorflow as tf
import tensorflow.contrib as tf_contrib
from utils_model_func import activation_function


class resnet(object):
    def __init__(self, net_name, num_layer, input_h, input_w, num_channel, num_classes, batch_size, opt,
                 learning_rate=0.0001, activation='relu', batch_padding=False):
        self.net_name = net_name
        self.residual_layer = num_layer
        self.img_h = input_h
        self.img_w = input_w
        self.channel_num = num_channel
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.opt = opt
        self.learning_rate = learning_rate
        self.activation = activation
        self.batch_padding = batch_padding

        self.residual_layer_list = list()
        self.model_logit = None
        self.train_op = None
        self.eval_op = None

        self.num_conv_layer = 0
        self.num_pool_layer = 0
        self.num_total_layer = 0

        self.weight_init = tf_contrib.layers.variance_scaling_initializer()
        self.weight_regularizer = tf_contrib.layers.l2_regularizer(0.0001)

    def conv_layer(self, x_input, filters, kernel=4, stride=2, padding='SAME', use_bias=True, scope='conv_0'):
        with tf.variable_scope(scope):
            layer = tf.layers.conv2d(inputs=x_input, filters=filters, kernel_size=kernel,
                                     kernel_initializer=self.weight_init, strides=stride, use_bias=use_bias,
                                     padding=padding)
            self.add_layer_num('conv', 1)
            self.add_layer_num('total', 1)
            return layer

    def fully_conneted_layer(self, x_input, units, use_bias=True, scope='fully_0'):
        with tf.variable_scope(scope):
            layer = tf.layers.flatten(x_input)
            layer = tf.layers.dense(layer, units=units, kernel_initializer=self.weight_init,
                                    kernel_regularizer=self.weight_regularizer, use_bias=use_bias)
            self.add_layer_num('total', 2)
            return layer

    def relu_layer(self, x_input):
        self.add_layer_num('total', 1)
        return tf.nn.relu(x_input)

    def global_avg_pooling(self, x_input):
        self.add_layer_num('total', 1)
        gap = tf.reduce_mean(x_input, axis=[1, 2], keepdims=True)
        return gap

    def batch_norm_layer(self, x_input, is_training=True, scope='batch_norm'):
        self.add_layer_num('total', 1)
        return tf_contrib.layers.batch_norm(x_input, decay=0.9, epsilon=1e-05, center=True, scale=True,
                                            updates_collections=None, is_training=is_training, scope=scope)

    @staticmethod
    def weight_variable(w_name, shape):
        init_w = tf.truncated_normal_initializer(stddev=0.1)
        return tf.get_variable(name=w_name, dtype=tf.float32, shape=shape, initializer=init_w)

    def residual_block(self, x_init, filters, is_training=True, use_bias=True, downsample=False, scope='resblock'):
        with tf.variable_scope(scope):
            x = self.batch_norm_layer(x_init, is_training, scope='batch_norm_0')
            x = self.relu_layer(x)

            if downsample:
                x = self.conv_layer(x, filters, kernel=3, stride=2, use_bias=use_bias, scope='conv_0')
                x_init = self.conv_layer(x_init, filters, kernel=1, stride=2, use_bias=use_bias, scope='conv_init')
            else:
                x = self.conv_layer(x, filters, kernel=3, stride=1, use_bias=use_bias, scope='conv_0')

            x = self.batch_norm_layer(x, is_training, scope='batch_norm_1')
            x = self.relu_layer(x)
            x = self.conv_layer(x, filters, kernel=3, stride=1, use_bias=use_bias, scope='conv_1')

            return x + x_init

    def bottle_residual_block(self, x_init, filters, is_training=True, use_bias=True, downsample=False,
                              scope='bottle_resblock'):
        with tf.variable_scope(scope):
            x = self.batch_norm_layer(x_init, is_training, scope='batch_norm_1x1_front')
            shortcut = tf.nn.relu(x)

            x = self.conv_layer(shortcut, filters, kernel=1, stride=1, use_bias=use_bias, scope='conv_1x1_front')
            x = self.batch_norm_layer(x, is_training, scope='batch_norm_3x3')
            x = self.relu_layer(x)

            if downsample:
                x = self.conv_layer(x, filters, kernel=3, stride=2, use_bias=use_bias, scope='conv_0')
                shortcut = self.conv_layer(shortcut, filters*4, kernel=1, stride=2, use_bias=use_bias, scope='conv_init')
            else:
                x = self.conv_layer(x, filters, kernel=3, stride=1, use_bias=use_bias, scope='conv_0')
                shortcut = self.conv_layer(shortcut, filters*4, kernel=1, stride=1, use_bias=use_bias, scope='conv_init')

            x = self.batch_norm_layer(x, is_training, scope='batch_norm_1x1_back')
            x = self.relu_layer(x)
            x = self.conv_layer(x, filters*4, kernel=1, stride=1, use_bias=use_bias, scope='conv_1x1_back')

            return x + shortcut

    def build(self, input_features, is_training):
        if self.batch_padding:
            train_input = input_features[0:self.batch_size, :, :, :]
        else:
            train_input = input_features

        with tf.variable_scope(self.net_name + '_instance'):
            if self.residual_layer < 50:
                residual_block = self.residual_block
            else:
                residual_block = self.bottle_residual_block

            self.get_residual_layer()
            ch = 32

            ########################################################################################################

            x = self.conv_layer(train_input, filters=ch, kernel=3, stride=1, scope='conv')

            for i in range(self.residual_layer_list[0]):
                x = residual_block(x, filters=ch, is_training=is_training, downsample=False, scope='resblock0_'+str(i))

            ########################################################################################################

            x = residual_block(x, filters=ch*2, is_training=is_training, downsample=True, scope='resblock1_0')

            for i in range(1, self.residual_layer_list[1]):
                x = residual_block(x, filters=ch*2, is_training=is_training, downsample=False, scope='resblock1_'+str(i))

            ########################################################################################################

            x = residual_block(x, filters=ch*4, is_training=is_training, downsample=True, scope='resblock2_0')

            for i in range(1, self.residual_layer_list[2]):
                x = residual_block(x, filters=ch*4, is_training=is_training, downsample=False, scope='resblock2_'+str(i))

            ########################################################################################################

            x = residual_block(x, filters=ch*8, is_training=is_training, downsample=True, scope='resblock_3_0')

            for i in range(1, self.residual_layer_list[3]):
                x = residual_block(x, filters=ch * 8, is_training=is_training, downsample=False, scope='resblock_3_'+str(i))

            ########################################################################################################

            x = self.batch_norm_layer(x, is_training, scope='batch_norm')
            x = self.relu_layer(x)

            x = self.global_avg_pooling(x)
            logit = self.fully_conneted_layer(x, units=self.num_classes, scope='logit')

            return logit

    def train(self, logits, input_labels):
        if self.batch_padding:
            batch_labels = input_labels[0:self.batch_size, :]
        else:
            batch_labels = input_labels

        with tf.name_scope('loss_'+self.net_name):
            cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=batch_labels, logits=logits)
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

    def evaluate(self, logits, eval_labels):
        with tf.name_scope('eval_' + self.net_name):
            pred = tf.nn.softmax(logits)
            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(eval_labels, 1))
            self.eval_op = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        return self.eval_op

    def get_residual_layer(self):
        if self.residual_layer == 18:
            self.residual_layer_list = [2, 2, 2, 2]
        elif self.residual_layer == 34:
            self.residual_layer_list = [3, 4, 6, 3]
        elif self.residual_layer == 50:
            self.residual_layer_list = [3, 4, 6, 3]
        elif self.residual_layer == 101:
            self.residual_layer_list = [3, 4, 23, 3]
        elif self.residual_layer == 152:
            self.residual_layer_list = [3, 8, 36, 3]
        else:
            raise ValueError('[ResNet] residual layer is invalid')

    def add_layer_num(self, layer_type, layer_num):
        if layer_type == 'pool':
            self.num_pool_layer += layer_num
            self.num_total_layer += layer_num
        elif layer_type == 'conv':
            self.num_conv_layer += layer_num
            self.num_total_layer += layer_num
        elif layer_type == 'total':
            self.num_total_layer += layer_num

    def get_layer_info(self):
        return self.num_conv_layer, self.num_pool_layer, self.num_total_layer

    def print_model_info(self):
        print('=====================================================================')
        print('number of conv layer: {}, number of pooling layer: {}, total layer: {}'.format(self.num_conv_layer, self.num_pool_layer, self.num_total_layer))
        print('=====================================================================')


