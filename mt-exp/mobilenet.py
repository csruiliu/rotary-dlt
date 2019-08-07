import tensorflow as tf
import tensorflow.contrib as tc

channel_num = 3
weight_decay = 1e-4
exp = 6  

#MobileNetV2
class mobilenet(object):
    def __init__(self, net_name, model_layer, input_h, input_w, batch_size, num_classes, opt, is_training=True):
        self.net_name = net_name
        self.model_layer_num = model_layer
        self.img_h = input_h
        self.img_w = input_w
        self.batch_size = batch_size
        self.input_size = input_h * input_w * channel_num
        self.num_classes = num_classes
        self.optimzier = opt
        self.is_training = is_training
        self.normalizer = tc.layers.batch_norm
        self.bn_params = {'is_training': self.is_training}
        self.model_size = 0
        self.cost = 0

    def build(self, input):
        instance_name = self.net_name + '_instance'
        with tf.variable_scope(instance_name):
            net = self._conv2d_block(input, 32, 3, 2, self.is_training, 'conv1_1')
            net = self._res_block(net, 1, 16, 1, self.is_training, block_name='res2_1')
            net = self._res_block(net, exp, 24, 2, self.is_training, block_name='res3_1')  # size/4
            net = self._res_block(net, exp, 24, 1, self.is_training, block_name='res3_2')

            net = self._res_block(net, exp, 32, 2, self.is_training, block_name='res4_1')  # size/8
            net = self._res_block(net, exp, 32, 1, self.is_training, block_name='res4_2')
            net = self._res_block(net, exp, 32, 1, self.is_training, block_name='res4_3')

            net = self._res_block(net, exp, 64, 2, self.is_training, block_name='res5_1')
            net = self._res_block(net, exp, 64, 1, self.is_training, block_name='res5_2')
            net = self._res_block(net, exp, 64, 1, self.is_training, block_name='res5_3')
            net = self._res_block(net, exp, 64, 1, self.is_training, block_name='res5_4')

            net = self._res_block(net, exp, 96, 1, self.is_training, block_name='res6_1')  # size/16
            net = self._res_block(net, exp, 96, 1, self.is_training, block_name='res6_2')
            net = self._res_block(net, exp, 96, 1, self.is_training, block_name='res6_3')

            net = self._res_block(net, exp, 160, 2, self.is_training, block_name='res7_1')  # size/32
            net = self._res_block(net, exp, 160, 1, self.is_training, block_name='res7_2')
            net = self._res_block(net, exp, 160, 1, self.is_training, block_name='res7_3')

            net = self._res_block(net, exp, 320, 1, self.is_training, block_name='res8_1', shortcut=False)
            net = self._pwise_block(net, 1280, self.is_training, block_name='conv9_1')
            net = self._global_avg(net)

            logits = tc.layers.flatten(self._conv_1x1(net, self.num_classes, name='logits'))

            return logits 

    def _res_block(self, input, expansion_ratio, output_dim, stride, is_train, block_name, bias=False, shortcut=True):
        with tf.variable_scope(block_name):
            bottleneck_dim = round(expansion_ratio * input.get_shape().as_list()[-1])
            net = self._conv_1x1(input, bottleneck_dim, name='pw', bias=bias)
            net = self._batch_norm(net, train=is_train, name='pw_bn')
            net = tf.nn.relu6(net, 'relu6')

            net = self._dwise_conv(net, strides=[1, stride, stride, 1], name='dw', bias=bias)
            net = self._batch_norm(net, train=is_train, name='dw_bn')
            net = tf.nn.relu6(net, 'relu6')

            net = self._conv_1x1(net, output_dim, name='pw_linear', bias=bias)
            net = self._batch_norm(net, train=is_train, name='pw_linear_bn')

            # element wise add, only for stride==1
            if shortcut and stride == 1:
                in_dim=int(input.get_shape().as_list()[-1])
                if in_dim != output_dim:
                    ins = self._conv_1x1(input, output_dim, name='ex_dim')
                    net = ins + net
                else:
                    net = input+net
            
            return net
            
    def _conv2d_block(self, input, out_dim, kernel_size, strides_size, is_train, block_name):
        with tf.variable_scope(block_name):
            block = self._conv2d(input, out_dim, kernel_size, kernel_size, strides_size, strides_size)
            block = self._batch_norm(block, train=is_train, name='bn')
            block = tf.nn.relu6(block, 'relu6')
            return block

    def _pwise_block(self, input, output_dim, is_train, block_name, bias=False):
        with tf.variable_scope(block_name):
            out = self._conv_1x1(input, output_dim, bias=bias, name='pwb')
            out = self._batch_norm(out, train=is_train, name='bn')
            block = tf.nn.relu6(out, 'relu6')
            return block


    def _conv2d(self, input, output_dim, kernel_height, kernel_width, strides_h, strides_w, stddev=0.02, bias=False, name='conv2d'):
        with tf.variable_scope(name):
            w = tf.get_variable('w', [kernel_height, kernel_width, input.get_shape()[-1], output_dim],
                regularizer = tc.layers.l2_regularizer(weight_decay),
                initializer = tf.truncated_normal_initializer(stddev=stddev))
            conv = tf.nn.conv2d(input, w, strides=[1, strides_h, strides_w, 1], padding='SAME')
            if bias:
                biases = tf.get_variable('bias', [output_dim], initializer = tf.constant_initializer(0.0))
                conv = tf.nn.bias_add(conv, biases)
            return conv

    def _dwise_conv(self, input, kernel_height=3, kernel_width=3, channel_multiplier=1, strides=[1,1,1,1], padding='SAME', stddev=0.02, name='dwise_conv', bias=False):
        with tf.variable_scope(name):
            in_channel = input.get_shape().as_list()[-1]
            w = tf.get_variable('w', [kernel_height, kernel_width, in_channel, channel_multiplier],
                        regularizer=tc.layers.l2_regularizer(weight_decay),
                        initializer=tf.truncated_normal_initializer(stddev=stddev))
            conv = tf.nn.depthwise_conv2d(input, w, strides, padding, rate=None, name=None, data_format=None)
            if bias:
                biases = tf.get_variable('bias', [in_channel*channel_multiplier], initializer=tf.constant_initializer(0.0))
                conv = tf.nn.bias_add(conv, biases)
            return conv

    def _batch_norm(self, x, momentum=0.9, epsilon=1e-5, train=True, name='bn'):
        return tf.layers.batch_normalization(x, momentum=momentum, epsilon=epsilon, scale=True, training=train, name=name)

    def _conv_1x1(self, input, output_dim, name, bias=False):
        with tf.name_scope(name):
            return self._conv2d(input, output_dim, 1,1,1,1, stddev=0.02, name=name, bias=bias)

    def _global_avg(self, x):
        with tf.name_scope('global_avg'):
            net = tf.layers.average_pooling2d(x, x.get_shape()[1:-1], 1)
            return net

    def train(self, logits, labels):
        with tf.name_scope('loss_'+self.net_name):
            #cross_entropy = tf.losses.hinge_loss(labels=labels, logits=logits)
            #cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
            cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
            cross_entropy_cost = tf.reduce_mean(cross_entropy)
        self.cost = cross_entropy_cost

        with tf.name_scope('optimizer_'+self.net_name):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                if self.optimzier == "Adam":
                    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy_cost)
                elif self.optimzier == "SGD":
                    train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy_cost)
        return train_step

    def train_step(self, logits, labels):
        with tf.name_scope('loss_'+self.net_name):
            #cross_entropy = tf.losses.hinge_loss(labels=labels, logits=logits)
            #cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
            cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
            cross_entropy_cost = tf.reduce_mean(cross_entropy)
        self.cost = cross_entropy_cost

        with tf.name_scope('optimizer_'+self.net_name):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                if self.optimzier == "Adam":
                    train_optimizer = tf.train.AdamOptimizer(1e-4)
                elif self.optimzier == "SGD":
                    train_optimizer = tf.train.GradientDescentOptimizer(1e-4)
                train_grads_and_vars = train_optimizer.compute_gradients(cross_entropy_cost, tf.trainable_variables())
                #train_vars_with_grads = [v for g, v in train_grads_and_vars if g is not None]
                train_ops = train_optimizer.apply_gradients(train_grads_and_vars)
        return train_optimizer, train_grads_and_vars, train_ops


    def getCost(self):
        return self.cost

    def getModelInstanceName(self):
        return (self.net_name + " with layer: " + str(self.model_layer_num))

    def getModelMemSize(self):
        return self.model_size * 4 / (1024**2)
