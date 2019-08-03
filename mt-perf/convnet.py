import tensorflow as tf

channel_num = 3

conv1_filter_size = 7 
conv1_num_filters = 64

conv2_filter_size = 3
conv2_num_filters = 64

filter_size_conv3 = 3
num_filters_conv3 = 64

class convnet(object):
    def __init__(self, net_name, model_layer, input_h, input_w, batch_size, num_classes):
        self.net_name = net_name
        self.model_layer_num = model_layer
        self.img_h = input_h
        self.img_w = input_w
        self.input_size = input_h * input_w * channel_num
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.model_size = 0
        self.cost = 0

    def create_weights(self, shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

    def create_biases(self, size):
        return tf.Variable(tf.constant(0.05, shape=[size]))

    def create_convolutional_layer(self, input, conv_filter_size, num_filters, in_filter, conv_stride, pool_stride, conv_padding, pool_padding):
        weights = self.create_weights(shape=[conv_filter_size, conv_filter_size, in_filter, num_filters])
        biases = self.create_biases(num_filters)

        layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, conv_stride, conv_stride, 1], padding=conv_padding)
        layer += biases
        layer = tf.nn.max_pool(value=layer, ksize=[1, 3, 3, 1], strides=[1, pool_stride, pool_stride, 1], padding=pool_padding)
        layer = tf.nn.relu(layer)

        layer_size = (conv_filter_size * conv_filter_size * in_filter + 1) * num_filters
        #print(layer_size)
        return layer, layer_size

    def build(self, input):
        with tf.variable_scope(self.net_name + '_instance'):

            layer_conv, layer_conv_size = self.create_convolutional_layer(input=input, in_filter=3,
                                                          conv_filter_size=conv1_filter_size,
                                                          num_filters=conv1_num_filters, conv_stride=1,
                                                          pool_stride=2, conv_padding='SAME', pool_padding='SAME')

            self.model_size += layer_conv_size


            for _ in range(self.model_layer_num):
                layer_conv, layer_conv_size = self.create_convolutional_layer(input=layer_conv, in_filter=64,
                                                          conv_filter_size=conv2_filter_size,
                                                          num_filters=conv2_num_filters, conv_stride=1,
                                                          pool_stride=2, conv_padding='SAME', pool_padding='SAME')
                self.model_size += layer_conv_size

            #print(self.model_size)
            layer_flat = tf.layers.flatten(layer_conv)
            #print("layer_flat[0]:",layer_flat.shape[0])
            #print("layer_flat[1]:",layer_flat.shape[1])
            x = tf.layers.dense(layer_flat, units=50, activation=tf.nn.relu)
            #print(int(layer_flat.shape[1]))
            self.model_size += (int(layer_flat.shape[1]) + 1) * 50
            #print(self.model_size)
            logits = tf.layers.dense(x, units=self.num_classes, activation=tf.nn.softmax)
            self.model_size += (int(x.shape[1]) + 1) * self.num_classes
            #print(self.model_size)
        return logits

    def train(self, logits, labels):
        with tf.name_scope('loss_'+self.net_name):
            cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
            cross_entropy_cost = tf.reduce_mean(cross_entropy)
        self.cost = cross_entropy_cost

        with tf.name_scope('optimizer_'+self.net_name):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy_cost)
        return train_step

    def train_step(self, logits, labels):
        with tf.name_scope('loss_'+self.net_name):
            cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
            cross_entropy_cost = tf.reduce_mean(cross_entropy)
        self.cost = cross_entropy_cost

        with tf.name_scope('optimizer_'+self.net_name):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_optimizer = tf.train.AdamOptimizer(1e-4)
                grads_and_vars = train_optimizer.compute_gradients(cross_entropy_cost, tf.trainable_variables())
                train_ops = train_optimizer.apply_gradients(grads_and_vars)
                
        return train_optimizer, grads_and_vars, train_ops

    def getCost(self):
        return self.cost

    def getModelInstanceName(self):
        return (self.net_name + " with layer: " + str(self.model_layer_num))

    def getModelMemSize(self):
        return self.model_size * 4 / 1024
