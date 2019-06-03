import tensorflow as tf

channel_num = 3

conv1_filter_size = 7 
conv1_num_filters = 64

conv2_filter_size = 3
conv2_num_filters = 64

filter_size_conv3 = 3
num_filters_conv3 = 64

class convnet(object):
    def __init__(self, net_name, model_layer, input_h, input_w, num_classes):
        self.net_name = net_name
        self.model_layer_num = model_layer
        self.img_h = input_h
        self.img_w = input_w
        self.input_size = input_h * input_w * channel_num
        self.num_classes = num_classes

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

        return layer

    def build(self, input):
        with tf.variable_scope(self.net_name + '_instance'):

            layer_conv = self.create_convolutional_layer(input=input, in_filter=3,
                                                          conv_filter_size=conv1_filter_size,
                                                          num_filters=conv1_num_filters, conv_stride=1, 
                                                          pool_stride=2, conv_padding='SAME', pool_padding='SAME')

            for _ in range(self.model_layer_num):
                layer_conv = self.create_convolutional_layer(input=layer_conv, in_filter=64, 
                                                          conv_filter_size=conv2_filter_size, 
                                                          num_filters=conv2_num_filters, conv_stride=1,
                                                          pool_stride=2, conv_padding='SAME', pool_padding='SAME')

            layer_flat = tf.layers.flatten(layer_conv)
            x = tf.layers.dense(layer_flat, units=50, activation=tf.nn.relu)
        
            logits = tf.layers.dense(x, units=self.num_classes, activation=tf.nn.softmax)
            
            print(logits)

        return logits

    def cost(self, logits, labels):
        with tf.name_scope('loss'):
            cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
        cross_entropy_cost = tf.reduce_mean(cross_entropy)
        return cross_entropy_cost

    def getModelInstanceName(self):
        return self.net_name
