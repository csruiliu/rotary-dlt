import tensorflow as tf

img_h = 224
img_w = 224
channel_num = 3

filter_size_conv1 = 3 
num_filters_conv1 = 64

filter_size_conv2 = 3
num_filters_conv2 = 64

filter_size_conv3 = 3
num_filters_conv3 = 128

fc_layer_size = 256

input_size = img_h * img_w * channel_num
classes_num = 1000


class convnet(object):
    def __init__(self, net_name):
        self.net_name = net_name

    def create_weights(self, shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

    def create_biases(self, size):
        return tf.Variable(tf.constant(0.05, shape=[size]))
    
    def create_convolutional_layer(self, input, channel_num, conv_filter_size, num_filters):  
        weights = self.create_weights(shape=[conv_filter_size, conv_filter_size, channel_num, num_filters])
        biases = self.create_biases(num_filters)

        layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1],padding='SAME')
        layer += biases
        layer = tf.nn.max_pool(value=layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        layer = tf.nn.relu(layer)

        return layer

    def create_flatten_layer(self, layer):
        layer = tf.reshape(layer, [-1, input_size])
        return layer

    def create_fc_layer(self, input, num_inputs, num_outputs, use_relu=True):
        weights = self.create_weights(shape=[num_inputs, num_outputs])
        biases = self.create_biases(num_outputs)

        layer = tf.matmul(input, weights) + biases
        if use_relu:
            layer = tf.nn.relu(layer)

        return layer

    def build(self, input):
        with tf.variable_scope(self.net_name + '_instance'):
            layer_conv1 = self.create_convolutional_layer(input=input, channel_num=channel_num,
                                                          conv_filter_size=filter_size_conv1,
                                                          num_filters=num_filters_conv1)
            layer_conv2 = self.create_convolutional_layer(input=layer_conv1, channel_num=channel_num, 
                                                          conv_filter_size=filter_size_conv2, 
                                                          num_filters=num_filters_conv2)
            layer_conv3= self.create_convolutional_layer(input=layer_conv2, channel_num=channel_num,
                                                         conv_filter_size=filter_size_conv3,
                                                         num_filters=num_filters_conv3)
            layer_flat = self.create_flatten_layer(layer_conv3)

            layer_fc1 = self.create_fc_layer(input=layer_flat, num_inputs=input_size, 
                                             num_outputs=fc_layer_size, use_relu=True)

            layer_fc2 = self.create_fc_layer(input=layer_fc1, num_inputs=fc_layer_size,
                                             num_outputs=classes_num, use_relu=False) 

            logits = tf.nn.softmax(layer_fc2)
        return logits

    def cost(self, logits, labels):
        with tf.name_scope('loss'):
            cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
        cross_entropy_cost = tf.reduce_mean(cross_entropy)
        return cross_entropy_cost

    def getModelInstanceName(self):
        return self.net_name
