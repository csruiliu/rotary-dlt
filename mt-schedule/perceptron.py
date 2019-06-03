import tensorflow as tf

channel_num = 3

class perceptron(object):
    def __init__(self, net_name, model_layer, input_h, input_w, num_classes):
        self.net_name = net_name
        self.model_layer_num = model_layer
        self.img_h = input_h
        self.img_w = input_w
        self.input_size = input_h * input_w * channel_num
        self.num_classes = num_classes
            
    def perceptron_layer(self, input):
        weights = tf.Variable(tf.random_normal([int(input.shape[1]), self.num_classes]))
        biases = tf.Variable(tf.random_normal([self.num_classes]))
        layer = tf.matmul(input, weights) + biases

        return layer

    def build(self, input):
        with tf.variable_scope(self.net_name + '_instance'):
            input_image = tf.reshape(input, [-1, self.input_size])
            layer = self.perceptron_layer(input = input_image)
            for _ in range(self.model_layer_num):
                layer = self.perceptron_layer(input = layer)
            
        return layer

    def cost(self, logits, labels):
        with tf.name_scope('loss'):
            cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
        cross_entropy_cost = tf.reduce_mean(cross_entropy)
        return cross_entropy_cost

    def getModelInstanceName(self):
        return self.net_name
