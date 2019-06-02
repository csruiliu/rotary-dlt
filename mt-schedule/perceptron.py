import tensorflow as tf

img_h = 224
img_w = 224
channel_num = 3

input_size = img_h * img_w * channel_num
classes_num = 1000

class perceptron(object):
    def __init__(self, net_name):
        self.net_name = net_name
    
    def build(self, input):
        with tf.variable_scope(self.net_name + '_instance'):
            input_image = tf.reshape(input, [-1, input_size])

            weights = tf.Variable(tf.random_normal([input_size, classes_num]))
            biases = tf.Variable(tf.random_normal([classes_num]))
            
            logits = tf.matmul(input_image, weights) + biases

        return logits

    def cost(self, logits, labels):
        with tf.name_scope('loss'):
            cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
        cross_entropy_cost = tf.reduce_mean(cross_entropy)
        return cross_entropy_cost

    def getModelInstanceName(self):
        return self.net_name
