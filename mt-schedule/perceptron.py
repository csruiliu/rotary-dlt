import tensorflow as tf

img_h = 224
img_w = 224

num_input = img_h * img_w
num_classes = 1000
n_hidden_1 = 256
n_hidden_2 = 256

class perceptron(object):
    def __init__(self, net_name):
        self.net_name = net_name
        self.weights = {
            'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
            'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
            'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes]))
	}
        self.biases = {
            'b1': tf.Variable(tf.random_normal([n_hidden_1])),
            'b2': tf.Variable(tf.random_normal([n_hidden_2])),
            'out': tf.Variable(tf.random_normal([num_classes]))
	}

    def build(self, input):
        with tf.variable_scope(self.net_name + '_instance'):
            layer_1 = tf.add(tf.matmul(input, self.weights['h1']), self.biases['b1'])
            layer_2 = tf.add(tf.matmul(layer_1, self.weights['h2']), self.biases['b2'])
            logits = tf.matmul(layer_2, self.weights['out']) + self.biases['out']

        return logits

    def cost(self, logits, labels):
        with tf.name_scope('loss'):
            cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
        cross_entropy_cost = tf.reduce_mean(cross_entropy)
        return cross_entropy_cost

    def getModelInstanceName(self):
        return self.net_name
