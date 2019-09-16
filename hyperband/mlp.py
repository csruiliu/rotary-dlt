import tensorflow as tf

channel_num = 3

class mlp(object):
	def __init__(self, net_name, model_layer, input_h, input_w, batch_size, num_classes, opt):
        self.net_name = net_name
        self.model_layer_num = 3
        self.img_h = input_h
        self.img_w = input_w
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.optimzier = opt
        self.input_size = input_h * input_w * channel_num


	def perceptron_layer(self, input):
        weights = tf.Variable(tf.random_normal([int(input.shape[1]), self.num_classes]))
        biases = tf.Variable(tf.random_normal([self.num_classes]))
        layer = tf.matmul(input, weights) + biases
        return layer 

    def build(self, input):
        with tf.variable_scope(self.net_name + '_instance'):
            input_image = tf.reshape(input, [-1, self.input_size])
            layer, layer_size = self.perceptron_layer(input_image)
            if self.model_layer_num >= 1:
                for _ in range(self.model_layer_num - 1):
                    layer, layer_size = self.perceptron_layer(input = layer)
                    self.model_size += layer_size
            logit = tf.nn.relu(layer)
        return logit

    def train(self, logits, labels):
        with tf.name_scope('loss_'+self.net_name):
            cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
            cross_entropy_cost = tf.reduce_mean(cross_entropy)
        with tf.name_scope('optimizer_'+self.net_name):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS,scope=self.net_name+'_instance')
            with tf.control_dependencies(update_ops):
                if self.optimzier == "Adam":
                    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy_cost)
                elif self.optimzier == "SGD":
                    train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy_cost)

        return train_step

        