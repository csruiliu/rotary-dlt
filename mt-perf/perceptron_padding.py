import tensorflow as tf

channel_num = 3

class perceptron_padding(object):
    def __init__(self, net_name, model_layer, input_h, input_w, batch_size, num_classes):
        self.net_name = net_name
        self.model_layer_num = model_layer
        self.img_h = input_h
        self.img_w = input_w
        self.input_size = input_h * input_w * channel_num
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.model_size = 0

    def perceptron_layer(self, input):
        weights = tf.Variable(tf.random_normal([int(input.shape[1]), self.num_classes]))
        biases = tf.Variable(tf.random_normal([self.num_classes]))
        layer = tf.matmul(input, weights) + biases
        layer_size = (int(input.shape[1]) + 1) * self.num_classes 

        return layer, layer_size

    def build(self, input):
        with tf.variable_scope(self.net_name + '_instance'):
            input_padding = input[0:self.batch_size,:,:,:]
            #print("padding shape:", input_padding.shape)
            input_image = tf.reshape(input_padding, [-1, self.input_size])
            layer, layer_size = self.perceptron_layer(input_image)
            self.model_size += layer_size
            for _ in range(self.model_layer_num):
                layer, layer_size = self.perceptron_layer(input = layer)
                self.model_size += layer_size

        return layer

    def cost(self, logits, labels):
        with tf.name_scope('loss_'+self.net_name):
            labels_padding = labels[0:self.batch_size,:]
            #print("labels padding shape:", labels_padding.shape)
            cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=labels_padding, logits=logits)
            cross_entropy_cost = tf.reduce_mean(cross_entropy)

        with tf.name_scope('optimizer_'+self.net_name):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy_cost)

        return train_step

    def getCost(self, logits, labels):
        labels_padding = labels[0:self.batch_size,:]
        cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=labels_padding, logits=logits)
        cross_entropy_cost = tf.reduce_mean(cross_entropy)
        return cross_entropy_cost

    def getModelInstanceName(self):
        return (self.net_name + " with layer: " + str(self.model_layer_num)) 

    def getModelMemSize(self):
        return self.model_size * 4 / (1024**2)
