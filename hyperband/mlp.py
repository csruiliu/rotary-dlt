import tensorflow as tf

class MLP(object):
    def __init__(self, net_name, model_layer, input_h, input_w, channel_num, batch_size, num_classes, opt, epochs):
        self.net_name = net_name
        self.model_layer_num = model_layer
        self.img_h = input_h
        self.img_w = input_w
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.optimzier = opt
        self.input_size = input_h * input_w * channel_num
        self.progress = 0
        self.epochs = epochs

    def perceptron_layer(self, input):
        weights = tf.Variable(tf.random_normal([int(input.shape[1]), self.num_classes]))
        biases = tf.Variable(tf.random_normal([self.num_classes]))
        layer = tf.matmul(input, weights) + biases
        return layer 

    def build(self, input):
        input_padding = input[0:self.batch_size,:,:,:]
        with tf.variable_scope(self.net_name + '_instance'):
            input_image = tf.reshape(input_padding, [-1, self.input_size])
            layer = self.perceptron_layer(input_image)
            if self.model_layer_num >= 1:
                for _ in range(self.model_layer_num - 1):
                    layer = self.perceptron_layer(input = layer)
            logit = tf.nn.relu(layer)
        return logit

    def train(self, logits, labels):
        with tf.name_scope('loss_'+self.net_name):
            cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
            cross_entropy_cost = tf.reduce_mean(cross_entropy)

        with tf.name_scope(self.optimzier+'_'+self.net_name):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS,scope=self.net_name+'_instance')
            with tf.control_dependencies(update_ops):
                if self.optimzier == 'Adam':
                    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy_cost)
                elif self.optimzier == 'SGD':
                    train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy_cost)
                elif self.optimzier == 'Adagrad':
                    train_step = tf.train.AdagradOptimizer(1e-4).minimize(cross_entropy_cost)
                elif self.optimzier == 'Momentum':
                    train_step = tf.train.MomentumOptimizer(1e-4,0.9).minimize(cross_entropy_cost)

        return train_step
 
    def evaluate(self, logits, labels):
        with tf.name_scope('accuracy_'+self.net_name):
            pred = tf.nn.softmax(logits)
            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(labels, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        
        return accuracy

    def getProgress(self):
        return self.progress

    def getEpochs(self):
        return self.epochs

    def setProgress(self, cur_prog):
        self.progress += cur_prog
