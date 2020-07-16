import tensorflow as tf


class SchRLModel:
    def __init__(self, n_feature, n_action_space, n_action_output, learning_rate=0.01):
        self.num_feature = n_feature
        self.num_action_space = n_action_space
        self.num_action_output = n_action_output
        self.learn_rate = learning_rate
        self.sch_logit = None

    def build_sch_model(self, model_input):
        with tf.variable_scope('sch_logit'):
            hidden_layer_neurons = 32
            variable_initializer = tf.contrib.layers.xavier_initializer()
            W1 = tf.get_variable("W1", shape=[self.num_feature, hidden_layer_neurons], initializer=variable_initializer)

        return self.sch_logit


    def train_sch_model(self):
        pass

    def run_sch_model(self):
        pass
