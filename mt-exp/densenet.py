import tensorflow as tf
import tensorflow.contrib as tc
from tensorflow.keras.layers import GlobalAveragePooling2D

channel_num = 3
class_num = 1000
growth_k = 32

#DenseNet-121
class densenet(object):
    def __init__(self, net_name, model_layer, input_h, input_w, batch_size, num_classes, opt):
        self.net_name = net_name
        self.model_layer_num = model_layer
        self.img_h = input_h
        self.img_w = input_w
        self.input_size = input_h * input_w * channel_num
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.optimzier = opt
        self.model_size = 0
        self.cost = 0


    def bottleneck_block(self, input, block_name):
        with tf.variable_scope(block_name):
            block = tf.layers.batch_normalization(input, training=True, trainable=True, name=block_name+'_bn_0')
            block = tf.nn.relu(block, name=block_name+'_relu_0')
            block = tf.layers.conv2d(block, filters=2*growth_k, kernel_size=(1,1), strides=1, padding='same', name=block_name+'_conv_0')

            block = tf.layers.batch_normalization(block, training=True, trainable=True, name=block_name+'_bn_1')
            block = tf.nn.relu(block, name=block_name+'_relu_1')
            block = tf.layers.conv2d(block, filters=2*growth_k, kernel_size=(3,3), strides=1, padding='same', name=block_name+'_conv_1')

            return block

    def dense_block(self, input, dn_layers, block_name):
        with tf.variable_scope(block_name): 
            block = self.bottleneck_block(input, block_name=block_name+'_bottleneck_block_0')
            block_input = tf.concat(values=[input, block],axis=3)
            for i in range(dn_layers - 1):
                block = self.bottleneck_block(block_input, block_name=block_name+'_bottleneck_block_'+str(i+1))
                block_input = tf.concat([block_input,block],axis=3)
            
            return block
    
    def transition_block(self, input, block_name):
        with tf.variable_scope(block_name):
            block = tf.layers.batch_normalization(input, training=True, trainable=True, name=block_name+'_bn_0')
            block = tf.layers.conv2d(block, filters=2*growth_k, kernel_size=(1,1), strides=1, padding='same', name=block_name+'_conv_0')
            block = tf.layers.average_pooling2d(block, pool_size=(2,2), strides=2, padding='same', name=block_name+'_avgpool_0')
        
            return block

    def build(self, input):
        instance_name = self.net_name + '_instance'
        with tf.variable_scope(instance_name):
            input_padding = input[0:self.batch_size,:,:,:]
            net = tf.layers.conv2d(input_padding, filters=2*growth_k, kernel_size=(7,7), strides=2, padding='same', name='conv_0')
            net = tf.layers.max_pooling2d(net, pool_size=(3,3), strides=2, padding='same', name='max_pool_0')
            net = self.dense_block(net, dn_layers=6, block_name='dense_block_0')
            net = self.transition_block(net, block_name='trans_block_0')
            net = self.dense_block(net, dn_layers=12, block_name='dense_block_1')
            net = self.transition_block(net, block_name='trans_block_1')
            net = self.dense_block(net, dn_layers=48, block_name='dense_block_2')
            net = self.transition_block(net, block_name='trans_block_2')
            net = self.dense_block(net, dn_layers=32, block_name='dense_block_3')
            net = GlobalAveragePooling2D()(net)
            net = tc.layers.flatten(net)
            logits = tf.layers.dense(net, units=class_num, name='full_connected')
            return logits
    
    def train(self, logits, labels):
        with tf.name_scope('loss_'+self.net_name):
            labels_padding = labels[0:self.batch_size:,]
            cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=labels_padding, logits=logits)
            cross_entropy_cost = tf.reduce_mean(cross_entropy)
        
        with tf.name_scope('optimizer_'+self.net_name):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.net_name+'_instance')
            with tf.control_dependencies(update_ops):
                if self.optimzier == "Adam":
                    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy_cost)
                elif self.optimzier == "SGD":
                    train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy_cost)
        return train_step
    
    def train_step(self, logits, labels):
        with tf.name_scope('loss_'+self.net_name):
            labels_padding = labels[0:self.batch_size:,]
            cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=labels_padding, logits=logits)
            cross_entropy_cost = tf.reduce_mean(cross_entropy)

        with tf.name_scope('optimizer_'+self.net_name):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.net_name+'_instance')
            with tf.control_dependencies(update_ops):
                if self.optimzier == "Adam":
                    train_optimizer = tf.train.AdamOptimizer(1e-4)
                elif self.optimzier == "SGD":
                    train_optimizer = tf.train.GradientDescentOptimizer(1e-4)
                train_grads_and_vars = train_optimizer.compute_gradients(cross_entropy_cost, tf.trainable_variables())
                train_step = train_optimizer.apply_gradients(train_grads_and_vars)
        return train_optimizer, train_grads_and_vars, train_step

    def getModelInstanceName(self):
        return (self.net_name + " with layer: " + str(self.model_layer_num))

    
    
