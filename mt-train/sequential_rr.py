import numpy as np
import tensorflow as tf
from mobilenet import *
from resnet import *
from img_utils import *

img_w = 224
img_h = 224
mini_batches = 10

class Seq(object):
    def __init__(self):
        pass

    def build(self):
        resnet1 = ResNet('resnet1')
        resnet2 = ResNet('resnet2')
        return resnet1, resnet2

    def train(self, X_train, Y_train):
        resnet1, resnet2 = self.build()
        features = tf.placeholder(tf.float32, [None, img_h, img_w, 3])
        labels = tf.placeholder(tf.int64, [None, 1000])

        logits_resnet1 = resnet1.build(features)
        logits_resnet2 = resnet2.build(features)

        cross_entropy_resnet1 = resnet1.cost(logits_resnet1, labels)
        cross_entropy_resnet2 = resnet2.cost(logits_resnet2, labels)

        with tf.name_scope('optimizer'):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_step_resnet1 = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy_resnet1)
                train_step_resnet2 = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy_resnet2)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            num_batch = Y_train.shape[0] // mini_batches

            total_time = 0
            for i in range(num_batch):
                print('step %d / %d' %(i+1, num_batch))
                X_mini_batch_feed = X_train[num_batch:num_batch + mini_batches,:,:,:]
                Y_mini_batch_feed = Y_train[num_batch:num_batch + mini_batches,:]
                start_time = timer()
                train_step_resnet1.run(feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})
                train_step_resnet2.run(feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})
                end_time = timer()
                total_time += end_time - start_time
            print("training time for 1 epoch:", total_time)

def main(_):
    data_dir = '/home/rui/Development/mtml-tf/dataset/test'
    label_path = '/home/rui/Development/mtml-tf/dataset/test.txt'
    X_data = load_images(data_dir)
    Y_data = load_labels_onehot(label_path)
    seq = Seq()
    seq.train(X_data,Y_data)

if __name__ == '__main__':
    tf.app.run(main=main)
