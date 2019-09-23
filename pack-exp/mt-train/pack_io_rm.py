import tensorflow as tf
from mobilenet import *
from resnet import *
from img_utils import *
from timeit import default_timer as timer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--batch", type=int, default=10, help="batch size")
parser.add_argument("-e", "--evaluation", action='store', dest='evaluation', type=str, default="epoch", choices=["epoch", "step"], help="evaluation option")
args = parser.parse_args()

mini_batches = args.batch
eval_options = args.evaluation

img_w = 224
img_h = 224

class PackIO(object):
    def __init__(self):
        pass

    def build(self):
        resnet = ResNet('resnet_packio_'+str(mini_batches)+'_'+eval_options)
        mobilenet = MobileNet('mobilenet_packio_'+str(mini_batches)+'_'+eval_options)
        return resnet, mobilenet

    def train(self, X_train_a, Y_train_a, X_train_b, Y_train_b):
        resnet, mobilenet = self.build()
        features_a = tf.placeholder(tf.float32, [None, img_h, img_w, 3])
        labels_a = tf.placeholder(tf.int64, [None, 1000])
        features_b = tf.placeholder(tf.float32, [None, img_h, img_w, 3])
        labels_b = tf.placeholder(tf.int64, [None, 1000])

        logits_resnet = resnet.build(features_a)
        logits_mobilenet = mobilenet.build(features_b)

        cross_entropy_resnet = resnet.cost(logits_resnet, labels_a)
        cross_entropy_mobilenet = mobilenet.cost(logits_mobilenet, labels_b)

        with tf.name_scope('optimizer'):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_step_resnet = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy_resnet)
                train_step_mobilenet = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy_mobilenet)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            num_batch = Y_train_a.shape[0] // mini_batches

            total_time = 0
            if eval_options == "epoch":
                for i in range(num_batch):
                    print('step %d / %d' %(i+1, num_batch))
                    X_mini_batch_feed_a = X_train_a[num_batch:num_batch + mini_batches,:,:,:]
                    Y_mini_batch_feed_a = Y_train_a[num_batch:num_batch + mini_batches,:]
                    X_mini_batch_feed_b = X_train_b[num_batch:num_batch + mini_batches,:,:,:]
                    Y_mini_batch_feed_b = Y_train_b[num_batch:num_batch + mini_batches,:]
                    start_time = timer()
                    sess.run([train_step_resnet, train_step_mobilenet], feed_dict={features_a: X_mini_batch_feed_a, labels_a: Y_mini_batch_feed_a, features_b: X_mini_batch_feed_b, labels_b: Y_mini_batch_feed_b})
                    end_time = timer()
                    total_time += end_time - start_time
                print("training time for 1 epoch:", total_time)

            elif eval_options == "step":
                total_step = 0
                for i in range(num_batch):
                    print('step %d / %d' %(i+1, num_batch))
                    X_mini_batch_feed_a = X_train_a[num_batch:num_batch + mini_batches,:,:,:]
                    Y_mini_batch_feed_a = Y_train_a[num_batch:num_batch + mini_batches,:]
                    X_mini_batch_feed_b = X_train_b[num_batch:num_batch + mini_batches,:,:,:]
                    Y_mini_batch_feed_b = Y_train_b[num_batch:num_batch + mini_batches,:]
                    start_time = timer()
                    sess.run([train_step_resnet, train_step_mobilenet], feed_dict={features_a: X_mini_batch_feed_a, labels_a: Y_mini_batch_feed_a, features_b: X_mini_batch_feed_b, labels_b: Y_mini_batch_feed_b})
                    end_time = timer()
                    if (i+1) % 5 == 0:
                        total_step += 1
                        total_time += end_time - start_time
                step_time = total_time / total_step
                print("sampled steps:", total_step)
                print("training time for 1 step:", step_time)

def main(_):
    data_dir = '/home/ruiliu/Development/mtml-tf/dataset/test'
    label_path = '/home/ruiliu/Development/mtml-tf/dataset/test-label.txt'
    X_data_a = load_images(data_dir)
    Y_data_a = load_labels_onehot(label_path)
    X_data_b = load_images(data_dir)
    Y_data_b = load_labels_onehot(label_path)
    pack_io_rm = PackIO()
    pack_io_rm.train(X_data_a,Y_data_a,X_data_b,Y_data_b)

if __name__ == '__main__':
    tf.app.run(main=main)