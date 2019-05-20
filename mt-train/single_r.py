import tensorflow as tf
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

img_h=224
img_w=224

class Single(object):
    def __init__(self):
        pass

    def build(self):
        resnet = ResNet('resnet_single_'+str(mini_batches)+'_'+eval_options)
        return resnet

    def train(self, X_train, Y_train):
        resnet = self.build()
        features = tf.placeholder(tf.float32, [None, img_h, img_w, 3])
        labels = tf.placeholder(tf.int64, [None, 1000])

        logits_resnet = resnet.build(features)

        cross_entropy_resnet = resnet.cost(logits_resnet, labels)

        with tf.name_scope('optimizer'):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_step_resnet = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy_resnet)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            num_batch = Y_train.shape[0] // mini_batches

            total_time = 0
            if eval_options == "epoch":
                for i in range(num_batch):
                    print('step %d / %d' %(i+1, num_batch))
                    X_mini_batch_feed = X_train[num_batch:num_batch + mini_batches,:,:,:]
                    Y_mini_batch_feed = Y_train[num_batch:num_batch + mini_batches,:]
                    start_time = timer()
                    train_step_resnet.run(feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})
                    end_time = timer()
                    total_time += end_time - start_time
                print("training time for 1 epoch:", total_time)

            elif eval_options == "step":
                total_step = 0
                for i in range(num_batch):
                    print('step %d / %d' %(i+1, num_batch))
                    X_mini_batch_feed = X_train[num_batch:num_batch + mini_batches,:,:,:]
                    Y_mini_batch_feed = Y_train[num_batch:num_batch + mini_batches,:]
                    start_time = timer()
                    train_step_resnet.run(feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})
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
    X_data = load_images(data_dir)
    Y_data = load_labels_onehot(label_path)
    single_r = Single()
    single_r.train(X_data, Y_data)

if __name__ == '__main__':
    tf.app.run(main=main)
