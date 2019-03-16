import sys
sys.path.append('/home/ruiliu/Development/tf-exp/models/research/slim')
sys.path.append('/home/ruiliu/Development/tf-exp/models/official')

import tensorflow as tf
from nets import resnet_v2
from datasets import imagenet
from nets.mobilenet import mobilenet_v2
import timeit
from mr_model import filter_net

tf.reset_default_graph()

img_w = 224
img_h = 224
img_size = img_w * img_h
img_channel = 3

net = tf.Graph()
with net.as_default():
    file_input1 = tf.placeholder(tf.string, ())
    file_input2 = tf.placeholder(tf.string, ())
    image1 = tf.image.decode_jpeg(tf.read_file(file_input1))
    image2 = tf.image.decode_jpeg(tf.read_file(file_input2))
    images1 = tf.expand_dims(image1, 0)
    images2 = tf.expand_dims(image2, 0)
    images1 = tf.cast(images1, tf.float32) / 128.  - 1
    images2 = tf.cast(images2, tf.float32) / 128.  - 1
    images1.set_shape((None, None, None, 3))
    images2.set_shape((None, None, None, 3))
    images1 = tf.image.resize_images(images1, (224, 224))
    images2 = tf.image.resize_images(images2, (224, 224))
    output1, output2 = filter_net(images1, images2)
    saver1 = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='resnet_v2_50'))
    saver2 = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='MobilenetV2') )

    with tf.Session(graph=net) as sess:
        checkpoint_r = "/home/ruiliu/Development/tf-exp/ckpts/resnet2/resnet_v2_50.ckpt"
        saver1.restore(sess,  checkpoint_r)
        checkpoint_m = "/home/ruiliu/Development/tf-exp/ckpts/mobilenet2/mobilenet_v2_1.0_224.ckpt"
        saver2.restore(sess,  checkpoint_m)
        for i in range(20):
            if i == 19:
                start = timeit.default_timer()
                x, y = sess.run([output1, output2], feed_dict={file_input1: '../data/airplane224.jpg', file_input2: '../data/panda224.jpg'})
                stop = timeit.default_timer()
                print("time", stop-start)
            else:
                x, y = sess.run([output1, output2], feed_dict={file_input1: '../data/airplane224.jpg', file_input2: '../data/panda224.jpg'})

        label_map = imagenet.create_readable_names_for_imagenet_labels()
        print("Resnet: Top 1 prediction: ", x.argmax(),label_map[x.argmax()], x.max())
        print("Mobilenet: Top 1 prediction: ", y.argmax(),label_map[y.argmax()], y.max())
