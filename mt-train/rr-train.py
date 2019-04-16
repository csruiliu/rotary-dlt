import sys
sys.path.append('/home/ruiliu/Development/mtml-tf/models/research/slim')
sys.path.append('/home/ruiliu/Development/mtml-tf/models/official')

import tensorflow as tf
from rr_model import rr_net


dataset_size = 1000
loop_num = 1
batch_size = 10
batch_num = dataset_size / batch_size

tf.reset_default_graph()

net = tf.Graph()
with net.as_default():
    img_path_r1 = tf.placeholder(tf.string, ())
    img_path_r2 = tf.placeholder(tf.string, ())

    batch_r1 = []
    batch_r2 = []
    pred_list_r1 = []
    pred_list_r2 = []

    for i in range(1, batch_size + 1):
        img_name = 'img' + str(i) + '.jpg'

        image_r1 = tf.image.decode_jpeg(tf.read_file(img_path_r1 + '/' + img_name))
        images_r1 = tf.expand_dims(image_r1, 0)
        images_r1 = tf.cast(images_r1, tf.float32) / 128. - 1
        images_r1.set_shape((None, None, None, 3))
        images_r1 = tf.image.resize_images(images_r1, (224, 224))
        batch_r1.append(images_r1)

        image_r2 = tf.image.decode_jpeg(tf.read_file(img_path_r2 + '/' + img_name))
        images_r2 = tf.expand_dims(image_r2, 0)
        images_r2 = tf.cast(images_r2, tf.float32) / 128. - 1
        images_r2.set_shape((None, None, None, 3))
        images_r2 = tf.image.resize_images(images_r2, (224, 224))
        batch_r2.append(images_r2)

    for j in range(batch_size):
        images_r1 = batch_r1[j]
        images_r2 = batch_r2[j]
        output_r1, output_r2 = rr_net(images_r1, images_r2)
        pred_list_r1.append(output_r1)
        pred_list_r2.append(output_r2)



    with tf.Session(graph=net) as sess:

        classification_loss = slim.losses.softmax_cross_entropy(
            predict_values, im_label)

        regularization_loss = tf.add_n(slim.losses.get_regularization_losses())
        total_loss = classification_loss + regularization_loss

        train_op = slim.learning.create_train_op(classification_loss, optimizer)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)

        slim.learning.train(
            train_op,
            logdir='/tmp/',
            number_of_steps=1000,
            save_summaries_secs=300,
            save_interval_secs=600)


        pred_r, pred_m = sess.run([pred_list_r, pred_list_m], feed_dict={img_path_r: '../data/img/img1', img_path_m: '../data/img/img2'})

        time = 0
        for i in range(loop_num):
            for j in range(1, int(batch_num) + 1):
                # img_name_r = 'img'+str(j)+'.jpg'
                # img_name_m = 'img'+str(j+1)+'.jpg'
                start = timeit.default_timer()
                folder_name_r = "../data/img/img" + str(j)
                folder_name_m = "../data/img/img" + str(1001 - j)
                x, y = sess.run([pred_list_r, pred_list_m],
                                feed_dict={img_path_r: folder_name_r, img_path_m: folder_name_m})
                stop = timeit.default_timer()
                time += stop - start
                # label_map = imagenet.create_readable_names_for_imagenet_labels()
                # print("Resnet: Top 1 prediction: ", x.argmax(),label_map[x.argmax()], x.max())
                # print("Mobilenet: Top 1 prediction: ", y.argmax(),label_map[y.argmax()], y.max())
        print("average time: ", time / float(loop_num))
