import tensorflow as tf
from PIL import Image
import numpy as np  
import argparse
import os

#parser = argparse.ArgumentParser()
#parser.add_argument("-f", "--file", type=str, help="filename path for created tfrecord")
#args = parser.parse_args()

#tf_file = args.dataset

def create_record(tf_writer, img_folder, label_file):
    with open(label_file) as lf:
        content = lf.readlines()
    labels = [x.strip() for x in content] 
    #print(labels)       

    for idx, img_name in enumerate(os.listdir(img_folder)):
        img_path = img_folder + "/" + img_name
        img = Image.open(img_path)
        img = img.resize((224,224))
        #print(img.size)
        img_raw = img.tobytes()
        
        example = tf.train.Example(features=tf.train.Features(feature={
            'label':tf.train.Feature(int64_list=tf.train.Int64List(value=[int(labels[idx])])),
            'img_raw':tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        }))   
        tf_writer.write(example.SerializeToString())

    tf_writer.close()


def read_and_decode(tf_file):
    tf_file_queue = tf.train.string_input_producer([tf_file])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(tf_file_queue)
    features = tf.parse_single_example(serialized_example,features={
        'label': tf.FixedLenFeature([], tf.int64),
        'img_raw': tf.FixedLenFeature([], tf.string)
    })
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [224, 224, 3]) 
    #img = tf.image.per_image_standardization(img)
    label = tf.cast(features['label'], tf.int32)
    return img, label 

def create_batch(img, label, batchSize):
    img_batch, label_batch = tf.train.shuffle_batch([img, label], batch_size=batchSize, capacity=200, min_after_dequeue=100)
    return img_batch, label_batch

if __name__ == '__main__':
    writer = tf.python_io.TFRecordWriter("../dataset/imagenet1k.tfrecords")
    img_folder = "/home/ruiliu/Development/mtml-tf/dataset/imagenet1k"
    label_file = "/home/ruiliu/Development/mtml-tf/dataset/imagenet1k-label.txt"
    output_folder = "/home/ruiliu/Development/mtml-tf/dataset/output"
    create_record(writer, img_folder, label_file)
    img, label = read_and_decode("../dataset/imagenet1k.tfrecords")
    img_batch, label_batch = create_batch(img, label, 10)
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        image, label = read_and_decode("../dataset/imagenet10k.tfrecords")
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(10):
            rimage, rlable = sess.run([image,label]) 
            img = Image.fromarray(rimage, 'RGB')
            img.save(output_folder + '/' + str(i) + '_' + 'Label_'+ str(rlable) + '.jpg') # 储存图片
        coord.request_stop()
        coord.join(threads)

