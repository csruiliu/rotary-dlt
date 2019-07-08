import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import cv2
import warnings

###############################
# convert images to bin 
###############################

def image_input(imgDir, img_w, img_h, img_num):
    all_arr = []
    count = 0
    for filename in sorted(os.listdir(imgDir)):
        #print(filename)
        count+=1
        if count > img_num:
            break
        arr_single = read_single_image(imgDir + '/'+ filename, img_w, img_h)
        if arr_single != []:
            if all_arr == []:
                all_arr = arr_single
            else:
                all_arr = np.concatenate((all_arr, arr_single))
    #print(all_arr.shape)
    return all_arr

def read_single_image(img_name, img_w, img_h):
    img = cv2.imread(img_name, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (img_w,img_h))
    img_ext = np.expand_dims(img, axis=0) 
    return img_ext

def check_channel_dir(img_dir):
    for img_name in os.listdir(img_dir):
        img = plt.imread(img_dir+'/'+img_name)
        shape = img.shape
        if len(shape) != 3:
            print(img_name)

def check_image(img_dir):
    for filename in sorted(os.listdir(img_dir)):
        img = cv2.imread(filename, cv2.IMREAD_COLOR)
        print(filename)
        try:
            img_byte = np.array(img, dtype=np.float32)
        except UserWarning:
            print('corrupt img', filename)


def pickle_save_bin(arr, output_file):
    img_data = {'image': arr}
    f = open(output_file, 'wb+')
    pickle.dump(img_data, f)
    f.close()

###############################
# convert images to tfrecord
###############################

def create_tfrecord(tf_writer, img_folder, label_file, img_w, img_h):
    with open(label_file) as lf:
        content = lf.readlines()
    labels = [x.strip() for x in content]

    #for idx, img_name in enumerate(os.listdir(img_folder)):
    #    img_path = img_folder + "/" + img_name
    #    img = Image.open(img_path)
    #    img = img.resize((img_w, img_h))
    #    img_raw = img.tobytes()
    #    example = tf.train.Example(features=tf.train.Features(feature={
    #        'label':tf.train.Feature(int64_list=tf.train.Int64List(value=[int(labels[idx])])),
    #        'img_raw':tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
    #    }))
    #    tf_writer.write(example.SerializeToString())

    #tf_writer.close()

def read_tfrecord(tf_file, img_w, img_h, num_channels):
    tf_file_queue = tf.train.string_input_producer([tf_file])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(tf_file_queue)
    features = tf.parse_single_example(serialized_example,features={
        'label': tf.FixedLenFeature([], tf.int64),
        'img_raw': tf.FixedLenFeature([], tf.string)
    })
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [img_w, img_h, num_channels])
    #img = tf.image.per_image_standardization(img)
    label = tf.cast(features['label'], tf.int32)
    return img, label

def create_batch(img, label, batch_size):
    img_batch, label_batch = tf.train.shuffle_batch([img, label], batch_size=batch_size, capacity=200, min_after_dequeue=100)
    return img_batch, label_batch

###############################
# main function
###############################

if __name__ == '__main__':

    pack_img_num = 1000
    #total_img = 149000

    #offset=int(np.random.randint(total_img-pack_img, size=1))
    #print(offset)
    imgDir = '/home/ruiliu/Development/mtml-tf/dataset/imagenet150k'
    outDir = '/home/ruiliu/Development/mtml-tf/dataset/imagenet1k.bin'
    
    arr_input = image_input(imgDir, 224, 224, pack_img_num)
    pickle_save_bin(arr_input, outDir)

    #check_image(imgDir)
    #check_channel_dir(imgDir)    

    #writer = tf.python_io.TFRecordWriter("../dataset/imagenet10k.tfrecords")
    #img_folder = "/home/ruiliu/Development/mtml-tf/dataset/imagenet10k"
    #label_file = "/home/ruiliu/Development/mtml-tf/dataset/imagenet10k-label.txt"
    #output_folder = "/home/ruiliu/Development/mtml-tf/dataset/output"
    #create_tfrecord(writer, img_folder, label_file, 224, 224)
    #img, label = read_tfrecord("../dataset/imagenet10k.tfrecords", 224, 224, 3)
    #img_batch, label_batch = create_batch(img, label, 10)
    #with tf.Session() as sess:
    #    init_op = tf.global_variables_initializer()
    #    sess.run(init_op)
    #    image, label = read_tfrecord("../dataset/imagenet10k.tfrecords", 224, 224, 3)
    #    coord = tf.train.Coordinator()
    #    threads = tf.train.start_queue_runners(coord=coord)
    #    for i in range(10):
    #        rimage, rlable = sess.run([image,label])
    #        img = Image.fromarray(rimage, 'RGB')
    #        img.save(output_folder + '/' + str(i) + '_' + 'Label_'+ str(rlable) + '.jpg') # 储存图片
    #    coord.request_stop()
    #    coord.join(threads)
