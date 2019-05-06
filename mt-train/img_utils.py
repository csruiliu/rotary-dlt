import tensorflow as tf
import cv2

def load_images(path):
    images = tf.gfile.ListDirectory(path)
    imgs_num = len(images)
    img_list = []
    for idx, img in enumerate(images):
        im = cv2.imread(path+"/"+img, cv2.IMREAD_COLOR)
        res = cv2.resize(im, dsize=(224, 224))
        res_exp = np.expand_dims(res, axis=0)
        img_list.append(res_exp)
    img_data = np.concatenate(img_list, axis=0)
    return img_data_batch

def load_labels_onehot(path):
    lines = open(path).readlines()
    labels_array = np.zeros((50000, 1000))
    for idx, val in enumerate(lines):
        hot = int(val.rstrip("\n"))
        labels_array[idx,hot-1] = 1
    return labels_array

def load_images_tf(path):
    images = tf.gfile.ListDirectory(path)
    imgs_num = len(images)
    print(imgs_num)
    img_data_list = []
    for idx, img in enumerate(images):
        image_raw_data = tf.gfile.GFile(path+"/"+img,'rb').read()
        img_data = tf.image.decode_jpeg(image_raw_data, channels = 3)
        img_data_float = tf.image.convert_image_dtype(img_data, dtype = tf.float32)
        img_data_resize = tf.image.resize_images(img_data_float, (224, 224))
        img_data_exp = tf.expand_dims(img_data_resize, 0)
        img_data_list.append(img_data_exp)
    img_data_batch = tf.concat(img_data_list, 0)
    print(img_data_batch.shape)
    return img_data_batch

def load_labels_onehot_tf(path):
    lines = open(path).readlines()
    #depth = len(lines)
    labels_list = []
    for idx, val in enumerate(lines):
        labels_list.append(int(val.rstrip("\n")))
    labels = tf.one_hot(labels_list, depth=1000)
    return labels

def mini_batches(x_data,y_data, mini_batch_size):
    num_img = y_data.shape[0]
    num_batch = num_img / mini_batch_size

    #X[mini_batch_size, :, :, :]
    #Y[mini_batch_size, :]
