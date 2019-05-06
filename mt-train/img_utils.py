import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def load_images(path):
    images = tf.gfile.ListDirectory(path)
    imgs_num = len(images)
    print(imgs_num)
    img_data_list = []
    for idx, img in enumerate(images):
        image_raw_data = tf.gfile.GFile(path+"/"+img).read()
        img_data = tf.image.decode_jpeg(image_raw_data, channels = 3)
        img_data_float = tf.image.convert_image_dtype(img_data, dtype = tf.float32)
        img_data_resize = tf.image.resize_images(img_data_float, (224, 224))
        img_data_exp = tf.expand_dims(img_data_resize, 0)
        img_data_list.append(img_data_exp)
    img_data_batch = tf.concat(img_data_list, 0)
    print(img_data_batch.shape)
    return img_data_batch

def load_labels_onehot(path):
    lines = open(path).readlines()
    #depth = len(lines)
    labels_list = []
    for idx, val in enumerate(lines):
        labels_list.append(int(val.rstrip("\n")))
    labels = tf.one_hot(labels_list, depth=1000)
    return labels

def random_mini_batches(X, Y, mini_batch_size=64, seed=None):
    """
    Creates a list of random minibatches from (X, Y)

    Arguments:
    X -- input data, of shape (input size, number of examples) (m, Hi, Wi, Ci)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples) (m, n_y)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.

    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """

    m = X.shape[0]  # number of training examples
    mini_batches = []
    np.random.seed(seed)

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :, :, :]
    shuffled_Y = Y[permutation, :]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = int(math.floor(m / mini_batch_size))
    print(num_complete_minibatches)
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :, :]
        mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size: m, :, :, :]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches
