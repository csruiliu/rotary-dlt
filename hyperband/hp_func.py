import tensorflow as tf
import numpy as np
import itertools
from operator import itemgetter
from datetime import datetime
import sys
from matplotlib import pyplot as plt

from img_utils import * 
from mobilenet import MobileNet
from mlp import MLP

imgWidth = 28
imgHeight = 28
numChannels = 1
numClasses = 10

data_eval_slice = 20 

#bin_dir = '/tank/local/ruiliu/dataset/imagenet1k.bin'
#bin_dir = '/home/ruiliu/Development/mtml-tf/dataset/imagenet1k.bin'
#label_path = '/tank/local/ruiliu/dataset/imagenet1k-label.txt'
#label_path = '/home/ruiliu/Development/mtml-tf/dataset/imagenet1k-label.txt'

#mnist_train_img_path = '/home/ruiliu/Development/mtml-tf/dataset/mnist-train-images.idx3-ubyte'
mnist_train_img_path = '/tank/local/ruiliu/dataset/mnist-train-images.idx3-ubyte'
#mnist_train_label_path = '/home/ruiliu/Development/mtml-tf/dataset/mnist-train-labels.idx1-ubyte'
mnist_train_label_path = '/tank/local/ruiliu/dataset/mnist-train-labels.idx1-ubyte'
#mnist_t10k_img_path = '/home/ruiliu/Development/mtml-tf/dataset/mnist-t10k-images.idx3-ubyte'
mnist_t10k_img_path = '/tank/local/ruiliu/dataset/mnist-t10k-images.idx3-ubyte'
#mnist_t10k_label_path = '/home/ruiliu/Development/mtml-tf/dataset/mnist-t10k-labels.idx1-ubyte'
mnist_t10k_label_path = '/tank/local/ruiliu/dataset/mnist-t10k-labels.idx1-ubyte'


def get_params(n_conf):
    batch_size = np.arange(10,61,5)
    opt_conf = ['Adam','SGD','Adagrad','Momentum']
    model_layer = np.arange(0,6,1)
    #data_conf = ['Same','Diff']
    #preprocess_list = ['Include','Not Include']
    #all_conf = [batch_size,opt_conf,data_conf,preprocess_list]
    all_conf = [batch_size, opt_conf, model_layer]

    hp_conf = list(itertools.product(*all_conf))
    #print(len(hp_conf))
    idx_list = np.random.choice(np.arange(0, len(hp_conf)), n_conf, replace=False)
    rand_conf = itemgetter(*idx_list)(hp_conf)

    return rand_conf

def run_params_pack_random():
    print("run random packing")

def run_params_pack_stack():
    print("run packing stack")

def run_params_pack_pool():
    print("run packing pool")

def run_params_pack_naive(batch_size, confs, iterations, conn):
    features = tf.placeholder(tf.float32, [None, imgWidth, imgHeight, numChannels])
    labels = tf.placeholder(tf.int64, [None, numClasses])
    X_data = load_mnist_image(mnist_train_img_path)
    Y_data = load_mnist_label_onehot(mnist_train_label_path)
    X_data_eval = load_mnist_image(mnist_t10k_img_path)
    Y_data_eval = load_mnist_label_onehot(mnist_t10k_label_path)

    if len(confs) == 2:
        dt = datetime.now()
        np.random.seed(dt.microsecond)
        net_instnace = np.random.randint(sys.maxsize)
        
        opt = confs[0]
        model_layer = confs[1]
        #print("\n*** batch size: {} | opt: {} | model layer: {}***".format(batch_size, opt, model_layer))
        
        #modelEntity = MobileNet("mobilenet_"+str(net_instnace), 1, imgHeight, imgWidth, batch_size, numClasses, opt[0])
        modelEntity = MLP("mlp_"+str(net_instnace), model_layer, imgHeight, imgWidth, numChannels, batch_size, numClasses, opt)
        modelLogit = modelEntity.build(features)
        trainOps = modelEntity.train(modelLogit, labels)
        evalOps = modelEntity.evaluate(modelLogit, labels)
        acc_pack = []
        config = tf.ConfigProto()
        config.allow_soft_placement = True    
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            num_batch = Y_data.shape[0] // batch_size
            for e in range(iterations):
                for i in range(num_batch):
                    print('epoch %d / %d, step %d / %d' %(e+1, iterations, i+1, num_batch))
                    batch_offset = i * batch_size
                    batch_end = (i+1) * batch_size
                    X_mini_batch_feed = X_data[batch_offset:batch_end,:,:,:]
                    Y_mini_batch_feed = Y_data[batch_offset:batch_end,:]
                    sess.run(trainOps, feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})
            
            acc_arg = evalOps.eval({features: X_data_eval, labels: Y_data_eval})
            acc_pack.append(acc_arg)
            conn.send(acc_pack)
            conn.close()
            print("Accuracy:", acc_pack)
    else:
        dt = datetime.now()
        np.random.seed(dt.microsecond)
        #print(len(confs))
        net_instnace = np.random.randint(sys.maxsize, size=len(confs)//2)
        
        train_pack = []
        eval_pack = [] 
        acc_pack = []

        for idx, _ in enumerate(confs):
            if (idx+1) % 2 == 0:
                modelEntity = MLP("mlp_"+str(net_instnace[(idx-1)//2]), confs[idx], imgHeight, imgWidth, numChannels, batch_size, numClasses, confs[idx-1])
                modelLogit = modelEntity.build(features)
                trainOps = modelEntity.train(modelLogit, labels)
                evalOps = modelEntity.evaluate(modelLogit, labels)
                train_pack.append(trainOps)
                eval_pack.append(evalOps)
        
        config = tf.ConfigProto()
        config.allow_soft_placement = True   
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            num_batch = Y_data.shape[0] // batch_size
            for e in range(iterations):
                for i in range(num_batch):
                    print('epoch %d / %d, step %d / %d' %(e+1, iterations, i+1, num_batch))
                    batch_offset = i * batch_size
                    batch_end = (i+1) * batch_size
                    X_mini_batch_feed = X_data[batch_offset:batch_end,:,:,:]
                    Y_mini_batch_feed = Y_data[batch_offset:batch_end,:]
                    sess.run(train_pack, feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})
            
            for evalOps in eval_pack:
                acc_arg = evalOps.eval({features: X_data_eval, labels: Y_data_eval})
                acc_pack.append(acc_arg)
            
            conn.send(acc_pack)
            conn.close()
            print("Accuracy:", acc_pack)
    
def run_params(hyper_params, iterations, conn):
    features = tf.placeholder(tf.float32, [None, imgWidth, imgHeight, numChannels])
    labels = tf.placeholder(tf.int64, [None, numClasses])
    X_data = load_mnist_image(mnist_train_img_path)
    Y_data = load_mnist_label_onehot(mnist_train_label_path)
    X_data_eval = load_mnist_image(mnist_t10k_img_path)
    Y_data_eval = load_mnist_label_onehot(mnist_t10k_label_path)

    dt = datetime.now()
    np.random.seed(dt.microsecond)
    net_instnace = np.random.randint(sys.maxsize)
    batch_size = hyper_params[0]
    opt = hyper_params[1]
    model_layer = hyper_params[2]
    print("\n*** batch size: {} | opt: {} | model layer: {}***".format(batch_size, opt, model_layer))

    #modelEntity = MobileNet("mobilenet_"+str(net_instnace), 1, imgHeight, imgWidth, batch_size, numClasses, opt)
    modelEntity = MLP("mlp_"+str(net_instnace), model_layer, imgHeight, imgWidth, numChannels, batch_size, numClasses, opt)
    modelLogit = modelEntity.build(features)
    trainOps = modelEntity.train(modelLogit, labels)
    evalOps = modelEntity.evaluate(modelLogit, labels)

    config = tf.ConfigProto()
    config.allow_soft_placement = True    
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        num_batch = Y_data.shape[0] // batch_size
        for e in range(iterations):
            for i in range(num_batch):
                print('epoch %d / %d, step %d / %d' %(e+1, iterations, i+1, num_batch))
                batch_offset = i * batch_size
                batch_end = (i+1) * batch_size
                X_mini_batch_feed = X_data[batch_offset:batch_end,:,:,:]
                Y_mini_batch_feed = Y_data[batch_offset:batch_end,:]
                sess.run(trainOps, feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})
        
        acc_arg = evalOps.eval({features: X_data_eval, labels: Y_data_eval})
        conn.send(acc_arg)
        conn.close()
        print("Accuracy:", acc_arg)

def evaluate_diff_batch():
    numChannels = 1
    numClasses = 10
    imgWidth = 28
    imgHeight = 28

    features = tf.placeholder(tf.float32, [None, imgWidth, imgHeight, numChannels])
    labels = tf.placeholder(tf.int64, [None, numClasses])

    X_data = load_mnist_image(mnist_train_img_path)
    Y_data = load_mnist_label_onehot(mnist_train_label_path)
    X_data_eval = load_mnist_image(mnist_t10k_img_path)
    Y_data_eval = load_mnist_label_onehot(mnist_t10k_label_path)

    net_instnace = 1
    batch_size = 50
    opt = 'Adam'
    epochs = 20

    modelEntity1 = MLP("mlp_"+str(net_instnace), 0, imgHeight, imgWidth, numChannels, batch_size, numClasses, opt, epochs)
    modelLogit1 = modelEntity1.build(features)
    trainOps1 = modelEntity1.train(modelLogit1, labels)
    evalOps1 = modelEntity1.evaluate(modelLogit1, labels)

    net_instnace = 1
    batch_size = 40
    opt = 'SGD'
    epochs = 25

    modelEntity2 = MLP("mlp_"+str(net_instnace), 0, imgHeight, imgWidth, numChannels, batch_size, numClasses, opt, epochs)
    modelLogit2 = modelEntity2.build(features)
    trainOps2 = modelEntity2.train(modelLogit2, labels)
    evalOps2 = modelEntity2.evaluate(modelLogit2, labels)
    
    trainOps = [trainOps1,trainOps2]

    iterations = 10

    config = tf.ConfigProto()
    config.allow_soft_placement = True

    batch_size_one = 40

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        num_batch = Y_data.shape[0] // batch_size_one
        for e in range(iterations):
            for i in range(num_batch):
                print('epoch %d / %d, step %d / %d' %(e+1, iterations, i+1, num_batch))
                batch_offset = i * batch_size_one
                batch_end = (i+1) * batch_size_one
                X_mini_batch_feed = X_data[batch_offset:batch_end,:,:,:]
                Y_mini_batch_feed = Y_data[batch_offset:batch_end,:]
                sess.run(trainOps, feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})
                

def evaluate_model():
    numChannels = 1
    numClasses = 10
    imgWidth = 28
    imgHeight = 28

    features = tf.placeholder(tf.float32, [None, imgWidth, imgHeight, numChannels])
    labels = tf.placeholder(tf.int64, [None, numClasses])

    #X_data = load_bin_raw(bin_dir, numChannels, imgWidth, imgHeight)
    #Y_data = load_labels_onehot(label_path, numClasses)
    #X_data_eval = X_data[0:data_eval_slice,:,:,:]
    #Y_data_eval = Y_data[0:data_eval_slice,:]
    
    X_data = load_mnist_image(mnist_train_img_path)
    Y_data = load_mnist_label_onehot(mnist_train_label_path)
    X_data_eval = load_mnist_image(mnist_t10k_img_path)
    Y_data_eval = load_mnist_label_onehot(mnist_t10k_label_path)
    
    #plt.imshow(X_data_eval[99,:,:,0], cmap='gray')
    #print(Y_data_eval[99,9])
    #plt.show()
    
    dt = datetime.now()
    np.random.seed(dt.microsecond)
    net_instnace = np.random.randint(sys.maxsize)

    batch_size = 40
    opt = 'Adam'

    #modelEntity = MobileNet("mobilenet_"+str(net_instnace), 1, imgHeight, imgWidth, batch_size, numClasses, opt)
    modelEntity = MLP("mlp_"+str(net_instnace), 0, imgHeight, imgWidth, numChannels, batch_size, numClasses, opt)
    modelLogit = modelEntity.build(features)
    trainOps = modelEntity.train(modelLogit, labels)
    evalOps = modelEntity.evaluate(modelLogit, labels)
    
    iterations = 30

    config = tf.ConfigProto()
    config.allow_soft_placement = True

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        num_batch = Y_data.shape[0] // batch_size
        for e in range(iterations):
            for i in range(num_batch):
                print('epoch %d / %d, step %d / %d' %(e+1, iterations, i+1, num_batch))
                batch_offset = i * batch_size
                batch_end = (i+1) * batch_size
                X_mini_batch_feed = X_data[batch_offset:batch_end,:,:,:]
                Y_mini_batch_feed = Y_data[batch_offset:batch_end,:]
                sess.run(trainOps, feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})
        
        #acc_arg = sess.run(evalOps, feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})
        acc_arg = evalOps.eval({features: X_data_eval, labels: Y_data_eval})
        print('accuracy:',acc_arg)
