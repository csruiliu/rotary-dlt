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
    np.random.seed(0)
    #print(len(hp_conf))
    idx_list = np.random.choice(np.arange(0, len(hp_conf)), n_conf, replace=False)
    rand_conf = itemgetter(*idx_list)(hp_conf)

    return rand_conf

def run_params_pack_random(confs, epochs, conn):
    print("running configurations:", confs)
    print("epochs:",epochs)
    
    features = tf.placeholder(tf.float32, [None, imgWidth, imgHeight, numChannels])
    labels = tf.placeholder(tf.int64, [None, numClasses])
    X_data = load_mnist_image(mnist_train_img_path)
    Y_data = load_mnist_label_onehot(mnist_train_label_path)
    X_data_eval = load_mnist_image(mnist_t10k_img_path)
    Y_data_eval = load_mnist_label_onehot(mnist_t10k_label_path)

    dt = datetime.now()
    np.random.seed(dt.microsecond)    
    net_instnace = np.random.randint(sys.maxsize, size=len(confs))
    
    desire_epochs = epochs
    
    entity_pack = []
    train_pack = []
    eval_pack = [] 
    acc_pack = []
    batch_size_set = set()

    max_bs = np.NINF

    for cidx, cf in enumerate(confs):
        batch_size = cf[0]
        batch_size_set.add(batch_size)
        opt = cf[1]
        model_layer = cf[2]
        
        desire_steps = Y_data.shape[0] // batch_size
        modelEntity = MLP("mlp_"+str(net_instnace[cidx]), model_layer, imgHeight, imgWidth, numChannels, batch_size, numClasses, opt)
        modelEntity.setDesireEpochs(desire_epochs)
        modelEntity.setDesireSteps(desire_steps)
        modelLogit = modelEntity.build(features)
        trainOps = modelEntity.train(modelLogit, labels)
        evalOps = modelEntity.evaluate(modelLogit, labels)
        entity_pack.append(modelEntity)
        train_pack.append(trainOps)
        eval_pack.append(evalOps)

    config = tf.ConfigProto()
    config.allow_soft_placement = True   
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        max_bs = max(batch_size_set)
        
        complete_flag = False

        while len(train_pack) != 0:
            print("current max batch size:",max_bs)
            num_steps = Y_data.shape[0] // max_bs
            for i in range(num_steps):
                #print("step {}/{}".format(i+1, num_steps))
                batch_offset = i * max_bs
                batch_end = (i+1) * max_bs
                X_mini_batch_feed = X_data[batch_offset:batch_end,:,:,:]
                Y_mini_batch_feed = Y_data[batch_offset:batch_end,:]
                sess.run(train_pack, feed_dict = {features: X_mini_batch_feed, labels: Y_mini_batch_feed})
                for me in entity_pack:
                    me.setCurStep()
                    if me.isCompleteTrain():
                        print("model has been trained completely:",me.getModelInstance())
                        sess.run(me.setBatchSize(Y_data_eval.shape[0]))
                        train_pack.remove(me.getTrainOp())
                        complete_flag = True   
                
                if len(train_pack) == 0:
                    break
                
                if complete_flag:
                    batch_size_set.discard(max_bs)
                    max_bs = max(batch_size_set)
                    complete_flag = False
                    break
    
        print("models have been training this run, start to evaluate")
        for ep in eval_pack:
            #num_steps = Y_data.shape[0] // max_bs
            acc_arg = ep.eval({features: X_data_eval, labels: Y_data_eval})
            #acc_arg = sess.run(ep, feed_dict = {features: X_mini_batch_feed, labels: Y_mini_batch_feed})
            acc_pack.append(acc_arg)
            print(acc_arg)
        
    conn.send(acc_pack)
    conn.close()
    print("Accuracy:", acc_pack)
        
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
        
        modelEntity = MLP("mlp_"+str(net_instnace), model_layer, imgHeight, imgWidth, numChannels, batch_size, numClasses, opt)
        modelLogit = modelEntity.build(features)
        trainOps = modelEntity.train(modelLogit, labels)
        evalOps = modelEntity.evaluate(modelLogit, labels)
        setbsOps= modelEntity.setBatchSize(Y_data_eval.shape[0]) 
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
            
            
            sess.run(setbsOps)
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
        
        setbs_pack = []
        train_pack = []
        eval_pack = [] 
        acc_pack = []

        for idx, _ in enumerate(confs):
            if (idx+1) % 2 == 0:
                modelEntity = MLP("mlp_"+str(net_instnace[(idx-1)//2]), confs[idx], imgHeight, imgWidth, numChannels, batch_size, numClasses, confs[idx-1])
                modelLogit = modelEntity.build(features)
                trainOps = modelEntity.train(modelLogit, labels)
                evalOps = modelEntity.evaluate(modelLogit, labels)
                setbs_pack.append(modelEntity.setBatchSize(Y_data_eval.shape[0]))
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
            
            
            sess.run(setbs_pack)

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
        

        sess.run(modelEntity.setBatchSize(Y_data_eval.shape[0]))
        acc_arg = evalOps.eval({features: X_data_eval, labels: Y_data_eval})
        conn.send(acc_arg)
        conn.close()
        print("Accuracy:", acc_arg)

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
