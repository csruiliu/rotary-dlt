import tensorflow as tf
import numpy as np
import itertools
from operator import itemgetter
from datetime import datetime
import sys
import random as rd
from multiprocessing import Process, Pipe
from timeit import default_timer as timer

from mlp import MLP
from img_utils import * 

imgWidth = 28
imgHeight = 28
numChannels = 1
numClasses = 10

mnist_train_img_path = '/home/ruiliu/Development/mtml-tf/dataset/mnist-train-images.idx3-ubyte'
#mnist_train_img_path = '/tank/local/ruiliu/dataset/mnist-train-images.idx3-ubyte'
mnist_train_label_path = '/home/ruiliu/Development/mtml-tf/dataset/mnist-train-labels.idx1-ubyte'
#mnist_train_label_path = '/tank/local/ruiliu/dataset/mnist-train-labels.idx1-ubyte'
mnist_t10k_img_path = '/home/ruiliu/Development/mtml-tf/dataset/mnist-t10k-images.idx3-ubyte'
#mnist_t10k_img_path = '/tank/local/ruiliu/dataset/mnist-t10k-images.idx3-ubyte'
mnist_t10k_label_path = '/home/ruiliu/Development/mtml-tf/dataset/mnist-t10k-labels.idx1-ubyte'
#mnist_t10k_label_path = '/tank/local/ruiliu/dataset/mnist-t10k-labels.idx1-ubyte'

def gen_confs(n_conf):
    batch_size = np.arange(10,61,5)
    opt_conf = ['Adam','SGD','Adagrad','Momentum']
    model_layer = np.arange(0,6,1)
    all_conf = [batch_size, opt_conf, model_layer]

    hp_conf = list(itertools.product(*all_conf))
    np.random.seed(100)
    idx_list = np.random.choice(np.arange(0, len(hp_conf)), n_conf, replace=False)
    rand_conf_list = list(itemgetter(*idx_list)(hp_conf))
    return rand_conf_list

def eval_trial_pair(conf_a, conf_b, conn):
    features = tf.placeholder(tf.float32, [None, imgWidth, imgHeight, numChannels])
    labels = tf.placeholder(tf.int64, [None, numClasses])

    seed = np.random.randint(100000, size=1)

    X_data = load_mnist_image(mnist_train_img_path, seed)
    Y_data = load_mnist_label_onehot(mnist_train_label_path, seed)

    dt = datetime.now()
    np.random.seed(dt.microsecond)    
    net_instnace = np.random.randint(sys.maxsize, size=2)

    modelEntity_a = MLP("mlp_"+str(net_instnace[0]), conf_a[2], imgHeight, imgWidth, numChannels, conf_a[0], numClasses, conf_a[1])
    modelLogit_a = modelEntity_a.build(features)
    trainOps_a = modelEntity_a.train(modelLogit_a, labels)
    
    modelEntity_b = MLP("mlp_"+str(net_instnace[1]), conf_b[2], imgHeight, imgWidth, numChannels, conf_b[0], numClasses, conf_b[1])    
    modelLogit_b = modelEntity_b.build(features)
    trainOps_b = modelEntity_b.train(modelLogit_b, labels)

    train_pair = [trainOps_a,trainOps_b]

    pack_dur_time = -1
    seq_dur_time = -1

    config = tf.ConfigProto()
    config.allow_soft_placement = True   
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        
        batch_size_a = conf_a[0]
        X_mini_batch_feed_a = X_data[0:batch_size_a,:,:,:]
        Y_mini_batch_feed_a = Y_data[0:batch_size_a,:]
        a_start_time = timer()
        sess.run(trainOps_a, feed_dict={features: X_mini_batch_feed_a, labels: Y_mini_batch_feed_a})
        a_end_time = timer()
        a_dur_time = a_end_time - a_start_time
        
        batch_size_b = conf_b[0]
        X_mini_batch_feed_b = X_data[0:batch_size_b,:,:,:]
        Y_mini_batch_feed_b = Y_data[0:batch_size_b,:]
        b_start_time = timer()
        sess.run(trainOps_b, feed_dict={features: X_mini_batch_feed_b, labels: Y_mini_batch_feed_b})
        b_end_time = timer()
        b_dur_time = b_end_time - b_start_time
        
        seq_dur_time = b_dur_time + a_dur_time

        batch_pack_size = max(conf_a[0],conf_b[0])
        X_mini_batch_feed = X_data[0:batch_pack_size,:,:,:]
        Y_mini_batch_feed = Y_data[0:batch_pack_size,:]
        pack_start_time = timer()
        sess.run(train_pair, feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})
        pack_end_time = timer()
        pack_dur_time = pack_end_time - pack_start_time
        
    result = seq_dur_time - pack_dur_time

    conn.send(result)
    conn.close()

def prep_trial(conf_list):
    trial_dict = dict()
    for cidx in conf_list:
        trial_dict[cidx] = list()
        for didx in conf_list:
            if cidx != didx:
                trial_dict[cidx].append(didx)
    return trial_dict

def run_trial(trial_dict):
    trial_result_dict = dict()
    for key, value in trial_dict.items():
        trial_result_dict[key] = dict()
        for vidx in value:
            #print("run two models:",key, vidx)
            parent_conn, child_conn = Pipe()
            p = Process(target=eval_trial_pair, args=(key, vidx, child_conn))
            p.start()
            result = parent_conn.recv()
            parent_conn.close()
            p.join()

            #result = float(np.random.uniform(0,100,1))
            trial_result_dict[key][vidx] = result
    
    return trial_result_dict

def pack_trial_standalone(confs, t_dict, tr_dict):
    #print("confs:",confs)
    select_num = 3

    trial_dict = t_dict
    trial_result_dict = tr_dict
    confs_list = confs

    while len(confs_list) > 1:
        trial_packed_list = []
        spoint = rd.choice(confs_list)
        #print("spoint:",spoint)
        trial_packed_list.append(spoint)
        
        spoint_list = trial_result_dict.get(spoint) 
        ssl = sorted(spoint_list.items(), key=lambda kv: kv[1], reverse=True)
        
        if select_num <= len(ssl):
            for sidx in range(select_num):
                selected_conf = ssl[sidx][0]
                #print("selected_conf:",selected_conf)
                trial_packed_list.append(selected_conf)
                confs_list.remove(selected_conf)

                trial_dict.pop(selected_conf)
                trial_result_dict.pop(selected_conf)

                for didx in trial_dict:
                    trial_dict.get(didx).remove(selected_conf)
                for ridx in trial_result_dict:
                    trial_result_dict.get(ridx).pop(selected_conf)

        else:
            for sidx in ssl:
                selected_conf = sidx[0]
                #print("selected_conf:",selected_conf)
                trial_packed_list.append(selected_conf)
                confs_list.remove(selected_conf)
                trial_dict.pop(selected_conf)
                trial_result_dict.pop(selected_conf)

                for didx in trial_dict:
                    trial_dict.get(didx).remove(selected_conf)
                for ridx in trial_result_dict:
                    trial_result_dict.get(ridx).pop(selected_conf)

        confs_list.remove(spoint)
        trial_dict.pop(spoint)
        trial_result_dict.pop(spoint)
        for didx in trial_dict:
            trial_dict.get(didx).remove(spoint)
        for ridx in trial_result_dict:
            trial_result_dict.get(ridx).pop(spoint)

        print(trial_packed_list)

def pack_trial(confs, topk):
    confs_list = list(confs)
    trial_dict = prep_trial(confs_list)
    trial_result_dict = run_trial(trial_dict)
    
    trial_pack_collection = []

    while len(confs_list) > 1:
        trial_packed_list = []
        spoint = rd.choice(confs_list)
        #print("spoint:",spoint)
        trial_packed_list.append(spoint)
        
        spoint_list = trial_result_dict.get(spoint) 
        ssl = sorted(spoint_list.items(), key=lambda kv: kv[1], reverse=True)
        
        if topk <= len(ssl):
            for sidx in range(topk):
                selected_conf = ssl[sidx][0]
                #print("selected_conf:",selected_conf)
                trial_packed_list.append(selected_conf)
                confs_list.remove(selected_conf)

                trial_dict.pop(selected_conf)
                trial_result_dict.pop(selected_conf)

                for didx in trial_dict:
                    trial_dict.get(didx).remove(selected_conf)
                for ridx in trial_result_dict:
                    trial_result_dict.get(ridx).pop(selected_conf)

        else:
            for sidx in ssl:
                selected_conf = sidx[0]
                #print("selected_conf:",selected_conf)
                trial_packed_list.append(selected_conf)
                confs_list.remove(selected_conf)
                trial_dict.pop(selected_conf)
                trial_result_dict.pop(selected_conf)

                for didx in trial_dict:
                    trial_dict.get(didx).remove(selected_conf)
                for ridx in trial_result_dict:
                    trial_result_dict.get(ridx).pop(selected_conf)

        confs_list.remove(spoint)
        trial_dict.pop(spoint)
        trial_result_dict.pop(spoint)
        for didx in trial_dict:
            trial_dict.get(didx).remove(spoint)
        for ridx in trial_result_dict:
            trial_result_dict.get(ridx).pop(spoint)

        trial_pack_collection.append(trial_packed_list)

    return trial_pack_collection


if __name__ == "__main__":
    confs_num = 6
    confs_list = gen_confs(confs_num)
    trial_dict = prep_trial(confs_list)
    trial_result_dict = run_trial(trial_dict)
    pack_trial_standalone(confs_list, trial_dict, trial_result_dict)

