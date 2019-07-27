import tensorflow as tf
#from tensorflow.python.profiler import model_analyzer
#from tensorflow.python.profiler import option_builder
from tensorflow.python.client import timeline
import numpy as np
import argparse
from timeit import default_timer as timer

from img_utils import *
from dnn_model import DnnModel


#########################
# Command Parameters
#########################

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gpuid', type=int, default=0, help='identify a GPU to run')
parser.add_argument('-bs', '--batchsize', type=int, default=10, help='identify the training batch size')
parser.add_argument('-iw', '--imgw', type=int, default=224, help='identify the weight of img')
parser.add_argument('-ih', '--imgh', type=int, default=224, help='identify the height of img')
parser.add_argument('-cls', '--clazz', type=int, default=1000, help='predication classes')
parser.add_argument('-ch', '--channel', type=int, default=3, help='identify the channel of input images')
parser.add_argument('-e', '--epoch', type=int, default=1, help='identify the epoch numbers')
parser.add_argument('-s', '--shuffle', action='store_true', default=False, help='use shuffle the batch input or not, default is sequential, if use different batch size, then this config will be ignored')
parser.add_argument('-d', '--diff', action='store_true', default=False, help='use different batch input for each model in the packed api, default is all the models in packed use all input batch')
parser.add_argument('-p', '--profile', action='store_true', default=False, help='use tf.profiler or not')
args = parser.parse_args()


#########################
# Global Variables
#########################

#image_dir = '/home/ruiliu/Development/mtml-tf/dataset/imagenet10k'
image_dir = '/tank/local/ruiliu/dataset/imagenet10k'
#bin_dir = '/home/ruiliu/Development/mtml-tf/dataset/imagenet1k.bin'
bin_dir = '/tank/local/ruiliu/dataset/imagenet10k.bin'
#label_path = '/home/ruiliu/Development/mtml-tf/dataset/imagenet1k-label.txt'
label_path = '/tank/local/ruiliu/dataset/imagenet10k-label.txt'
#profile_dir = '/home/ruiliu/Development/mtml-tf/mt-perf/profile_dir'
profile_dir = '/tank/local/ruiliu/mtml-tf/mt-perf/profile_dir'

gpuId = args.gpuid
batchSize = args.batchsize
imgWidth = args.imgw
imgHeight = args.imgh
numClasses = args.clazz
numChannels = args.channel
numEpochs = args.epoch
isShuffle = args.shuffle
isDiffernetBatch = args.diff
isProfile = args.profile

input_model_num = 4

if isDiffernetBatch:
    names = locals()
    input_dict = {}
    for idx in range(input_model_num):
        names['features' + str(idx)] = tf.placeholder(tf.float32, [batchSize, imgWidth, imgHeight, numChannels])
        names['labels' + str(idx)] = tf.placeholder(tf.int64, [batchSize, numClasses])
else:
    features = tf.placeholder(tf.float32, [batchSize, imgWidth, imgHeight, numChannels])
    labels = tf.placeholder(tf.int64, [batchSize, numClasses])

def prepareModelsMan():
    modelCollection = []
    model_class_num = [input_model_num]
    model_class = ["mobilenet_padding"]
    all_batch_list = [40,40,40,50]
    #all_batch_list = np.random.choice([10,20,40,50], input_model_num, replace=False).tolist()
    #all_batch_list = np.repeat(batchSize, input_model_num).tolist()
    layer_list = np.repeat(1, input_model_num).tolist()
    model_name_abbr = np.random.choice(100000, sum(model_class_num), replace=False).tolist()
    for idx, mls in enumerate(model_class):
        for _ in range(model_class_num[idx]):
            dm = DnnModel(mls, str(model_name_abbr.pop()), model_layer=layer_list.pop(), input_w=imgWidth, input_h=imgHeight, num_classes=numClasses, batch_size=all_batch_list.pop(), desired_accuracy=0.9)
            modelCollection.append(dm)
    return modelCollection

def printAllModels(model_collection):
    for idm in model_collection:
        print(idm.getInstanceName())
        print(idm.getModelName())
        print(idm.getModelLayer())
        print(idm.getBatchSize())
        print("===============")

def buildPackedModels(model_collection):
    modelEntityCollection = []
    trainCollection = []
    for midx, mc in enumerate(modelCollection):
        modelEntity = mc.getModelEntity()
        modelEntityCollection.append(modelEntity)
        if isDiffernetBatch:
            modelLogit = modelEntity.build(names['features' + str(midx)])
            modelTrain = modelEntity.cost(modelLogit, names['labels' + str(midx)])
        else:
            modelLogit = modelEntity.build(features)
            modelTrain = modelEntity.cost(modelLogit, labels)
        trainCollection.append(modelTrain)
    return trainCollection

def buildPackedModelsCombine():
    combineTrain = 0
    for midx, mc in enumerate(modelCollection):
        modelEntity = mc.getModelEntity()
        modelLogit = modelEntity.build(features)
        modelTrain = modelEntity.getCost(modelLogit, labels)
        combineTrain += modelTrain

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(combineTrain)

    return train_step

def execPack(train_collection, num_epoch, X_train, Y_train):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    with tf.Session(config=config) as sess:
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        sess.run(tf.global_variables_initializer())
        num_batch = Y_train.shape[0] // batchSize
        num_batch_list = np.arange(num_batch)
        for e in range(num_epoch):
            for i in range(num_batch):
                print('epoch %d / %d, step %d / %d' %(e+1, num_epoch, i+1, num_batch))
                batch_offset = i * batchSize
                batch_end = (i+1) * batchSize
                X_mini_batch_feed = X_train[batch_offset:batch_end,:,:,:]
                Y_mini_batch_feed = Y_train[batch_offset:batch_end,:]
                if isDiffernetBatch:
                    for ridx in range(input_model_num):
                        rand_idx = int(np.random.choice(num_batch_list, 1))
                        names['X_mini_batch_feed' + str(ridx)] = X_train[rand_idx:rand_idx+batchSize,:,:,:]
                        names['Y_mini_batch_feed' + str(ridx)] = Y_train[rand_idx:rand_idx+batchSize,:]
                        input_dict[names['features' + str(ridx)]] = names['X_mini_batch_feed' + str(ridx)]
                        input_dict[names['labels' + str(ridx)]] = names['Y_mini_batch_feed' + str(ridx)]
                    if isProfile:
                        sess.run(train_collection, feed_dict=input_dict, options=run_options, run_metadata=run_metadata)
                        trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                        trace_file = open('/tank/local/ruiliu/mtml-tf/mt-padding/profile_dir/tf_packed_b20b40_'+str(i)+'.json', 'w')
                        #trace_file = open('/home/ruiliu/Development/mtml-tf/mt-perf/profile_dir/test/tf_packed_'+str(i)+'.json', 'w')
                        trace_file.write(trace.generate_chrome_trace_format(show_memory=True))
                    else:
                        sess.run(train_collection, feed_dict=input_dict)
                else:
                    if isShuffle:
                        rand_idx = int(np.random.choice(num_batch_list, 1, replace=False))
                        X_mini_batch_feed = X_train[rand_idx:rand_idx+batchSize,:,:,:]
                        Y_mini_batch_feed = Y_train[rand_idx:rand_idx+batchSize,:]
                    else:
                        X_mini_batch_feed = X_train[i:i+batchSize,:,:,:]
                        Y_mini_batch_feed = Y_train[i:i+batchSize,:]
                    if isProfile:
                        sess.run(train_collection, feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed}, options=run_options, run_metadata=run_metadata)
                        trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                        trace_file = open('/tank/local/ruiliu/mtml-tf/mt-padding/profile_dir/tf_packed_b20b40_'+str(i)+'.json', 'w')
                        #trace_file = open('/home/ruiliu/Development/mtml-tf/mt-perf/profile_dir/test/tf_packed_'+str(i)+'.json', 'w')
                        trace_file.write(trace.generate_chrome_trace_format(show_memory=True))
                    else:
                        #start_time = timer()
                        sess.run(train_collection, feed_dict={features:X_mini_batch_feed, labels:Y_mini_batch_feed})
                        #end_time = timer()
                        #dur_time = end_time - start_time
                        #total_time += dur_time
                        #print("training time for 1 epoch:", dur_time) 
                    
def testOps():

    ######################
    #test tf ops 
    ######################

    image_raw = tf.placeholder(tf.int64,shape=[500, 375, 3])
    #trans_op = tf.image.resize_images(image_raw, (224, 224))
    trans_op = tf.contrib.image.transform(image_raw,transforms=[1,0,0,0,1,0,0,0])
    img = cv2.imread(image_dir+'/'+image_name)
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    with tf.Session(config=config) as sess:
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        sess.run(trans_op, feed_dict={image_raw:img}, options=run_options, run_metadata=run_metadata)
        trace = timeline.Timeline(step_stats=run_metadata.step_stats)
        trace_file = open(profile_dir+'/tf_transform.json', 'w')
        trace_file.write(trace.generate_chrome_trace_format(show_memory=True))

    ######################
    #test tf.data api 
    ######################

    #with tf.Session(config=config) as sess:
    #    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    #    run_metadata = tf.RunMetadata()
    #    sess.run(tf.global_variables_initializer())
    #    my_profiler = model_analyzer.Profiler(graph=sess.graph)

    #    dataset_it = generate_image_dataset(image_dir,label_path)
    #    next_data = dataset_it.get_next()
    
    #    for i in range(10):
    #        sess.run(next_data)
    #        image, label = sess.run(next_data, options=run_options, run_metadata=run_metadata)
 
    #        trace = timeline.Timeline(step_stats=run_metadata.step_stats)
    #        trace_file = open('/home/ruiliu/Development/mtml-tf/mt-perf/profile_dir/test/sss.json', 'w')
    #        trace_file.write(trace.generate_chrome_trace_format(show_memory=True))
    #        my_profiler.add_step(step=i, run_meta=run_metadata)

    #profile_graph_builder = option_builder.ProfileOptionBuilder(option_builder.ProfileOptionBuilder.time_and_memory())
    #profile_graph_builder.with_timeline_output(timeline_file='/home/ruiliu/Development/mtml-tf/mt-perf/profile_dir/test/timeline.json')
    #profile_graph_builder.with_step([0,9])
    #my_profiler.profile_graph(profile_graph_builder.build())

if __name__ == '__main__':
    X_data = load_images_bin(bin_dir, numChannels, imgWidth, imgHeight)
    Y_data = load_labels_onehot(label_path, numClasses)
    modelCollection = prepareModelsMan()
    printAllModels(modelCollection)
    trainCollection = buildPackedModels(modelCollection)
    execPack(trainCollection, numEpochs, X_data, Y_data)
