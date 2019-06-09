import tensorflow as tf
from dnn_model import DnnModel
from img_utils import *
from cifar_utils import *
import numpy as np
from datetime import datetime
from multiprocessing import Process
from timeit import default_timer as timer
import os

img_w = 224
img_h = 224
num_channel = 3
num_classes = 1000

modelCollection = []
modelEntityCollection = []
logitCollection = []
scheduleCollection = []

features = tf.placeholder(tf.float32, [None, img_w, img_h, num_channel])
labels = tf.placeholder(tf.int64, [None, num_classes])

data_dir = '/home/ruiliu/Development/mtml-tf/dataset/test'
label_path = '/home/ruiliu/Development/mtml-tf/dataset/test-label.txt'
X_data = load_images(data_dir)
Y_data = load_labels_onehot(label_path)

model_class_num = [3, 2, 1]
model_class_total = sum(model_class_num)
model_name_abbr = np.random.choice(100000, model_class_total, replace=False).tolist()

model_class = ["resnet", "mobilenet", "perceptron"]
all_batch_list = [10, 20, 40, 50, 80, 100]

def prepareModels():
    for idx, mls in enumerate(model_class):
        for _ in range(model_class_num[idx]):
            batch_num = np.random.randint(1, len(all_batch_list))
            batch_set = np.random.choice(all_batch_list, size=batch_num, replace=False)  
            layer_num = np.random.randint(1, 10)
            
            dm = DnnModel(mls, str(model_name_abbr.pop()), model_layer=layer_num, input_w=img_w, input_h=img_h,  
                            num_classes=num_classes, batch_size_range=batch_set, desired_accuracy=0.9)
            modelCollection.append(dm)
    
def buildModels():
    for midx in modelCollection:
        modelEntity = midx.getModelEntity()
        modelEntityCollection.append(modelEntity) 
        modelLogit = modelEntity.build(features)
        logitCollection.append(modelLogit)

    for lidx in modelEntityCollection:
        print(lidx.getModelInstanceName())
        print(lidx.getModelMemSize())

def scheduleNaive():
    for lit in logitCollection:
        scheduleUnit = []
        scheduleUnit.append(lit)
        scheduleCollection.append(lit)

def executeSch(sch_unit, X_train, Y_train):
    total_time = 0
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        mini_batches = 10
        num_batch = Y_train.shape[0] // mini_batches            
        for i in range(num_batch):
            print('step %d / %d' %(i+1, num_batch))
            X_mini_batch_feed = X_train[num_batch:num_batch + mini_batches,:,:,:]
            Y_mini_batch_feed = Y_train[num_batch:num_batch + mini_batches,:]
            start_time = timer()
            sess.run(sch_unit, feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})
            end_time = timer()
            total_time += end_time - start_time
            print("training time for 1 epoch:", total_time)  

if __name__ == '__main__':
    prepareModels()
    buildModels()
    scheduleNaive()
    
    for sit in scheduleCollection:
        p = Process(target=executeSch, args=(sit, X_data, Y_data,))
        p.start()
        print(p.pid)
        p.join()
