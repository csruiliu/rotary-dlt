import tensorflow as tf
from schedule import Schedule
from dnn_model import DnnModel
from img_utils import *
from cifar_utils import *
import numpy as np
from datetime import datetime
from multiprocessing import Process
import os

img_w = 224
img_h = 224
num_classes = 1000

data_dir = '/home/ruiliu/Development/mtml-tf/dataset/test'
label_path = '/home/ruiliu/Development/mtml-tf/dataset/test-label.txt'
X_data = load_images(data_dir)
Y_data = load_labels_onehot(label_path)

model_class_num = [3, 5, 7, 10]
model_class_total = sum(model_class_num)
model_name_abbr = np.random.choice(100000, model_class_total, replace=False).tolist()

model_class = ["resnet", "mobilenet", "convnet", "perceptron"]
all_batch_list = [10, 20, 40, 50, 80, 100]
model_collection = []

for idx, mls in enumerate(model_class):
    for _ in range(model_class_num[idx]):
        batch_num = np.random.randint(1, len(all_batch_list))
        batch_set = np.random.choice(all_batch_list, size=batch_num, replace=False)  
        layer_num = np.random.randint(1, 10)
        
        dm = DnnModel(mls, str(model_name_abbr.pop()), model_layer=layer_num, input_w=img_w, input_h=img_h,  
                        num_classes=num_classes, batch_size_range=batch_set, desired_accuracy=0.9)
        model_collection.append(dm)
    
sch = Schedule(model_collection, img_w, img_h, num_classes)

def process(schedule):
    schedule.e

if __name__ == '__main__':
    p = Process(target=)
