import tensorflow as tf
from schedule import Schedule
from dnn_model import DnnModel
from img_utils import *
from cifar_utils import *
import numpy as np


img_w = 224
img_h = 224
num_classes = 1000

def main(_):
    #load image size:224x224, class:1000
    #data_dir = '/home/ruiliu/Development/mtml-tf/dataset/test'
    #label_path = '/home/ruiliu/Development/mtml-tf/dataset/test-label.txt'
    #X_data = load_images(data_dir)
    #Y_data = load_labels_onehot(label_path)

    #load image cifar-10, image size:32x32, class:10
    #cifar10_dir = '/home/ruiliu/Development/mtml-tf/dataset/cifar-10'
    #cifar = cifar_utils(cifar10_dir)
    #X_data, _, Y_data = cifar.load_evaluation_data()
    
    #model_class_num = random.sample(xrange(10), 4)
    model_class_num = [3, 5, 7, 10]
    model_class = ["resnet", "mobilenet", "convnet", "perceptron"]
    
    all_batch_list = [10, 20, 40, 50, 80, 100]

    model_collection = []

    for idx, mls in enumerate(model_class):
        for i in range(model_class_num[idx]):
            batch_num = np.random.randint(1, len(all_batch_list))
            
            batch_set = np.random.choice(all_batch_list, size=batch_num, replace=False)  
            
            layer_num = np.random.randint(1, 10)
            
            dm = DnnModel(mls, model_layer=layer_num, input_w=img_w, input_h=img_h,  
                          num_classes=num_classes, batch_size_range=batch_set, desired_accuracy=0.9)
            model_collection.append(dm)

    print(model_collection)

    #mp = DnnModel("perceptron", 2, {10,20}, 0.9)
    #mp = DnnModel("convnet", 3, {10,20}, 0.9)
    #mp = DnnModel("resnet", 3, {10,20}, 0.9)
    #mp = DnnModel(model_name="resnet", model_layer=2, input_w=img_w, input_h=img_h, 
    #              num_classes=num_classes, batch_size_range={10,20}, desired_accuracy=0.9)

    #resnetModel = DnnModel("resnet",  50, {10,20}, 0.9)
    #mobilenetModel1 = DnnModel("mobilenet", 40, {20,40}, 0.8)
    #mobilenetModel2 = DnnModel("mobilenet", 40, {20,40}, 0.8)

    #model_list = []
    #model_list.append(resnetModel)
    #model_list.append(mobilenetModel1)
    #model_list.append(mobilenetModel2)
    #sch = Schedule(model_list, img_w, img_h, num_classes)
    #sch.showAllModelInstances()
    #sch.schedule()
    #sch.executeSch(X_data, Y_data)
    #sch.testSingleModel(mp, X_data, Y_data)
    
if __name__ == '__main__':
    tf.app.run(main=main)
