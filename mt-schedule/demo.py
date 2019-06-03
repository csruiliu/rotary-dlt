import tensorflow as tf
from schedule import Schedule
from dnn_model import DnnModel
from img_utils import *
from cifar_utils import *

img_w = 224
img_h = 224
num_classes = 1000

def main(_):
    data_dir = '/home/ruiliu/Development/mtml-tf/dataset/test'
    label_path = '/home/ruiliu/Development/mtml-tf/dataset/test-label.txt'

    #load image cifar-10, image size:32x32, class:10
    #cifar10_dir = '/home/ruiliu/Development/mtml-tf/dataset/cifar-10'
    #cifar = cifar_utils(cifar10_dir)
    #X_data, _, Y_data = cifar.load_evaluation_data()
    
    #load image size:224x224, class:1000
    X_data = load_images(data_dir)
    Y_data = load_labels_onehot(label_path)

    #mp = DnnModel("perceptron", 2, {10,20}, 0.9)
    #mp = DnnModel("convnet", 3, {10,20}, 0.9)
    #mp = DnnModel("resnet", 3, {10,20}, 0.9)
    mp = DnnModel(model_name="resnet", model_layer=2, input_w=img_w, input_h=img_h, 
                  num_classes=num_classes, batch_size_range={10,20}, desired_accuracy=0.9)

    #resnetModel = DnnModel("resnet",  50, {10,20}, 0.9)
    #mobilenetModel1 = DnnModel("mobilenet", 40, {20,40}, 0.8)
    #mobilenetModel2 = DnnModel("mobilenet", 40, {20,40}, 0.8)

    model_list = []
    #model_list.append(resnetModel)
    #model_list.append(mobilenetModel1)
    #model_list.append(mobilenetModel2)
    sch = Schedule(model_list, img_w, img_h, num_classes)
    #sch.showAllModelInstances()
    #sch.schedule()
    #sch.executeSch(X_data, Y_data)
    sch.testSingleModel(mp, X_data, Y_data)
    
if __name__ == '__main__':
    tf.app.run(main=main)
