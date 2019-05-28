import tensorflow as tf
from schedule import Schedule 
from dnn_model import DnnModel
from img_utils import *



def main(_):
    data_dir = '/home/ruiliu/Development/mtml-tf/dataset/test'
    label_path = '/home/ruiliu/Development/mtml-tf/dataset/test-label.txt'
    X_data = load_images(data_dir)
    Y_data = load_labels_onehot(label_path)

    resnetModel = DnnModel("resnet",  50, {10,20}, 0.9)
    mobilenetModel1 = DnnModel("mobilenet", 40, {20,40}, 0.8)
    mobilenetModel2 = DnnModel("mobilenet", 40, {20,40}, 0.8)
    
    model_list = []
    model_list.append(resnetModel)
    model_list.append(mobilenetModel1)
    model_list.append(mobilenetModel2)
    sch = Schedule(model_list)
    sch.showAllModels()

    sch.pack()
    sch.executeSch(X_data, Y_data)
    
    
if __name__ == '__main__':
    tf.app.run(main=main)
