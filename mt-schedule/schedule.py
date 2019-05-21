import tensorflow as tf
from resnet import *
from mobilenet import *
from img_utils import *

img_w = 224
img_h = 224

class Schedule(object):
    def __init__(self, model_collection):
        self.model_set = model_collection

def main(_):
    data_dir = '/home/ruiliu/Development/mtml-tf/dataset/test'
    label_path = '/home/ruiliu/Development/mtml-tf/dataset/test-label'
    #X_data = load_images(data_dir)
    #Y_data = load_labels_onehot(label_path)
    

if __name__ == '__main__':
    tf.app.run(main=main)
