import tensorflow as tf
import numpy as np
import pickle
import os

class cifar_utils(object):
    def __init__(self, dir_name, img_height=32, img_width=32, num_channels=3, num_classes=10, num_files_train=5, images_per_file=10000):
        self.cifar10_data = dir_name
        self.img_h = img_height
        self.img_w = img_width 
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.num_files_train = num_files_train
        self.images_per_file = images_per_file
        self.num_images_train = num_files_train * images_per_file
        
    def unpickle(self, file_name):
        cifar10_data = os.path.join(self.cifar10_data, file_name)
        with open(cifar10_data, mode='rb') as file:
            data = pickle.load(file, encoding='bytes')
        return data

    def one_hot_encoded(self, class_numbers, num_classes=None):
        if num_classes is None:
            num_classes = np.max(class_numbers) + 1

        return np.eye(num_classes, dtype=float)[class_numbers]

    def convert_image(self, raw_images, num_channels, image_width, image_height):
        raw_float = np.array(raw_images, dtype=float) / 255.0
        images = raw_float.reshape([-1, num_channels, self.img_w, self.img_h])
        images = images.transpose([0, 2, 3, 1])
        return images

    def load_class_name(self):
        raw = self.unpickle(file_name="batches.meta")[b'label_names']
        names = [x.decode('utf-8') for x in raw]
        return names

    def load_data(self, filename):
        data = self.unpickle(file_name=filename)
        raw_images = data[b'data']
        labels = np.array(data[b'labels'])
        images = self.convert_image(raw_images, self.num_channels, self.img_w, self.img_h)
        return images, labels

    def load_training_data(self, num_images_train):
        images = np.zeros(shape=[self.num_images_train, self.img_w, self.img_h, self.num_channels], dtype=float)
        labels = np.zeros(shape=[self.num_images_train], dtype=int)
        begin = 0
        for i in range(self.num_files_train):
            images_batch, cls_batch = self.load_data(filename="data_batch_" + str(i + 1))
            num_images = len(images_batch)
            end = begin + num_images
            images[begin:end, :] = images_batch
            labels[begin:end] = cls_batch
            begin = end
        return images, labels, self.one_hot_encoded(class_numbers=labels, num_classes=self.num_classes)

    def load_evaluation_data(self):
        images, labels = self.load_data(filename="test_batch")
        return images, labels, self.one_hot_encoded(class_numbers=labels, num_classes=self.num_classes)