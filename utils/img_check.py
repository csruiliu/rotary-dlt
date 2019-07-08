import numpy as np
import matplotlib.pyplot as plt

def check_channel_dir(img_dir):
    for img_name in os.listdir(img_dir):
        img = plt.imread(img_dir+'/'+img_name)
        shape = img.shape
        if len(shape) != 3:
            print(img_name)

if __name__ == '__main__':
    img_dir = '/home/ruiliu/Development/mtml-tf/dataset/imagenet150k'
    check_channel_dir(img_dir)