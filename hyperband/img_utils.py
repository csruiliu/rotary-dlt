import numpy as np
import pickle
import os
import cv2
from matplotlib import pyplot as plt

imgWidth = 224
imgHeight = 224
numChannels = 3
numClasses = 1000

# image format: [batch, height, width, channels]

def convert_image_bin(imgDir, img_h, img_w):
    all_arr = []
    for filename in sorted(os.listdir(imgDir)):
        print(filename)
        arr_single = read_single_image(imgDir + '/'+ filename, img_h, img_w)
        if all_arr == []:
            all_arr = arr_single
        else:
            all_arr = np.concatenate((all_arr, arr_single))    
    return all_arr

# use pure write to store
def save_bin_raw(arr, output_file):
    with open(output_file, 'wb+') as of:
        of.write(arr)

def load_bin_raw(img_bin):
    with open(img_bin, mode='rb') as file:
        raw_images = file.read()
        buffer_images = np.frombuffer(raw_images, dtype=np.uint8)
        float_images = np.array(buffer_images, dtype=float) / 255.0
        img_num = int(float_images.size / imgWidth / imgHeight / numChannels)
        images = float_images.reshape((img_num, imgWidth, imgHeight, numChannels))
    return images

# use pickle package to store, but it is slow
def save_bin_pickle(arr, output_file):
    img_data = {'image': arr}
    f = open(output_file, 'wb+')
    pickle.dump(img_data, f, pickle.HIGHEST_PROTOCOL)
    f.close()

def load_bin_pickle(path, num_channels, img_w, img_h):
    with open(path, mode='rb') as file:
        data = pickle.load(file, encoding='bytes')
    raw_images = data['image']
    raw_float = np.array(raw_images, dtype=float) / 255.0
    #raw_float = raw_float.transpose([0, 3, 1, 2])
    return raw_float

def read_single_image(img_name, img_h, img_w):
    img = plt.imread(img_name)
    img = cv2.resize(img, (img_h, img_w))
    img_ext = np.expand_dims(img, axis=0)
    return img_ext

if __name__ == "__main__":

    imgDir = '/home/ruiliu/Development/mtml-tf/dataset/imagenet10k'
    outDir = '/home/ruiliu/Development/mtml-tf/dataset/imagenet10k.bin'

    #bin_dir = '/tank/local/ruiliu/dataset/imagenet10k.bin'
    #bin_dir = '/home/ruiliu/Development/mtml-tf/dataset/imagenet1k.bin'
    #label_path = '/tank/local/ruiliu/dataset/imagenet10k-label.txt'
    #label_path = '/home/ruiliu/Development/mtml-tf/dataset/imagenet1k-label.txt'
    
    arr_image = convert_image_bin(imgDir, imgWidth, imgHeight)
    save_bin_raw(arr_image, outDir)
    #images = load_bin_raw('/home/ruiliu/Development/mtml-tf/dataset/imagenet1k.bin')
    
    #plt.imshow(images[4,:,:,:])
    #plt.show()