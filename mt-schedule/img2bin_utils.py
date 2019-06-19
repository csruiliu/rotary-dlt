import pickle
from PIL import Image
import numpy as np
import os
import cv2
import matplotlib.image as plimg


def image_input(imgDir, img_w, img_h):
    all_arr = []
    for filename in os.listdir(imgDir):    
        arr_single = read_single_image(imgDir + '/'+ filename, img_w, img_h)
        if arr_single != []:    
            if all_arr == []:
                all_arr = arr_single
            else:
                all_arr = np.concatenate((all_arr, arr_single))
    return all_arr

def read_single_image(img_name, img_w, img_h):
    img = Image.open(img_name)
    img = img.resize((img_w,img_h))
    rgb = img.split()
    if len(rgb) == 1:
        print(img_name)
        return []

    r, g, b = img.split()
    img_size = img_w * img_h
    r_arr = plimg.pil_to_array(r)
    g_arr = plimg.pil_to_array(g)
    b_arr = plimg.pil_to_array(b)

    r_arr_re = r_arr.reshape(img_size)
    g_arr_re = g_arr.reshape(img_size)
    b_arr_re = b_arr.reshape(img_size)

    arr = np.concatenate((r_arr_re, g_arr_re, b_arr_re))
    return arr

def pickle_save(arr, output_file):
    img_data = {'image': arr}
    f = open(output_file, 'wb+')
    pickle.dump(img_data, f)
    f.close()

def unpickle_load_images(imgbinDir, num_channels, img_w, img_h):
    with open(imgbinDir, mode='rb') as file:
        data = pickle.load(file, encoding='bytes')
    raw_images = data['image']
    
    raw_float = np.array(raw_images, dtype=float) / 255.0
    images = raw_float.reshape([-1, num_channels, img_w, img_h])
    images = images.transpose([0, 2, 3, 1])
    return images
    
    
if __name__ == '__main__':
    imgDir = '/home/ruiliu/Development/mtml-tf/dataset/imagenet10k'
    outDir = '/home/ruiliu/Development/mtml-tf/dataset/imagenet10k.bin'
    arr_input = image_input(imgDir, 224, 224)
    pickle_save(arr_input, outDir)
    #print(arr.size)