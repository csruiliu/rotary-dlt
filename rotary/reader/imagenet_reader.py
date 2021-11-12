import numpy as np
import cv2
import os


########################################################
# read imagenet raw images
########################################################
def load_imagenet_raw(image_dir, batch_list, img_h, img_w):
    img_list = []
    for img in batch_list:
        # print(image_dir+"/"+img)
        im = cv2.imread(image_dir + "/" + img, cv2.IMREAD_COLOR)
        res = cv2.resize(im, dsize=(img_w, img_h))
        res_exp = np.expand_dims(res, axis=0)
        img_list.append(res_exp)
    img_data = np.concatenate(img_list, axis=0)
    return img_data


########################################################
# read imagenet label
########################################################
def load_imagenet_labels_onehot(path):
    num_classes = 1000
    lines = open(path).readlines()
    labels_array = np.zeros((len(lines), num_classes))
    for idx, val in enumerate(lines):
        hot = int(val.rstrip('\n'))
        labels_array[idx, hot - 1] = 1
    return labels_array


########################################################
# read imagenet bin
########################################################
def load_imagenet_bin(path, num_channels, img_w, img_h):
    image_arr = np.fromfile(path, dtype=np.uint8)
    img_num = int(image_arr.size / img_w / img_h / num_channels)
    images = image_arr.reshape((img_num, img_w, img_h, num_channels))
    return images


########################################################
# convert imagenet raw images to bin
########################################################
def convert_imagenet_bin(path):
    img_w = 224
    img_h = 224
    img_list = []
    img_filename_list = sorted(os.listdir(path))
    for filename in img_filename_list:
        img = cv2.imread(os.path.join(path, filename))
        if img is not None:
            img_resize = cv2.resize(img, dsize=(img_w, img_h))
            img_expand = np.expand_dims(img_resize, axis=0)
            img_list.append(img_expand)
    img_data = np.concatenate(img_list, axis=0)
    output_file = open("imagenet1k.bin", "wb")
    binary_format = bytearray(img_data)
    output_file.write(binary_format)
    output_file.close()
