import numpy as np
import pickle
import os
from PIL import Image
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
        arr_single = read_single_image(imgDir + '/' + filename, img_h, img_w)
        if all_arr == []:
            all_arr = arr_single
        else:
            all_arr = np.concatenate((all_arr, arr_single))    
    return all_arr

# use pure write to store
def save_bin_raw(arr, output_file):
    with open(output_file, 'wb+') as of:
        of.write(arr)

def load_bin_raw(img_bin, num_channels, img_w, img_h):
    image_arr = np.fromfile(img_bin, dtype=np.uint8)
    img_num = int(image_arr.size / img_w / img_h / num_channels)
    images = image_arr.reshape((img_num, img_w, img_h, num_channels))
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

# read single image
def read_single_image(img_name, img_h, img_w):
    img = Image.open(img_name).convert("RGB")
    img_resize = img.resize((img_h, img_w), Image.NEAREST)
    np_im = np.array(img_resize, dtype=np.uint8)
    img_ext = np.expand_dims(np_im, axis=0)
    return img_ext

def load_labels_onehot(path, num_classes):
    lines = open(path).readlines()
    labels_array = np.zeros((len(lines), num_classes))
    for idx, val in enumerate(lines):
        hot = int(val.rstrip('\n'))
        labels_array[idx, hot-1] = 1
    return labels_array


# read mnist database
def load_mnist_image(path):
    mnist_numChannels = 1
    with open(path, 'rb') as bytestream:
        _ = int.from_bytes(bytestream.read(4), byteorder='big')
        num_images = int.from_bytes(bytestream.read(4), byteorder='big')
        mnist_imgWidth = int.from_bytes(bytestream.read(4), byteorder='big')
        mnist_imgHeight = int.from_bytes(bytestream.read(4), byteorder='big')
        buf = bytestream.read(mnist_imgWidth * mnist_imgHeight * num_images)
        img_raw = np.frombuffer(buf, dtype=np.uint8).astype(np.float32) / 255.0
        img = img_raw.reshape(num_images, mnist_imgHeight, mnist_imgWidth, mnist_numChannels)
    return img

def load_mnist_label_onehot(path):
    num_classes = 10
    with open(path, 'rb') as bytestream:
        _ = int.from_bytes(bytestream.read(4), byteorder='big')
        num_images = int.from_bytes(bytestream.read(4), byteorder='big')
        buf = bytestream.read(num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
        
        labels_array = np.zeros((num_images, num_classes))
        for lidx, lval in enumerate(labels):
            labels_array[lidx, lval] = 1 

    return labels_array

if __name__ == "__main__":

    #imgPath = '/home/ruiliu/Development/mtml-tf/dataset/imagenet10k'
    #imgPath = '/tank/local/ruiliu/dataset/imagenet10k'
    #binPath = '/home/ruiliu/Development/mtml-tf/dataset/imagenet10k.bin'
    #binPath = '/tank/local/ruiliu/dataset/imagenet10k.bin'
    #labelPath = '/home/ruiliu/Development/mtml-tf/dataset/imagenet10k-label.txt'
    #labelPath = '/tank/local/ruiliu/dataset/imagenet10k-label.txt'

    #mnist_train_img_path = '/home/ruiliu/Development/mtml-tf/dataset/mnist-train-images.idx3-ubyte'
    mnist_train_label_path = '/tank/local/ruiliu/dataset/mnist-train-images.idx3-ubyte'
    #mnist_train_label_path = '/home/ruiliu/Development/mtml-tf/dataset/mnist-train-labels.idx1-ubyte'
    mnist_train_label_path = '/tank/local/ruiliu/dataset/mnist-train-labels.idx1-ubyte'
    #mnist_t10k_img_path = '/home/ruiliu/Development/mtml-tf/dataset/mnist-t10k-images.idx3-ubyte'
    mnist_t10k_img_path = '/tank/local/ruiliu/dataset/mnist-t10k-images.idx3-ubyte'
    #mnist_t10k_label_path = '/home/ruiliu/Development/mtml-tf/dataset/mnist-t10k-labels.idx1-ubyte'
    mnist_t10k_label_path = '/tank/local/ruiliu/dataset/mnist-t10k-labels.idx1-ubyte'

    
    #images = load_mnist_image(mnist_t10k_img_path)
    #print(images[100,:,:,:])
    #labels = read_mnist_label(mnist_t10k_label_path)
    
    #arr_image = convert_image_bin(imgDir, imgWidth, imgHeight)
    #save_bin_raw(arr_image, outDir)
    #images = load_bin_raw(binPath, numChannels, imgHeight, imgWidth)
    
    #print(images.shape)
    #print(labels[99])
    #plt.imshow(images[99,:,:,1], cmap='gray')
    #plt.show()