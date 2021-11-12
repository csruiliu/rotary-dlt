import numpy as np


#####################################
# read mnist training data
#####################################
def load_mnist_image(path):
    mnist_numChannels = 1
    with open(path, 'rb') as bytestream:
        _ = int.from_bytes(bytestream.read(4), byteorder='big')
        num_images = int.from_bytes(bytestream.read(4), byteorder='big')
        mnist_imgWidth = int.from_bytes(bytestream.read(4), byteorder='big')
        mnist_imgHeight = int.from_bytes(bytestream.read(4), byteorder='big')
        buf = bytestream.read(mnist_imgWidth * mnist_imgHeight * num_images)
        img_raw = np.frombuffer(buf, dtype=np.uint8).astype(np.float32) / 255.0
        img = img_raw.reshape(num_images, mnist_imgHeight,
                              mnist_imgWidth, mnist_numChannels)

    return img


#####################################
# read mnist test data
#####################################
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
