import numpy as np
from PIL import Image
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--img', type=str, help='identify a image to test')
args = parser.parse_args()

imgfile = args.img
img_dir = '/home/ruiliu/Development/mtml-tf/dataset/imagenet10k'

#img = Image.open(imgfile)
img = Image.open(img_dir + '/' +imgfile)
img_byte = np.array(img, dtype=np.float32)
