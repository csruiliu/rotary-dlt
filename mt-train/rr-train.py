import sys
sys.path.append('/home/ruiliu/Development/mtml-tf/models/research/slim')
sys.path.append('/home/ruiliu/Development/mtml-tf/models/official')

import tensorflow as tf
from rr_model import rr_net

tf.reset_default_graph()

net = tf.Graph()
with net.as_default():

