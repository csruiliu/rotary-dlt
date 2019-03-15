import sys
sys.path.append('/home/rui/Development/tf-exp/models/research/slim')
sys.path.append('/home/rui/Development/tf-exp/models/official')

import tensorflow as tf
from tensorflow.python.framework import meta_graph
from nets.mobilenet import mobilenet_v2
from nets import resnet_v2
from mr_model import filter_net
from mr_utils import tensor_assign

tf.reset_default_graph()

img_w = 224
img_h = 224
img_size = img_w * img_h 
img_channel = 3

filter_net = tf.Graph()
with filter_net.as_default():
    input = tf.placeholder(tf.float32, shape=(img_w, img_h*2, img_channel), name="input")
    input = tf.reshape(input, [img_w*3,img_h*2])
    begin_slice=[0,0]
    size_slice=[224*3,224] 
    output = tf.slice(input, begin_slice, size_slice)

#input = tf.placeholder(tf.float32, shape=(img_w, img_h, img_channel), name="input")
#input = tf.Variable(tf.zeros([img_w, img_h*2]), name="input")
#inputs = tensor_assign(input, [0,0], 1)
#one_img = tf.expand_dims(input), -1)



with tf.Session(graph=filter_net) as sess:
    x = tf.get_variable("IN", shape=(224, 448, 3), initializer=tf.zeros_initializer())
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print(x.eval())
    #ins = inputs.eval()
    #filter_net(inputs.eval())
    ss = sess.run([output],feed_dict={input: x})
    print(ss.shape)
#mobile_net = tf.Graph()
#with mobile_net.as_default():
#    file_input_m = tf.placeholder(tf.string, ())
#    image_m = tf.image.decode_jpeg(tf.read_file(file_input_m))
#    images_m = tf.expand_dims(image_m, 0)
#    images_m = tf.cast(images, tf.float32) / 128.  - 1
#    images_m.set_shape((None, None, None, 3))
#    images_m = tf.image.resize_images(images_m, (224, 224))

#    with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope(is_training=False)):
#        logits_m, endpoints_m = mobilenet_v2.mobilenet(images_m)

#resnet_net = tf.Graph()
#with resnet_net.as_default():
#    file_input_r = tf.placeholder(tf.string, ())
#    image_r = tf.image.decode_jpeg(tf.read_file(file_input_r))
#    images_r = tf.expand_dims(image_r, 0)
#    images_r = tf.cast(images_r, tf.float32) / 128.  - 1
#    images_r.set_shape((None, None, None, 3))
#    images_r = tf.image.resize_images(images_r, (224, 224))

#    with tf.contrib.slim.arg_scope(resnet_v2.resnet_arg_scope()):
#        net_r, endpoints_r = resnet_v2.resnet_v2_50(images_r, 1001, is_training=False)

#graph = tf.get_default_graph()

#meta_graph_mobilenet = tf.train.export_meta_graph(graph=mobile_net)

#meta_graph.import_scoped_meta_graph(meta_graph_mobilenet, input_map={'input': x}, import_scope='mobilenet')
#out1 = graph.get_tensor_by_name('resnet/output:0')

#meta_graph_resnet = tf.train.export_meta_graph(graph=resnet_net)
#meta_graph.import_scoped_meta_graph(meta_graph_resnet, input_map={'input': out1}, import_scope='graph2')