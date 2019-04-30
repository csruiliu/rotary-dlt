# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 2019
@author: Rui Liu
"""

import tensorflow as tf
import preprocessing

import resnet
import timeit

tf.app.flags.DEFINE_string('checkpoint_path','/home/rui/Development/mtml-tf/ckpt/resnet/resnet_v2_50.ckpt','Path to ckpt of ResNet50')
tf.app.flags.DEFINE_string('record_path','/home/rui/Development/mtml-tf/dataset/kaggle/train.record','Path to training tfrecord file.')
tf.app.flags.DEFINE_string('log_path','/home/rui/Development/mtml-tf/mt-train','Path to log directory.')

tf.app.flags.DEFINE_float('learning_rate', 0.0001, 'Initial learning rate.')
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.1, 'Learning rate decay factor.')
tf.app.flags.DEFINE_integer('batch_size', 48, 'Batch size')
tf.app.flags.DEFINE_float('num_epochs_per_decay', 1.0, 'Number of epochs after which learning rate decays.')
tf.app.flags.DEFINE_integer('num_samples', 25000, 'Number of samples.')
tf.app.flags.DEFINE_integer('num_steps', 10000, 'Number of steps.')

FLAGS = tf.app.flags.FLAGS
slim = tf.contrib.slim

def get_record_dataset(record_path,reader=None,num_samples=25000,num_classes=2):
    if not reader:
        reader = tf.TFRecordReader

    keys_to_features = {'image/encoded':tf.FixedLenFeature((), tf.string, default_value=''),
                        'image/format':tf.FixedLenFeature((), tf.string, default_value='jpeg'),
                        'image/class/label':tf.FixedLenFeature([1], tf.int64, default_value=tf.zeros([1], dtype=tf.int64))}

    items_to_handlers = {'image': slim.tfexample_decoder.Image(image_key='image/encoded',format_key='image/format'),
                         'label': slim.tfexample_decoder.Tensor('image/class/label', shape=[])}

    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)
    labels_to_names = None
    items_to_descriptions = {'image': 'An image with shape image_shape.',
                             'label': 'A single integer.'}
    return tf.contrib.slim.dataset.Dataset(
        data_sources=record_path,
        reader=reader,
        decoder=decoder,
        num_samples=num_samples,
        num_classes=num_classes,
        items_to_descriptions=items_to_descriptions,
        labels_to_names=labels_to_names)

def configure_learning_rate(num_samples_per_epoch, global_step):
    decay_steps = int(num_samples_per_epoch * FLAGS.num_epochs_per_decay /
                      FLAGS.batch_size)
    return tf.train.exponential_decay(FLAGS.learning_rate,
                                      global_step,
                                      decay_steps,
                                      FLAGS.learning_rate_decay_factor,
                                      staircase=True,
                                      name='exponential_decay_learning_rate')

def get_init_fn(checkpoint_exclude_scopes=None):
    if FLAGS.checkpoint_path is None:
        return None

    if tf.train.latest_checkpoint(FLAGS.log_path):
        tf.logging.info('Ignoring --checkpoint_path because a checkpoint already exists in %s' % FLAGS.log_path)
        return None

    exclusions = []
    if checkpoint_exclude_scopes:
        exclusions = [scope.strip() for scope in checkpoint_exclude_scopes.split(',')]

    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
        if not excluded:
            variables_to_restore.append(var)

    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    else:
        checkpoint_path = FLAGS.checkpoint_path

    tf.logging.info('Fine-tuning from %s' % checkpoint_path)

    return slim.assign_from_checkpoint_fn(
        checkpoint_path,
        variables_to_restore,
        ignore_missing_vars=True)

def get_trainable_variables(checkpoint_exclude_scopes=None):
    exclusions = []
    if checkpoint_exclude_scopes:
        exclusions = [scope.strip() for scope in
                     checkpoint_exclude_scopes.split(',')]
    variables_to_train = []
    for var in tf.trainable_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
        if not excluded:
            variables_to_train.append(var)
    return variables_to_train

def main(_):
    num_samples = FLAGS.num_samples
    dataset = get_record_dataset(FLAGS.record_path, num_samples=num_samples, num_classes=61)
    data_provider = slim.dataset_data_provider.DatasetDataProvider(dataset)
    image, label = data_provider.get(['image', 'label'])
    image = preprocessing.border_expand(image, resize=True, output_height=368, output_width=368)
    inputs, labels = tf.train.batch([image, label],batch_size=FLAGS.batch_size,allow_smaller_final_batch=True)
    cls_model = resnet.ResNet(is_training=True, num_classes=61)
    preprocessed_inputs = cls_model.preprocess(inputs)
    prediction_dict = cls_model.predict(preprocessed_inputs)
    loss_dict = cls_model.loss(prediction_dict, labels)
    loss = loss_dict['loss']
    postprocessed_dict = cls_model.postprocess(prediction_dict)
    acc = cls_model.accuracy(postprocessed_dict, labels)
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', acc)
    global_step = slim.create_global_step()
    learning_rate = configure_learning_rate(num_samples, global_step)
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.9)
    vars_to_train = get_trainable_variables()
    train_op = slim.learning.create_train_op(loss, optimizer,
                                             summarize_gradients=True,
                                             variables_to_train=vars_to_train)
    tf.summary.scalar('learning_rate', learning_rate)

    init_fn = get_init_fn()

    slim.learning.train(train_op=train_op, logdir=FLAGS.log_path,
                        init_fn=init_fn, number_of_steps=FLAGS.num_steps,
                        save_summaries_secs=20,
                        save_interval_secs=600)

if __name__ == '__main__':
    tf.app.run()
