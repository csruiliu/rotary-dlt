import tensorflow as tf

from workload.tensorflow_cifar.models.resnet import ResNet
from workload.tensorflow_cifar.models.densenet import DenseNet
from workload.tensorflow_cifar.models.mobilenet_v2 import MobileNetV2
from workload.tensorflow_cifar.models.mobilenet import MobileNet
from workload.tensorflow_cifar.models.vgg import VGG
from workload.tensorflow_cifar.models.lenet import LeNet
from workload.tensorflow_cifar.models.inception import Inception
from workload.tensorflow_cifar.models.alexnet import AlexNet
from workload.tensorflow_cifar.models.resnext import ResNeXt
from workload.tensorflow_cifar.models.xception import Xception
from workload.tensorflow_cifar.models.squeezenet import SqueezeNet
from workload.tensorflow_cifar.models.zfnet import ZFNet
from workload.tensorflow_cifar.models.efficientnet import EfficientNet
from workload.tensorflow_cifar.models.shufflenet import ShuffleNet
from workload.tensorflow_cifar.models.shufflenet_v2 import ShuffleNetV2

from workload.tensorflow_nlp.models.bert import BERT
from workload.tensorflow_nlp.models.lstm import LSTMNet
from workload.tensorflow_nlp.models.bi_lstm import BiLSTM


def build_nlp_model(model_type,
                    max_length,
                    opt,
                    lr):
    if model_type == 'bert':
        # only use bert-tiny
        model = BERT(max_length=max_length, hidden_size=128, num_hidden_layers=2, learn_rate=lr, optimizer=opt)

    elif model_type == 'lstm':
        model = LSTMNet(max_length=max_length, learn_rate=lr, optimizer=opt)

    elif model_type == 'bilstm':
        model = BiLSTM(max_length=max_length, learn_rate=lr, optimizer=opt)

    else:
        raise ValueError('model type is not support')

    return model


def build_cv_model(job_data,
                   opt,
                   lr,
                   n_class,
                   feature,
                   label):

    model_type = job_data['model']
    if model_type == 'alexnet':
        model = AlexNet(num_classes=n_class)
    elif model_type == 'densenet':
        model = DenseNet(residual_layer=121, num_classes=n_class)
    elif model_type == 'efficientnet':
        model = EfficientNet(num_classes=n_class)
    elif model_type == 'inception':
        model = Inception(num_classes=n_class)
    elif model_type == 'lenet':
        model = LeNet(num_classes=n_class)
    elif model_type == 'mobilenet':
        model = MobileNet(num_classes=n_class)
    elif model_type == 'mobilenet_v2':
        model = MobileNetV2(num_classes=n_class)
    elif model_type == 'resnet':
        model = ResNet(residual_layer=18, num_classes=n_class)
    elif model_type == 'resnext':
        model = ResNeXt(cardinality=2, num_classes=n_class)
    elif model_type == 'shufflenet':
        model = ShuffleNet(num_groups=2, num_classes=n_class)
    elif model_type == 'squeezenet':
        model = SqueezeNet(num_classes=n_class)
    elif model_type == 'vgg':
        model = VGG(conv_layer=11, num_classes=n_class)
    elif model_type == 'xception':
        model = Xception(num_classes=n_class)
    elif model_type == 'zfnet':
        model = ZFNet(num_classes=n_class)
    elif model_type == 'shufflenet_v2':
        model = ShuffleNetV2(complexity=1, num_classes=n_class)
    else:
        raise ValueError("the model type is not supported")

    logit = model.build(feature)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(label, logit)

    train_loss = tf.reduce_mean(cross_entropy)

    ###########################################
    # count overall trainable parameters
    ###########################################

    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters

    ###########################################
    # configure the optimizer for training
    ###########################################

    if opt == 'Adam':
        train_op = tf.train.AdamOptimizer(lr).minimize(train_loss)
    elif opt == 'SGD':
        train_op = tf.train.GradientDescentOptimizer(lr).minimize(train_loss)
    elif opt == 'Adagrad':
        train_op = tf.train.AdagradOptimizer(lr).minimize(train_loss)
    elif opt == 'Momentum':
        train_op = tf.train.MomentumOptimizer(lr, 0.9).minimize(train_loss)
    else:
        raise ValueError('Optimizer is not recognized')

    prediction = tf.equal(tf.argmax(logit, -1), tf.argmax(label, -1))
    eval_op = tf.reduce_mean(tf.cast(prediction, tf.float32))

    return train_op, eval_op, total_parameters
