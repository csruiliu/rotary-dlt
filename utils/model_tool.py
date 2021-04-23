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

from workload.tensorflow_ptb.models.nnlm import NNLM
from workload.tensorflow_ptb.models.word2vec import Word2Vec
from workload.tensorflow_ptb.models.bi_lstm import BiLSTM
from workload.tensorflow_ptb.models.textrnn import TextRNN


def build_model(job_data,
                n_class,
                feature,
                label):

    job_id = job_data['id']
    model_type = job_data['model_type']
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
        model = ResNeXt(cardinality=8, num_classes=n_class)
    elif model_type == 'shufflenet':
        model = ShuffleNet(num_groups=2, num_classes=n_class)
    elif model_type == 'squeezenet':
        model = SqueezeNet(num_classes=n_class)
    elif model_type == 'vgg':
        model = VGG(conv_layer=16, num_classes=n_class)
    elif model_type == 'xception':
        model = Xception(num_classes=n_class)
    elif model_type == 'zfnet':
        model = ZFNet(num_classes=n_class)
    elif model_type == 'shufflenet_v2':
        model = ShuffleNetV2(complexity=1, num_classes=n_class)
    elif model_type == 'nnlm':
        model = NNLM(n_class=n_class, n_step=2, n_hidden=2)
    elif model_type == 'bilstm':
        model = BiLSTM(n_class=n_class, n_step=2, n_hidden=2)
    elif model_type == 'word2vec':
        model = Word2Vec(voc_size=n_class, embedding_size=2)
    elif model_type == 'textrnn':
        model = TextRNN(n_class=n_class, n_step=2, n_hidden=2)
    else:
        raise ValueError("the model type is not supported")

    logit = model.build(feature)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(label, logit)
    train_loss = tf.reduce_mean(cross_entropy)
    tf.trainable_variables()
    train_op = tf.train.AdamOptimizer(3e-4).minimize(train_loss)

    prediction = tf.equal(tf.argmax(model, -1), tf.argmax(label, -1))
    eval_op = tf.reduce_mean(tf.cast(prediction, tf.float32))

    job_name = str(job_id) + '-' + str(model_type)

    return train_op, eval_op, job_name
