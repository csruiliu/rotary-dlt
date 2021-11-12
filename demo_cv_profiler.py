import argparse
from rotary.profiler.cv_profiler import CVProfiler


def main():

    ###################################
    # get all parameters
    ###################################

    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model_name', action='store', type=str,
                        choices=['alexnet', 'vgg', 'zfnet', 'lenet', 'resnet', 'resnext', 'xception',
                                 'squeezene', 'shufflenet_v2', 'shufflenet', 'inception',
                                 'efficientnet', 'densenet', 'mobilenet', 'mobilenet_v2'],
                        required=True,
                        help='indicate training model')
    parser.add_argument('-l', '--layer', action='store', type=int,
                        help='indicate the layer for some models like resnet, densenet, vgg')
    parser.add_argument('-g', '--group', action='store', type=int,
                        help='indicate the conv group for shufflenet')
    parser.add_argument('-x', '--complex', action='store', type=float,
                        help='indicate the model complex for shufflenetv2')
    parser.add_argument('-c', '--card', action='store', type=int,
                        help='indicate the cardinality for resnext')

    parser.add_argument('-p', '--profile', action='store', type=str,
                        choices=['accuracy', 'steptime'], default='accuracy',
                        help='profile metric accuracy or steptime')
    parser.add_argument('-d', '--dataset', action='store', type=str, default='cifar10',
                        help='indicate the training dataset.')
    parser.add_argument('-b', '--batch_size', action='store', type=int, default=32,
                        help='indicate the batch size for training.')
    parser.add_argument('-r', '--learning_rate', action='store', type=float, default=0.001,
                        help='indicate the learning rate for training.')
    parser.add_argument('-o', '--optimizer', action='store', type=str, default='Momentum',
                        help='indicate the optimizer for training.')
    parser.add_argument('-e', '--epoch', action='store', type=int, default=1,
                        help='indicate the training epoch.')

    args = parser.parse_args()

    model_name = args.model_name
    model_layer = args.layer
    model_group = args.group
    model_complex = args.complex
    model_card = args.card

    profile_metric = args.profile
    train_dataset = args.dataset
    batch_size = args.batch_size
    learn_rate = args.learning_rate
    optimizer = args.optimizer
    epoch = args.epoch

    ###################################
    # profile the model training
    ###################################

    out_path = 'home/ruiliu/Development/rotary/knowledgebase'

    cv_profiler = CVProfiler(model_name,
                             train_dataset,
                             batch_size,
                             optimizer,
                             learn_rate,
                             epoch,
                             profile_metric,
                             out_path,
                             model_layer=model_layer,
                             model_group=model_group,
                             model_complex=model_complex,
                             model_card=model_card)

    cv_profiler.run()


if __name__ == "__main__":
    main()
