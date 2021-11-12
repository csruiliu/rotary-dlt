import argparse

from rotary.profiler.nlp_profiler import NLPProfiler


def main():
    ###################################
    # get all parameters
    ###################################

    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model_name', action='store', type=str,
                        choices=['bert', 'lstm', 'bilstm'],
                        help='indicate training model')
    parser.add_argument('-l', '--hidden_layer', action='store', type=int,
                        choices=[2, 4, 8, 12],
                        help='indicate the hidden layer for bert')
    parser.add_argument('-s', '--hidden_size', action='store', type=int,
                        choices=[128, 256, 512, 768],
                        help='indicate the hidden size for bert')
    parser.add_argument('-sl', '--max_seq_length', action='store', type=int, default=128,
                        help='indicate the max sequence length for bert')

    parser.add_argument('-p', '--profile', action='store', type=str,
                        choices=['accuracy', 'steptime'], default='accuracy',
                        help='profile metric accuracy or steptime')
    parser.add_argument('-b', '--batch_size', action='store', type=int,
                        help='indicate the batch size for training.')
    parser.add_argument('-r', '--learning_rate', action='store', type=float,
                        help='indicate the learning rate for training.')
    parser.add_argument('-o', '--optimizer', action='store', type=str,
                        help='indicate the optimizer for training.')
    parser.add_argument('-e', '--epoch', action='store', type=int,
                        help='indicate the training epoch.')

    args = parser.parse_args()

    model_name = args.model_name
    model_hidden_layer = args.hidden_layer
    model_hidden_size = args.hidden_size
    model_max_seq_len = args.max_seq_length

    profile_metric = args.profile
    batch_size = args.batch_size
    learn_rate = args.learning_rate
    optimizer = args.optimizer
    epoch = args.epoch

    ###################################
    # profile the model training
    ###################################

    out_path = 'home/ruiliu/Development/rotary/knowledgebase'

    nlp_profiler = NLPProfiler(model_name,
                               batch_size,
                               optimizer,
                               learn_rate,
                               epoch,
                               profile_metric,
                               out_path,
                               model_hidden_layer=model_hidden_layer,
                               model_hidden_size=model_hidden_size,
                               model_max_seq_len=model_max_seq_len)

    nlp_profiler.run()


if __name__ == "__main__":
    main()
