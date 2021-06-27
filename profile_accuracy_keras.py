import argparse
import numpy as np
import os
import re
import json
import pandas as pd
import urllib
import tensorflow as tf
import tensorflow_hub as hub
from bert.tokenization import FullTokenizer
from tensorflow.keras import backend as K
from keras.preprocessing.sequence import pad_sequences

from workload.tensorflow_nlp.models.bert import BERT
from workload.tensorflow_nlp.models.bi_lstm import BiLSTM
from workload.tensorflow_nlp.models.lstm import LSTMNet
import workload.tensorflow_nlp.tools.udtb_reader as udtb_reader


# Load all files from a directory in a DataFrame.
def load_directory_data(directory):
    data = {}
    data["sentence"] = []
    data["sentiment"] = []
    for file_path in os.listdir(directory):
        with tf.gfile.GFile(os.path.join(directory, file_path), "r") as f:
            data["sentence"].append(f.read())
            data["sentiment"].append(re.match("\d+_(\d+)\.txt", file_path).group(1))
    return pd.DataFrame.from_dict(data)


# Merge positive and negative examples, add a polarity column and shuffle.
def load_dataset(directory):
    pos_df = load_directory_data(os.path.join(directory, "pos"))
    neg_df = load_directory_data(os.path.join(directory, "neg"))
    pos_df["polarity"] = 1
    neg_df["polarity"] = 0
    return pd.concat([pos_df, neg_df]).sample(frac=1).reset_index(drop=True)


# Download and process the dataset files.
def download_and_load_datasets(force_download=False):
    if not os.path.exists('/tank/local/ruiliu/dataset/aclImdb.tar.gz'):
        dataset = tf.keras.utils.get_file(
            fname="aclImdb.tar.gz",
            origin="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
            extract=True,
            cache_dir='/tank/local/ruiliu/'
        )
    else:
        dataset = '/tank/local/ruiliu/dataset/aclImdb.tar.gz'

    train_df = load_dataset(os.path.join(os.path.dirname(dataset), "aclImdb", "train"))
    test_df = load_dataset(os.path.join(os.path.dirname(dataset), "aclImdb", "test"))

    return train_df, test_df


class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.
    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.
    We use this class instead of `None` because treating `None` as padding
    batches could cause silent errors.
    """


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
            Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence.
                    For single sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
                    Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example.
                   This should be specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


def create_tokenizer_from_hub_module(bert_path, sess):
    """Get the vocab file and casing info from the Hub module."""
    bert_module = hub.Module(bert_path)
    tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
    vocab_file, do_lower_case = sess.run(
        [tokenization_info["vocab_file"], tokenization_info["do_lower_case"]]
    )

    return FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)


def convert_single_example(tokenizer, example, max_seq_length=256):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        input_ids = [0] * max_seq_length
        input_mask = [0] * max_seq_length
        segment_ids = [0] * max_seq_length
        label = 0
        return input_ids, input_mask, segment_ids, label

    tokens_a = tokenizer.tokenize(example.text_a)
    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0: (max_seq_length - 2)]

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return input_ids, input_mask, segment_ids, example.label


def convert_examples_to_features(tokenizer, examples, max_seq_length=256):
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""

    input_ids, input_masks, segment_ids, labels = [], [], [], []
    # for example in tqdm(examples, desc="Converting examples to features"):
    for example in examples:
        input_id, input_mask, segment_id, label = convert_single_example(
            tokenizer, example, max_seq_length
        )
        input_ids.append(input_id)
        input_masks.append(input_mask)
        segment_ids.append(segment_id)
        labels.append(label)
    return (
        np.array(input_ids),
        np.array(input_masks),
        np.array(segment_ids),
        np.array(labels).reshape(-1, 1),
    )


def convert_text_to_examples(texts, labels):
    """Create InputExamples"""
    InputExamples = []
    for text, label in zip(texts, labels):
        InputExamples.append(
            InputExample(guid=None, text_a=" ".join(text), text_b=None, label=label)
        )
    return InputExamples


def to_categorical(sequences, categories):
    cat_sequences = []
    for s in sequences:
        cats = []
        for item in s:
            cats.append(np.zeros(categories))
            cats[-1][item] = 1.0
        cat_sequences.append(cats)
    return np.array(cat_sequences)


if __name__ == "__main__":

    ###################################
    # get all parameters
    ###################################

    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', action='store', type=str, help='indicate training model')

    parser.add_argument('-l', '--hidden_layer', action='store', type=int,
                        help='indicate the hidden layer for bert')

    parser.add_argument('-s', '--hidden_size', action='store', type=int,
                        help='indicate the hidden size for bert')

    parser.add_argument('-b', '--batch_size', action='store', type=int, help='indicate the batch size for training.')
    parser.add_argument('-r', '--learn_rate', action='store', type=float,
                        help='indicate the learning rate for training.')
    parser.add_argument('-o', '--opt', action='store', type=str, help='indicate the optimizer for training.')
    parser.add_argument('-e', '--epoch', action='store', type=int, help='indicate the training epoch.')

    args = parser.parse_args()

    ###################################
    # prepare the model
    ###################################

    if args.model == 'bert':
        # Params for bert model and tokenization
        bert_path = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
        max_seq_length = 128

        train_df, test_df = download_and_load_datasets()

        # Create datasets (Only take up to max_seq_length words for memory)
        train_text = train_df["sentence"].tolist()
        train_text = [" ".join(t.split()[0:max_seq_length]) for t in train_text]
        train_text = np.array(train_text, dtype=object)[:, np.newaxis]
        train_label = train_df["polarity"].tolist()

        test_text = test_df["sentence"].tolist()
        test_text = [" ".join(t.split()[0:max_seq_length]) for t in test_text]
        test_text = np.array(test_text, dtype=object)[:, np.newaxis]
        test_label = test_df["polarity"].tolist()

        with tf.Session() as sess:
            # Instantiate tokenizer
            tokenizer = create_tokenizer_from_hub_module(bert_path, sess)

            # Convert data to InputExample format
            train_examples = convert_text_to_examples(train_text, train_label)
            test_examples = convert_text_to_examples(test_text, test_label)

            # Convert to features
            (
                train_input_ids,
                train_input_masks,
                train_segment_ids,
                train_labels,
            ) = convert_examples_to_features(tokenizer, train_examples, max_seq_length=max_seq_length)

            (
                test_input_ids,
                test_input_masks,
                test_segment_ids,
                test_labels,
            ) = convert_examples_to_features(tokenizer, test_examples, max_seq_length=max_seq_length)

            model = BERT(max_length=max_seq_length,
                         hidden_size=args.hidden_size,
                         num_hidden_layers=args.hidden_layer,
                         learn_rate=args.learn_rate,
                         optimizer=args.opt)

            logit, trainable_parameters = model.build()

            # Instantiate variables
            sess.run(tf.local_variables_initializer())
            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())
            K.set_session(sess)

            hist = logit.fit([train_input_ids, train_input_masks, train_segment_ids],
                             train_labels,
                             epochs=args.epoch,
                             batch_size=args.batch_size,
                             verbose=0)

            acc_record_list = hist.history['acc']
            acc_record_list = [float(item) for item in acc_record_list]

            # scores = logit.evaluate([test_input_ids, test_input_masks, test_segment_ids], test_labels)
            # scores = logit.evaluate([train_input_ids, train_input_masks, train_segment_ids], train_labels)
            # print('{}: {}'.format(logit.metrics_names[1], scores[1]))

    elif args.model == 'bilstm' or args.model == 'lstm':

        ####################################################
        # Download and load the dataset
        ####################################################

        UD_ENGLISH_TRAIN = './datasets/ud_treebank/en_partut-ud-train.conllu'
        UD_ENGLISH_DEV = './datasets/ud_treebank/en_partut-ud-dev.conllu'
        UD_ENGLISH_TEST = './datasets/ud_treebank/en_partut-ud-test.conllu'

        if not os.path.exists(UD_ENGLISH_TRAIN):
            urllib.request.urlretrieve('http://archive.aueb.gr:8085/files/en_partut-ud-train.conllu', UD_ENGLISH_TRAIN)
        if not os.path.exists(UD_ENGLISH_DEV):
            urllib.request.urlretrieve('http://archive.aueb.gr:8085/files/en_partut-ud-dev.conllu', UD_ENGLISH_DEV)
        if not os.path.exists(UD_ENGLISH_TEST):
            urllib.request.urlretrieve('http://archive.aueb.gr:8085/files/en_partut-ud-test.conllu', UD_ENGLISH_TEST)

        train_sentences = udtb_reader.read_conllu(UD_ENGLISH_TRAIN)
        val_sentences = udtb_reader.read_conllu(UD_ENGLISH_DEV)
        test_sentences = udtb_reader.read_conllu(UD_ENGLISH_TEST)

        ####################################################
        # Preprocessing
        ####################################################

        train_text = udtb_reader.text_sequence(train_sentences)
        test_text = udtb_reader.text_sequence(test_sentences)
        # val_text = udtb_reader.text_sequence(val_sentences)

        train_label = udtb_reader.tag_sequence(train_sentences)
        test_label = udtb_reader.tag_sequence(test_sentences)
        # val_label = udtb_reader.tag_sequence(val_sentences)

        ####################################################
        # Build dictionary with tag vocabulary
        ####################################################

        words, tags = set([]), set([])

        for s in train_text:
            for w in s:
                words.add(w.lower())

        for ts in train_label:
            for t in ts:
                tags.add(t)

        word2index = {w: i + 2 for i, w in enumerate(list(words))}
        # The special value used for padding
        word2index['-PAD-'] = 0
        # The special value used for OOVs
        word2index['-OOV-'] = 1

        tag2index = {t: i + 1 for i, t in enumerate(list(tags))}
        # The special value used to padding
        tag2index['-PAD-'] = 0

        train_sentences_X, test_sentences_X, train_tags_y, test_tags_y = [], [], [], []

        for s in train_text:
            s_int = []
            for w in s:
                try:
                    s_int.append(word2index[w.lower()])
                except KeyError:
                    s_int.append(word2index['-OOV-'])

            train_sentences_X.append(s_int)

        for s in test_text:
            s_int = []
            for w in s:
                try:
                    s_int.append(word2index[w.lower()])
                except KeyError:
                    s_int.append(word2index['-OOV-'])

            test_sentences_X.append(s_int)

        for s in train_label:
            train_tags_y.append([tag2index[t] for t in s])

        for s in test_label:
            test_tags_y.append([tag2index[t] for t in s])

        MAX_LENGTH = len(max(train_sentences_X, key=len))

        train_sentences_X = pad_sequences(train_sentences_X, maxlen=MAX_LENGTH, padding='post')
        test_sentences_X = pad_sequences(test_sentences_X, maxlen=MAX_LENGTH, padding='post')
        train_tags_y = pad_sequences(train_tags_y, maxlen=MAX_LENGTH, padding='post')
        test_tags_y = pad_sequences(test_tags_y, maxlen=MAX_LENGTH, padding='post')

        if args.model == 'bilstm':
            model = BiLSTM(max_length=MAX_LENGTH,
                           learn_rate=args.learn_rate,
                           optimizer=args.opt)

        else:
            model = LSTMNet(max_length=MAX_LENGTH,
                            learn_rate=args.learn_rate,
                            optimizer=args.opt)

        logit, trainable_parameters = model.build(word2index, tag2index)

        hist = logit.fit(train_sentences_X,
                         to_categorical(train_tags_y, len(tag2index)),
                         batch_size=args.batch_size,
                         epochs=args.epoch)

        acc_record_list = hist.history['acc']
        acc_record_list = [float(item) for item in acc_record_list]

    else:
        raise ValueError('model is not supported')

    json_acc_path = '/tank/local/ruiliu/rotary/knowledgebase/model_acc.json'

    if os.path.exists(json_acc_path):
        with open(json_acc_path) as f:
            model_json_list = json.load(f)
    else:
        model_json_list = list()

    # create a dict for the conf
    model_perf_dict = dict()

    model_perf_dict['model_name'] = args.model
    model_perf_dict['num_parameters'] = trainable_parameters
    model_perf_dict['batch_size'] = args.batch_size
    model_perf_dict['opt'] = args.opt
    model_perf_dict['learn_rate'] = args.learn_rate

    if args.model == 'bert':
        model_perf_dict['training_data'] = 'stanford-lmrd'
        model_perf_dict['classes'] = 2
    else:
        model_perf_dict['training_data'] = 'udtreebank'
        model_perf_dict['classes'] = 17

    model_perf_dict['accuracy'] = acc_record_list

    model_json_list.append(model_perf_dict)

    with open(json_acc_path, 'w+') as f:
        json.dump(model_json_list, f)
