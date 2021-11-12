import numpy as np
import os
import json
import urllib
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences

from rotary.models.nlp.bert import BERT
from rotary.models.nlp.lstm import LSTMNet
from rotary.models.nlp.bi_lstm import BiLSTM

import rotary.reader.udtb_reader as udtb_reader
import rotary.reader.lmrd_reader as lmrd_reader


class NLPProfiler:
    def __init__(self,
                 model_name,
                 batch_size,
                 optimizer,
                 learning_rate,
                 epoch,
                 profile_metric,
                 output_dir,
                 *args, **kwargs):

        self.model_name = model_name
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.profile_metric = profile_metric
        self.output_dir = output_dir

        if model_name == 'bert':
            self.max_seq_length = kwargs['model_max_seq_len']
            self.hidden_layer = kwargs['model_hidden_layer']
            self.hidden_size = kwargs['model_hidden_size']
            if self.max_seq_length is None:
                raise ValueError('Max sequence length for BERT is missing')
            elif self.hidden_layer is None:
                raise ValueError('hidden layer for BERT is missing')
            elif self.hidden_size is None:
                raise ValueError('hidden size for BERT is missing')
            else:
                pass

    def run(self):
        ###################################
        # prepare the model
        ###################################

        if self.model_name == 'bert':
            # Params for bert model and tokenization
            bert_path = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"

            train_df, test_df = lmrd_reader.download_and_load_datasets()

            # Create datasets (Only take up to max_seq_length words for memory)
            train_text = train_df["sentence"].tolist()
            train_text = [" ".join(t.split()[0:self.max_seq_length]) for t in train_text]
            train_text = np.array(train_text, dtype=object)[:, np.newaxis]
            train_label = train_df["polarity"].tolist()

            test_text = test_df["sentence"].tolist()
            test_text = [" ".join(t.split()[0:self.max_seq_length]) for t in test_text]
            test_text = np.array(test_text, dtype=object)[:, np.newaxis]
            test_label = test_df["polarity"].tolist()

            with tf.Session() as sess:
                # Instantiate tokenizer
                tokenizer = lmrd_reader.create_tokenizer_from_hub_module(bert_path, sess)

                # Convert data to InputExample format
                train_examples = lmrd_reader.convert_text_to_examples(train_text, train_label)
                test_examples = lmrd_reader.convert_text_to_examples(test_text, test_label)

                # Convert to features
                (
                    train_input_ids,
                    train_input_masks,
                    train_segment_ids,
                    train_labels,
                ) = lmrd_reader.convert_examples_to_features(tokenizer, train_examples, self.max_seq_length)

                (
                    test_input_ids,
                    test_input_masks,
                    test_segment_ids,
                    test_labels,
                ) = lmrd_reader.convert_examples_to_features(tokenizer, test_examples, self.max_seq_length)

                model = BERT(max_length=self.max_seq_length,
                             hidden_size=self.hidden_size,
                             num_hidden_layers=self.hidden_layer,
                             learn_rate=self.learning_rate,
                             optimizer=self.optimizer)

                logit, trainable_parameters = model.build()

                # Instantiate variables
                sess.run(tf.local_variables_initializer())
                sess.run(tf.global_variables_initializer())
                sess.run(tf.tables_initializer())
                tf.keras.backend.set_session(sess)

                hist = logit.fit([train_input_ids, train_input_masks, train_segment_ids],
                                 train_labels,
                                 epochs=self.epoch,
                                 batch_size=self.batch_size,
                                 verbose=0)

                acc_record_list = hist.history['acc']
                acc_record_list = [float(item) for item in acc_record_list]

                # scores = logit.evaluate([test_input_ids, test_input_masks, test_segment_ids], test_labels)
                # scores = logit.evaluate([train_input_ids, train_input_masks, train_segment_ids], train_labels)
                # print('{}: {}'.format(logit.metrics_names[1], scores[1]))

        elif self.model_name == 'bilstm' or self.model_name == 'lstm':

            ####################################################
            # Download and load the dataset
            ####################################################

            UD_ENGLISH_TRAIN = './datasets/ud_treebank/en_partut-ud-train.conllu'
            UD_ENGLISH_DEV = './datasets/ud_treebank/en_partut-ud-dev.conllu'
            UD_ENGLISH_TEST = './datasets/ud_treebank/en_partut-ud-test.conllu'

            if not os.path.exists(UD_ENGLISH_TRAIN):
                urllib.request.urlretrieve('http://archive.aueb.gr:8085/files/en_partut-ud-train.conllu',
                                           UD_ENGLISH_TRAIN)
            if not os.path.exists(UD_ENGLISH_DEV):
                urllib.request.urlretrieve('http://archive.aueb.gr:8085/files/en_partut-ud-dev.conllu',
                                           UD_ENGLISH_DEV)
            if not os.path.exists(UD_ENGLISH_TEST):
                urllib.request.urlretrieve('http://archive.aueb.gr:8085/files/en_partut-ud-test.conllu',
                                           UD_ENGLISH_TEST)

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

            if self.model_name == 'bilstm':
                model = BiLSTM(max_length=MAX_LENGTH,
                               learn_rate=self.learning_rate,
                               optimizer=self.optimizer)

            else:
                model = LSTMNet(max_length=MAX_LENGTH,
                                learn_rate=self.learning_rate,
                                optimizer=self.optimizer)

            logit, trainable_parameters = model.build(word2index, tag2index)

            hist = logit.fit(train_sentences_X,
                             udtb_reader.to_categorical(train_tags_y, len(tag2index)),
                             batch_size=self.batch_size,
                             epochs=self.epoch)

            acc_record_list = hist.history['acc']
            acc_record_list = [float(item) for item in acc_record_list]

        else:
            raise ValueError('model is not supported')

        results_path = self.output_dir + '/' + self.model_name + '_' + self.profile_metric

        if os.path.exists(results_path):
            with open(results_path) as f:
                model_json_list = json.load(f)
        else:
            model_json_list = list()

        # create a dict for the conf
        model_perf_dict = dict()

        model_perf_dict['model_name'] = self.model_name
        model_perf_dict['num_parameters'] = trainable_parameters
        model_perf_dict['batch_size'] = self.batch_size
        model_perf_dict['opt'] = self.optimizer
        model_perf_dict['learn_rate'] = self.learning_rate

        if self.model_name == 'bert':
            model_perf_dict['training_data'] = 'stanford-lmrd'
            model_perf_dict['classes'] = 2
        else:
            model_perf_dict['training_data'] = 'udtreebank'
            model_perf_dict['classes'] = 17

        model_perf_dict['accuracy'] = acc_record_list

        model_json_list.append(model_perf_dict)

        with open(results_path, 'w+') as f:
            json.dump(model_json_list, f)
