import numpy as np
import tensorflow as tf

import rotary.reader.lmrd_reader as lmrd_reader
from rotary.common.model_utils import build_nlp_model


def get_bert_dataset(max_seq_length=128):
    train_df = lmrd_reader.download_and_load_datasets()

    # Create datasets (Only take up to max_seq_length words for memory)
    train_text = train_df["sentence"].tolist()
    train_text = [" ".join(t.split()[0:max_seq_length]) for t in train_text]
    train_text = np.array(train_text, dtype=object)[:, np.newaxis]
    train_label = train_df["polarity"].tolist()

    return train_text, train_label


# init the config for training
def init_tf_config():
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True

    return config


# init the variable for training
def init_tf_vars(sess):
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    tf.keras.backend.set_session(sess)
