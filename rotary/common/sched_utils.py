import numpy as np
import tensorflow as tf

from rotary.reader import lmrd_reader


def get_bert_dataset(max_seq_length=128):
    train_df = lmrd_reader.download_and_load_datasets()

    # Create datasets (Only take up to max_seq_length words for memory)
    train_text = train_df["sentence"].tolist()
    train_text = [" ".join(t.split()[0:max_seq_length]) for t in train_text]
    train_text = np.array(train_text, dtype=object)[:, np.newaxis]
    train_label = train_df["polarity"].tolist()

    return train_text, train_label


def prepare_bert_dataset(bert_path, tf_sess, train_text, train_label, max_seq_length):
    # Instantiate tokenizer
    tokenizer = lmrd_reader.create_tokenizer_from_hub_module(bert_path, tf_sess)
    # Convert data to InputExample format
    train_examples = lmrd_reader.convert_text_to_examples(train_text, train_label)
    # Convert to features
    (
        train_input_ids,
        train_input_masks,
        train_segment_ids,
        train_labels,
    ) = lmrd_reader.convert_examples_to_features(tokenizer,
                                                 train_examples,
                                                 max_seq_length)

    return train_input_ids, train_input_masks, train_segment_ids, train_labels


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


def compared_item(item):
    return item['goal_value']
