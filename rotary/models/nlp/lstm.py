from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


class LSTMNet:
    def __init__(self, max_length, learn_rate=0.01, optimizer='Adam'):
        self.max_length = max_length
        if optimizer == 'Adam':
            self.opt = keras.optimizers.Adam(learning_rate=learn_rate)
        elif optimizer == 'SGD':
            self.opt = keras.optimizers.SGD(learning_rate=learn_rate)
        elif optimizer == 'Adagrad':
            self.opt = keras.optimizers.Adagrad(learning_rate=learn_rate)
        elif optimizer == 'Momentum':
            self.opt = keras.optimizers.SGD(
                learning_rate=learn_rate, decay=1e-6, momentum=0.9)
        else:
            raise ValueError('Optimizer is not recognized')

    def build(self, word2index, tag2index):
        model = keras.Sequential()
        model.add(layers.InputLayer(input_shape=(self.max_length,)))
        model.add(layers.Embedding(len(word2index), 128))
        model.add(layers.LSTM(256, return_sequences=True))
        model.add(layers.LSTM(256, return_sequences=True))
        model.add(layers.TimeDistributed(layers.Dense(len(tag2index))))
        model.add(layers.Activation('softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer=self.opt,
                      metrics=['accuracy'])

        trainable_count = int(
            np.sum([keras.backend.count_params(p) for p in set(model.trainable_weights)]))

        return model, trainable_count
