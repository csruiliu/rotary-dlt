import tensorflow as tf
import tensorflow_hub as hub
import numpy as np


class BertLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        n_fine_tune_layers=2,
        pooling="mean",
        bert_path="https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1",
        **kwargs,
    ):
        self.n_fine_tune_layers = n_fine_tune_layers
        self.trainable = True
        self.output_size = 768
        self.pooling = pooling
        self.bert_path = bert_path
        if self.pooling not in ["first", "mean"]:
            raise NameError(
                f"Undefined pooling type (must be either first or mean, but is {self.pooling}"
            )

        super(BertLayer, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'n_fine_tune_layers': self.n_fine_tune_layers,
            'pooling': self.pooling,
            'bert_path': self.bert_path
        })
        return config

    def build(self, input_shape):
        self.bert = hub.Module(
            self.bert_path, trainable=self.trainable, name=f"{self.name}_module"
        )

        # Remove unused layers
        trainable_vars = self.bert.variables
        if self.pooling == "first":
            trainable_vars = [
                var for var in trainable_vars if not "/cls/" in var.name]
            trainable_layers = ["pooler/dense"]

        elif self.pooling == "mean":
            trainable_vars = [
                var
                for var in trainable_vars
                if not "/cls/" in var.name and not "/pooler/" in var.name
            ]
            trainable_layers = []
        else:
            raise NameError(
                f"Undefined pooling type (must be either first or mean, but is {self.pooling}"
            )

        # Select how many layers to fine tune
        for i in range(self.n_fine_tune_layers):
            trainable_layers.append(f"encoder/layer_{str(11 - i)}")

        # Update trainable vars to contain only the specified layers
        trainable_vars = [
            var
            for var in trainable_vars
            if any([l in var.name for l in trainable_layers])
        ]

        # Add to trainable weights
        for var in trainable_vars:
            self._trainable_weights.append(var)

        for var in self.bert.variables:
            if var not in self._trainable_weights:
                self._non_trainable_weights.append(var)

        super(BertLayer, self).build(input_shape)

    def call(self, inputs):
        inputs = [tf.keras.backend.cast(x, dtype="int32") for x in inputs]
        input_ids, input_mask, segment_ids = inputs
        bert_inputs = dict(
            input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids
        )
        if self.pooling == "first":
            pooled = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)[
                "pooled_output"
            ]
        elif self.pooling == "mean":
            result = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)[
                "sequence_output"
            ]

            def mul_mask(x, m): return x * tf.expand_dims(m, axis=-1)
            def masked_reduce_mean(x, m): return tf.reduce_sum(mul_mask(x, m), axis=1) / (
                tf.reduce_sum(m, axis=1, keepdims=True) + 1e-10)
            input_mask = tf.cast(input_mask, tf.float32)
            pooled = masked_reduce_mean(result, input_mask)
        else:
            raise NameError(
                f"Undefined pooling type (must be either first or mean, but is {self.pooling}")

        return pooled

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_size


class BERT:
    # hidden size | hidden layer | model type
    # 128 | 2 | bert-tiny
    # 256 | 4 | bert-mini
    # 512 | 4 | bert-small
    # 512 | 8 | bert-medium
    # 768 | 12 | bert-base
    def __init__(self, max_length, hidden_size=128, num_hidden_layers=2, learn_rate=0.01, optimizer='Adam'):
        self.max_seq_length = max_length
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        if optimizer == 'Adam':
            self.opt = tf.keras.optimizers.Adam(learning_rate=learn_rate)
        elif optimizer == 'SGD':
            self.opt = tf.keras.optimizers.SGD(learning_rate=learn_rate)
        elif optimizer == 'Adagrad':
            self.opt = tf.keras.optimizers.Adagrad(learning_rate=learn_rate)
        elif optimizer == 'Momentum':
            self.opt = tf.keras.optimizers.SGD(
                learning_rate=learn_rate, decay=1e-6, momentum=0.9)
        else:
            raise ValueError('Optimizer is not recognized')

    def build(self):
        in_id = tf.keras.layers.Input(
            shape=(self.max_seq_length,), name="input_ids")
        in_mask = tf.keras.layers.Input(
            shape=(self.max_seq_length,), name="input_masks")
        in_segment = tf.keras.layers.Input(
            shape=(self.max_seq_length,), name="segment_ids")
        bert_inputs = [in_id, in_mask, in_segment]

        bert_output = BertLayer(
            n_fine_tune_layers=self.num_hidden_layers)(bert_inputs)
        dense = tf.keras.layers.Dense(
            self.hidden_size, activation="relu")(bert_output)
        pred = tf.keras.layers.Dense(1, activation="sigmoid")(dense)

        model = tf.keras.models.Model(inputs=bert_inputs, outputs=pred)
        model.compile(loss="binary_crossentropy",
                      optimizer=self.opt, metrics=["accuracy"])
        # model.summary()

        trainable_count = int(np.sum(
            [tf.keras.backend.count_params(p) for p in set(model.trainable_weights)]))
        # non_trainable_count = int(np.sum([K.count_params(p) for p in set(model.non_trainable_weights)]))

        return model, trainable_count
