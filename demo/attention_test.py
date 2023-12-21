import tensorflow as tf
from tensorflow.keras.layers import *
import numpy as np


class Attention(tf.keras.layers.Layer):

    def __init__(self, query_dim=None, key_dim=None, value_dim=None, output_dim=None, num_of_head=None):
        super(Attention, self).__init__()

        self.query_dim = query_dim // num_of_head
        self.key_dim = key_dim // num_of_head
        self.value_dim = value_dim // num_of_head
        self.output_dim = output_dim
        self.num_of_head = num_of_head

        self.wq = [Dense(self.query_dim) for h in range(self.num_of_head)]
        self.wk = [Dense(self.key_dim) for h in range(self.num_of_head)]
        self.wv = [Dense(self.value_dim) for h in range(self.num_of_head)]
        self.wo = Dense(self.output_dim)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'query_dim': self.query_dim,
            'key_dim': self.key_dim,
            'value_dim': self.value_dim,
            'output_dim': self.output_dim,
            'num_of_head': self.num_of_head,
            'wq': self.wq,
            'wk': self.wk,
            'wv': self.wv,
            'wo': self.wo
        })
        return config

    def call(self, x):
        heads_context = []

        for h in range(self.num_of_head):
            q = self.wq[h](x)
            k = self.wk[h](x)
            v = self.wv[h](x)

            pre_score = tf.matmul(q, k, transpose_b=True)
            # score = tf.keras.activations.softmax(pre_score, axis=1) / tf.math.sqrt(
            #     tf.dtypes.cast(self.key_dim, tf.float32))
            score = tf.keras.activations.softmax(pre_score / tf.math.sqrt(
                tf.dtypes.cast(self.key_dim, tf.float32)), axis=-1)

            Z_context_vector = tf.matmul(score, v)

            heads_context.append(Z_context_vector)

        heads = tf.concat(heads_context, axis=-1)
        # heads = tf.concat(heads_context)
        heads = self.wo(heads)

        return heads


# input :-> (batch size,time_steps,embeddings_dims)
# output:-> (batch_size,time_steps,output_dims)

# x = tf.ones((12, 16, 512))
#
x = tf.ones((8))

f = Attention(query_dim=4, key_dim=4, value_dim=4, output_dim=2, num_of_head=2)(x)

print(f.shape)

# z = f(x)
#
# print(z.shape)
