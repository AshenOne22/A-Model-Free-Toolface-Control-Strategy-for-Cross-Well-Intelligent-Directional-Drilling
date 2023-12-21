import numpy as np
import tensorflow as tf
from tensorflow import keras


def get_critic(input_state_shape, input_action_shape, name=''):
    """
    Build critic network
    :param input_state_shape: state
    :param input_action_shape: action
    :param name: name
    :return: Q value Q(s,a)
    """
    state_input = keras.layers.Input(input_state_shape, name='C_s_input')
    action_input = keras.layers.Input(input_action_shape, name='C_a_input')
    layer = keras.layers.concatenate([state_input, action_input], axis=-1)
    layer = keras.layers.Dense(64, tf.nn.relu, name='C_l1')(layer)
    layer = keras.layers.Dense(64, tf.nn.relu, name='C_l2')(layer)
    layer = keras.layers.Dense(1, name='C_out')(layer)
    return keras.models.Model(inputs=[state_input, action_input], outputs=layer, name='Critic' + name)


critic=get_critic([None, 3], [None, 1])
sl=np.arange(3*3)
snp=sl.reshape(3,3)
print(snp.shape)
al=np.arange(3*1)
anp=al.reshape(3,1)
r=critic(snp,anp)
print(r)

