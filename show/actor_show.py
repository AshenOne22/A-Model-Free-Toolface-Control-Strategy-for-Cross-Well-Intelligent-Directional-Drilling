import tensorflow as tf
from tensorflow import keras

action_dim = 6
action_range = 2


def get_actor(input_state_shape, name=''):
    """
    Build actor network
    :param input_state_shape: state
    :param name: name
    :return: action
    """
    input_layer = keras.layers.Input(input_state_shape, name='A_input')
    layer = keras.layers.Dense(128, tf.nn.relu, name='A_l1')(input_layer)
    layer = keras.layers.Dense(128, tf.nn.relu, name='A_l2')(layer)
    layer = keras.layers.Dense(64, tf.nn.relu, name='A_l3')(layer)
    layer = keras.layers.Dense(64, tf.nn.relu, name='A_l4')(layer)
    layer = keras.layers.Dense(action_dim, tf.nn.tanh, name='A_out')(layer)
    # layer = keras.layers.Lambda(lambda x: action_range * x)(layer)
    return keras.models.Model(inputs=input_layer, outputs=layer, name='Actor' + name)


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
    layer = keras.layers.Dense(128, tf.nn.relu, name='C_l1')(layer)
    layer = keras.layers.Dense(128, tf.nn.relu, name='C_l2')(layer)
    layer = keras.layers.Dense(64, tf.nn.relu, name='C_l3')(layer)
    layer = keras.layers.Dense(64, tf.nn.relu, name='C_l4')(layer)
    layer = keras.layers.Dense(1, name='C_out')(layer)
    return keras.models.Model(inputs=[state_input, action_input], outputs=layer, name='Critic' + name)


modela = get_actor(12)
modela.summary()
modelc = get_critic(12, 6)
modelc.summary()
