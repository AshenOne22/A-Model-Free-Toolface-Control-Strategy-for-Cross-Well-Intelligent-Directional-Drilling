import os
import time

import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

# ####################  hyper parameters  ####################
# reward不除以10
ENV_ID = 'Pendulum-v0'  # environment id
RANDOM_SEED = 2  # random seed, can be either an int number or None
RENDER = True  # render while training

ALG_NAME = 'DDPG-t2'
TRAIN_EPISODES = 600  # total number of episodes for training
MAX_STEPS = 200  # total number of steps for each episode

LR_A = 0.001  # learning rate for actor
LR_C = 0.002  # learning rate for critic
GAMMA = 0.9  # reward discount
TAU = 0.01  # soft replacement
MEMORY_CAPACITY = 10000  # size of replay buffer
# MEMORY_CAPACITY = 100  # size of replay buffer
BATCH_SIZE = 32  # update action batch size
VAR = 2  # control exploration

all_episode_reward = []


# ##############################  DDPG  ####################################

class DDPG:
    def __init__(self, action_dim, state_dim, action_range):
        self.memory = np.zeros((MEMORY_CAPACITY, action_dim + state_dim * 2 + 1), dtype=np.float32)
        self.pointer = 0
        self.action_dim, self.state_dim, self.action_range = action_dim, state_dim, action_range
        self.var = VAR

        w_init = tf.random_normal_initializer(mean=0, stddev=0.3)
        b_init = tf.constant_initializer(0.1)

        def get_actor(input_state_shape, name=''):
            """
            Build actor network
            :param input_state_shape: state
            :param name: name
            :return: action
            """
            input_layer = keras.layers.Input(input_state_shape, name='A_input')
            layer = keras.layers.Dense(64, tf.nn.relu, name='A_l1')(input_layer)
            layer = keras.layers.Dense(64, tf.nn.relu, name='A_l2')(layer)
            layer = keras.layers.Dense(action_dim, tf.nn.tanh, name='A_a')(layer)
            layer = keras.layers.Lambda(lambda x: action_range * x)(layer)
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
            layer = keras.layers.Dense(64, tf.nn.relu, name='C_l1')(layer)
            layer = keras.layers.Dense(64, tf.nn.relu, name='C_l2')(layer)
            layer = keras.layers.Dense(1, name='C_out')(layer)
            return keras.models.Model(inputs=[state_input, action_input], outputs=layer, name='Critic' + name)

        self.actor = get_actor(state_dim)
        self.critic = get_critic(state_dim, action_dim)
        self.actor.summary()
        self.critic.summary()

        # self.actor.train()
        # self.critic.train()

        def copy_parameters(from_model, to_model):
            """
            Copy parameters for soft updating
            :param from_model: latest model
            :param to_model: target model
            :return: None
            """
            for i, j in zip(from_model.trainable_weights, to_model.trainable_weights):
                j.assign(i)

        self.actor_target = get_actor(state_dim, name='_target')
        copy_parameters(self.actor, self.actor_target)
        # self.actor_target.eval()

        self.critic_target = get_critic(state_dim, action_dim, name='_target')
        copy_parameters(self.critic, self.critic_target)
        # self.critic_target.eval()

        self.ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)  # soft replacement

        self.actor_opt = tf.optimizers.Adam(learning_rate=LR_A)
        self.critic_opt = tf.optimizers.Adam(learning_rate=LR_C)

    def ema_update(self):
        """
        Soft updating by exponential smoothing
        :return: None
        """
        parameters = self.actor.trainable_weights + self.critic.trainable_weights
        self.ema.apply(parameters)
        for i, j in zip(self.actor_target.trainable_weights + self.critic_target.trainable_weights, parameters):
            i.assign(self.ema.average(j))

    def get_action(self, s, greedy=False):
        """
        Choose action
        :param s: state
        :param greedy: get action greedy or not
        :return: act
        """
        a = self.actor(np.array([s], dtype=np.float32))[0]
        if greedy:
            return a
        return np.clip(np.random.normal(a, self.var), -self.action_range,
                       self.action_range)  # add randomness to action selection for exploration

    def learn(self):
        """
        Update parameters
        :return: None
        """
        self.var *= .9995
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        data = self.memory[indices, :]
        states = data[:, :self.state_dim]
        actions = data[:, self.state_dim:self.state_dim + self.action_dim]
        rewards = data[:, -self.state_dim - 1:-self.state_dim]
        states_ = data[:, -self.state_dim:]

        with tf.GradientTape() as tape:
            actions_ = self.actor_target(states_)
            q_ = self.critic_target([states_, actions_])
            y = rewards + GAMMA * q_
            q = self.critic([states, actions])
            td_error = tf.losses.mean_squared_error(y, q)
        critic_grads = tape.gradient(td_error, self.critic.trainable_weights)
        self.critic_opt.apply_gradients(zip(critic_grads, self.critic.trainable_weights))

        with tf.GradientTape() as tape:
            a = self.actor(states)
            q = self.critic([states, a])
            actor_loss = -tf.reduce_mean(q)  # maximize the q
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_weights)
        self.actor_opt.apply_gradients(zip(actor_grads, self.actor.trainable_weights))
        self.ema_update()
        return actor_loss

    def store_transition(self, s, a, r, s_):
        """
        Store data in data buffer
        :param s: state
        :param a: act
        :param r: reward
        :param s_: next state
        :return: None
        """
        s = s.astype(np.float32)
        s_ = s_.astype(np.float32)
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def save(self):
        """
        save trained weights
        :return: None
        """
        path = os.path.join('model', '_'.join([ALG_NAME, ENV_ID]))
        if not os.path.exists(path):
            os.makedirs(path)
        self.actor.save(os.path.join(path, 'actor.h5'))
        self.actor_target.save(os.path.join(path, 'actor_target.h5'))
        self.critic.save(os.path.join(path, 'critic.h5'))
        self.critic_target.save(os.path.join(path, 'actor_critic.h5'))

    def load(self):
        """
        load trained weights
        :return: None
        """
        path = os.path.join('model', '_'.join([ALG_NAME, ENV_ID]))
        self.actor.load_weights(os.path.join(path, 'actor.h5'))
        self.actor_target.load_weights(os.path.join(path, 'actor_target.h5'))
        self.critic.load_weights(os.path.join(path, 'critic.h5'))
        self.critic_target.load_weights(os.path.join(path, 'actor_critic.h5'))


class Record:
    def __init__(self):
        self.alist = []
        self.anp = None
        self.blist = []
        self.bnp = None
        path = os.path.join('record', '_'.join([ALG_NAME, ENV_ID, 't2']))
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

    def add(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        self.alist.append(transition)

    def save(self, ep):
        filename = str(ep) + '.csv'
        filepath = os.path.join(self.path, filename)

        self.anp = np.array(self.alist)
        name_l = ['state0', 'state1', 'state2', 'action', 'reward', 'state_0', 'state_1', 'state_2']
        df = pd.DataFrame(self.anp, columns=name_l)
        df.to_csv(filepath)

    def add_l(self, s, a, r, s_, a_l):
        transition = np.hstack((s, a, [r], s_, [a_l]))
        self.blist.append(transition)

    def save_loss(self, ep):
        filename = 'loss_' + str(ep) + '.csv'
        filepath = os.path.join(self.path, filename)

        self.bnp = np.array(self.blist)
        name_l = ['state0', 'state1', 'state2', 'action', 'reward', 'state_0', 'state_1', 'state_2', 'a_loss']
        df = pd.DataFrame(self.bnp, columns=name_l)
        df.to_csv(filepath)

    def save_r(self, all_r):
        rnp = np.array(all_r)
        df = pd.DataFrame(rnp)
        df.to_csv(os.path.join(self.path, '0_all_r.csv'))

    def save_r_2(self, all_r):
        rnp = np.array(all_r)
        df = pd.DataFrame(rnp)
        df.to_csv(os.path.join(self.path, '0_all_r_2.csv'))

    def save_state0(self, all_state0):
        state0_np = np.array(all_state0)
        df = pd.DataFrame(state0_np)
        df.to_csv(os.path.join(self.path, 'all_state0.csv'))


if __name__ == '__main__':
    env = gym.make(ENV_ID).unwrapped

    # reproducible
    env.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_range = env.action_space.high  # scale action, [-action_range, action_range]

    agent = DDPG(action_dim, state_dim, action_range)
    all_r_2 = []
    all_state0 = []

    t0 = time.time()
    for episode in range(TRAIN_EPISODES):
        state = env.reset()
        episode_reward = 0
        record = Record()
        all_state0.append(state)

        for step in range(MAX_STEPS):
            if RENDER:
                env.render()

            # Add exploration noise
            action = agent.get_action(state)
            state_, reward, done, _ = env.step(action)
            # reward /= 10  # normalize reward
            agent.store_transition(state, action, reward, state_)
            record.add(state, action, reward, state_)

            if agent.pointer > MEMORY_CAPACITY:
                a_loss = agent.learn()
                record.add_l(state, action, reward, state_, a_loss)

            state = state_
            episode_reward += reward
            if done:
                break

        record.save(episode)
        all_r_2.append(episode_reward)
        if record.blist:
            record.save_loss(episode)
        if episode == 0:
            all_episode_reward.append(episode_reward)
        else:
            all_episode_reward.append(all_episode_reward[-1] * 0.99 + episode_reward * 0.01)
        print('Episode: {}/{}  |  Episode Reward: {:.4f}  |  Running Time: {:.4f}'.format(
            episode + 1, TRAIN_EPISODES, episode_reward, time.time() - t0
        ))

    plt.plot(np.arange(len(all_episode_reward)), all_episode_reward)
    if not os.path.exists('image'):
        os.makedirs('image')
    plt.savefig(os.path.join('image', '_'.join([ALG_NAME, ENV_ID])))
    record.save_r(all_episode_reward)
    record.save_r_2(all_r_2)
    record.save_state0(all_state0)
    env.close()
    agent.save()
