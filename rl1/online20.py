import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.utils import np_utils

# ####################  hyper parameters  ####################
TRAIN_EPISODES = 1000  # total number of episodes for training
MAX_STEPS = 500  # total number of steps for each episode
labels = 10  # action标签为10类
LR_A = 0.001  # learning rate for actor
LR_C = 0.002  # learning rate for critic
GAMMA = 0.9  # reward discount
TAU = 0.01  # soft replacement
MEMORY_CAPACITY = 10000  # size of replay buffer
BATCH_SIZE = 64  # update action batch size
sim_size = 6
true_size = 58
VAR = 1  # control exploration

all_episode_reward = []


# ##############################  DDPG  ####################################

class DDPG:
    def __init__(self, action_dim, state_dim):
        self.memory = np.zeros((MEMORY_CAPACITY, action_dim + state_dim * 2 + 1 + 1), dtype=np.float32)
        self.pointer = 0
        self.record_id = 0
        self.action_dim, self.state_dim = action_dim, state_dim
        self.var = VAR
        self.record = None

        def create_simdata(csv_file):
            dataset1 = pd.read_csv(csv_file,
                                   usecols=['TF5', 'TF4', 'TF3', 'TF2', 'TF1', 'TAR'], header=0)
            target1 = pd.read_csv(csv_file,
                                  usecols=['Label'], header=0)
            values = dataset1.values
            values = values.astype('float32')
            target1 = target1.values

            values = values / 360

            target_y = np_utils.to_categorical(target1, num_classes=labels)

            return values, target_y

        def get_actor(n):
            """
            Build actor network
            :param input_state_shape: state
            :param n: Hidden layers
            :return: action
            """
            model = keras.Sequential()
            model.add(layers.Input(6))
            for i in range(n):
                model.add(layers.Dense(256, activation='relu'))
            model.add(layers.Dense(10, activation='softmax'))
            model.compile(optimizer='adam',
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
            # model.summary()
            return model

        def get_critic():
            state_input = layers.Input(shape=6)
            s1 = layers.Dense(256, activation='relu')(state_input)
            s2 = layers.Dense(256, activation='relu')(s1)
            action_input = layers.Input(shape=10)
            a1 = layers.Dense(256, activation='relu')(action_input)
            c1 = layers.concatenate([s2, a1], axis=-1)
            c2 = layers.Dense(256, activation='relu')(c1)
            output = layers.Dense(1, activation='linear')(c2)
            model = keras.Model(inputs=[state_input, action_input], outputs=output, name='critic')
            return model

        self.actor = get_actor(3)
        self.actor.load_weights(r'F:\A-4\model_ori\actor_3_256.h5')
        self.critic = get_critic()

        def copy_parameters(from_model, to_model):
            """
            Copy parameters for soft updating
            :param from_model: latest model
            :param to_model: target model
            :return: None
            """
            for i, j in zip(from_model.trainable_weights, to_model.trainable_weights):
                j.assign(i)

        self.actor_target = get_actor(3)
        copy_parameters(self.actor, self.actor_target)

        self.critic_target = get_critic()
        copy_parameters(self.critic, self.critic_target)

        self.ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)  # soft replacement

        self.actor_opt = tf.optimizers.Adam()
        self.critic_opt = tf.optimizers.Adam()

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
        a = self.actor(np.array(s, dtype=np.float32))
        if greedy:
            return a
        return np.argmax(a), a  # add randomness to action selection for exploration

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
        rewards = data[:, self.state_dim + self.action_dim:self.state_dim + self.action_dim + 1]
        states_ = data[:, self.state_dim + self.action_dim + 1:-1]

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

        critic_loss = np.linalg.norm(td_error, ord=2)

        return critic_loss, actor_loss

    def store_transition(self, s, a_logit, r, s_, a_label):
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
        transition1 = np.hstack(s)
        # transition1 = np.hstack(transition1)
        transition2 = np.hstack(s_)
        # transition2 = np.hstack(transition2)
        a_logit = np.hstack(a_logit)
        transition = np.hstack((transition1, a_logit, [r], transition2, [a_label]))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def store_record(self, s, s_, q, r, p, n, a_label, episode):
        """
        Store data in data buffer
        :param s: state
        :param a: act
        :param r: reward
        :param s_: next state
        :return: None
        """
        transition1 = np.hstack(s)
        transition = np.hstack((transition1, [s_], q[0], [r], [p], [n], [a_label], [episode]))
        transition = transition.reshape(1, 13)
        if self.record_id == 0:
            self.record = transition
        else:
            self.record = np.concatenate((self.record, transition))
        self.record_id += 1

    def save(self):
        """
        save trained weights
        :return: None
        """
        path = os.path.join('model_with_gbdt')
        if not os.path.exists(path):
            os.makedirs(path)
        self.actor.save(os.path.join(path, 'actor.h5'))
        self.critic.save(os.path.join(path, 'critic.h5'))
        self.actor_target.save(os.path.join(path, 'target_actor.h5'))
        self.critic_target.save(os.path.join(path, 'target_critic.h5'))

    def load(self):
        """
        load trained weights
        :return: None
        """
        path = os.path.join('model_with_gbdt')
        self.actor = keras.models.load_model(os.path.join(path, 'actor.h5'))
        self.critic = keras.models.load_model(os.path.join(path, 'critic.h5'))
        self.actor_target = keras.models.load_model(os.path.join(path, 'target_actor.h5'))
        self.critic_target = keras.models.load_model(os.path.join(path, 'target_critic.h5'))
        print('load finished')


class Env():
    def __init__(self):
        self.state = None
        # self.initial_p = None
        # self.initial_n = None
        self.initial_p = 10000
        self.initial_n = -5000
        self.nor_p = self.initial_p
        self.nor_n = self.initial_n
        self.true_p = self.initial_p
        self.true_n = self.initial_n
        self.last_p = self.initial_p
        self.last_n = self.initial_n

    def get_state_new(self, action):
        d_s_1 = (self.state[0, 4] - self.state[0, 3]) * 360
        if action == 0:
            d_s = np.random.normal(d_s_1, 0.1)
        elif action == 1:
            d_s = np.random.normal(d_s_1 - 5, 0.1)
        elif action == 2:
            d_s = np.random.normal(d_s_1 + 5, 0.1)
        elif action == 3:
            d_s = np.random.normal(d_s_1 + 3, 0.1)
        elif action == 4:
            d_s = np.random.normal(d_s_1 - 2, 0.1)
        elif action == 5:
            d_s = np.random.normal(d_s_1 + 1, 0.1)
        elif action == 6:
            d_s = np.random.normal(d_s_1 + 5, 0.1)
        elif action == 7:
            d_s = np.random.normal(d_s_1 - 3, 0.1)
        elif action == 8:
            d_s = np.random.normal(d_s_1 + 1, 0.1)
        elif action == 9:
            d_s = np.random.normal(0, 0.1)
        else:
            d_s = np.random.normal(0, 1)
        state_new = self.state[0, 4] * 360 + d_s
        return state_new

    def reset(self):
        csv_file = r'F:\A-1\com_data\true_data\label\label_0_end_all_2.csv'
        dataset1 = pd.read_csv(csv_file,
                               usecols=['TF5', 'TF4', 'TF3', 'TF2', 'TF1', 'TAR', 'TOR_P', 'TOR_N'], header=0)
        values = dataset1.values
        values = values.astype('float32')
        length = len(values)
        idx = np.random.choice(np.arange(length), 1)
        data = values[idx]
        self.state = data[0, 0:6]
        self.state = np.array(self.state).reshape((1, 6)) / 360
        self.nor_p, self.nor_n = data[0, 6], data[0, 7]
        self.true_p, self.true_n = data[0, 6], data[0, 7]
        return self.state

    def get_pn(self, action):
        if action == 0:
            n = self.nor_n
            p = self.nor_p
        elif action == 1:
            p = self.nor_p - 1000
            n = self.nor_n - 500
        elif action == 2:
            p = self.nor_p + 1000
            n = self.nor_n
        elif action == 3:
            p = self.nor_p + 500
            n = self.nor_n
        elif action == 4:
            p = self.nor_p - 200
            n = self.nor_n - 200
        elif action == 5:
            p = self.nor_p + 200
            n = self.nor_n
        elif action == 6:
            p = self.nor_p + 500
            n = self.nor_n + 500
        elif action == 7:
            p = self.nor_p - 500
            n = self.nor_n - 500
        elif action == 8:
            p = self.nor_p + 200
            n = self.nor_n + 200
        elif action == 9:
            dist = self.state[0, -2] - self.state[0, -1]
            if dist > 180:
                dist = dist - 360
            elif dist < 180:
                dist = dist + 360
            if dist >= 0:
                p = self.nor_p + 1000
                n = self.nor_n
            else:
                p = self.nor_p - 1000
                n = self.nor_n - 500
        else:
            p = self.initial_p
            n = self.initial_n
        self.nor_p = p
        self.nor_n = n
        return p, n

    def get_reward(self, s):
        # r1为角度偏差平方，r2为速度平方，r3为动作平方
        alpha = 0.5
        k1, k2, k3, k4 = 1, 0.1, 0.01, 1

        dist = s[0, 4::-1] - s[0, 5]
        v = s[0, 4:0:-1] - s[0, 3::-1]

        for i in range(len(dist)):
            if dist[i] > 0.5:
                dist[i] = dist[i] - 1
            elif dist[i] < -0.5:
                dist[i] = dist[i] + 1
        for i in range(len(v)):
            if v[i] > 0.5:
                v[i] = v[i] - 1
            elif v[i] < -0.5:
                v[i] = v[i] + 1

        dist = dist * 2 * np.math.pi
        dist = np.multiply(dist, dist)
        v = v * 2 * np.math.pi
        v = np.multiply(v, v)
        k_alpha1 = np.ones(5) * alpha
        k_alpha2 = np.ones(4) * alpha
        k_1 = k1 * np.power(k_alpha1, np.arange(5))
        k_2 = k2 * np.power(k_alpha2, np.arange(4))
        r1 = np.vdot(k_1, dist)
        r2 = np.vdot(k_2, v)

        dp = self.nor_p - self.last_p
        dn = self.nor_n - self.last_n
        r3 = k3 * ((dp / 1000) ** 2 + (dn / 1000) ** 2)

        r = r1 + r2 + r3

        dist_p = abs(self.nor_p - self.true_p)
        dist_n = abs(self.nor_n - self.true_n)
        rew = dist_p + dist_n + r
        rew = -rew
        return rew

    def step(self, action, tf1, tar):
        temp_p, temp_n = self.get_pn(action)
        self.judge_pn(temp_p, temp_n)
        gbdt_input = np.hstack((self.state[0, -1] * 360, self.state[0, :5] * 360, self.nor_p, self.nor_n))
        gbdt_input = gbdt_input.reshape((1, 8))
        # state_new = new_gbdt.predict(gbdt_input)
        # state_new = state_new[0]
        state_new = self.get_state_new(action)
        state_new = state_new / 360
        if state_new >= 1:
            state_new = state_new - 1
        elif state_new < 0:
            state_new = state_new + 1
        state_ = np.hstack((self.state[0, 1:5], state_new, self.state[0, -1]))
        state_ = state_.reshape((1, 6))
        self.state = state_

        tf1 = np.array(tf1)
        tar = np.ones(len(tf1)) * tar
        reward = self.get_reward(state_)
        done = False
        self.last_p, self.last_n = self.nor_p, self.nor_n
        return state_, reward, done, state_new

    def judge_pn(self, p_p, p_n):
        p_p_2, p_n_2 = p_p, p_n
        if p_p < 3000 or p_p > 16000:
            p_p_2 = self.initial_p
        if p_n < -10000 or p_n > -2000:
            p_n_2 = self.initial_n
        if p_p <= -p_n:
            p_p_2 = self.initial_p
            p_n_2 = self.initial_n
        self.nor_p, self.nor_n = p_p_2, p_n_2
        self.true_p, self.true_n = p_p_2, p_n_2
        return p_p_2, p_n_2


def main():
    env = Env()
    state_dim = 6
    action_dim = 10
    agent = DDPG(action_dim, state_dim)
    load = True
    if load:
        agent.load()

    # tcpSerSock = socket(AF_INET, SOCK_STREAM)  # scoket实例化
    # port = 4002
    # ip_post = ("127.0.0.1", int(port))
    # tcpSerSock.bind(ip_post)
    # tcpSerSock.listen(5)
    # tcpCliSock, addr = tcpSerSock.accept()

    t0 = time.time()

    all_q = []
    all_reward = []
    all_actor_loss = []
    all_critic_loss = []

    for episode in range(200):
        t1 = time.time()
        ep_r = []
        ep_q = []
        episode_reward = 0
        num = 0
        state = env.reset()
        all_tf1 = []
        all_tar = state[0, 5]
        true_num = 0
        for step in range(MAX_STEPS):
            # while True:

            all_tf1.append(state[0, 4])
            t2 = time.time()
            action, a_logit = agent.get_action(state)
            state_, reward, done, state_new = env.step(action, all_tf1, all_tar)
            # print(str(step)+','+str(action) + ',' + str(env.nor_p) + ',' + str(env.nor_n)+','+str((state[0,4]-state[0,5])*360))
            # # 显示工具面状态变化趋势
            # # if len(all_reward) < 10:
            # plt.ion()
            # plt.plot(np.arange(5), state[0, :5] * 360, 'b', label='sgtf')
            # plt.plot(np.arange(5), 360 * state[0, 5] * np.ones(5), 'g', label='tar')
            # plt.ylim(0, 360)
            # plt.legend()
            # plt.pause(2)  # 显示秒数
            # plt.close()

            agent.store_transition(state, a_logit, reward, state_, action)
            if agent.pointer > BATCH_SIZE:
                # print('agent.pointer:',agent.pointer)
                critic_loss, actor_loss = agent.learn()
                all_actor_loss.append(actor_loss)
                all_critic_loss.append(critic_loss)
                agent.save()
            q = agent.critic([state, a_logit])
            all_q.append(q[0])
            all_reward.append(reward)
            ep_r.append(reward)
            ep_q.append(q[0])
            agent.store_record(s=state * 360, s_=state_new * 360, q=q, r=reward, p=env.nor_p, n=env.nor_n,
                               a_label=action, episode=episode)
            state = state_
            episode_reward += reward
            num += 1
            if abs((state_[0, 4] - state_[0, 5]) * 360) > 10 and true_num != 0:
                true_num = 0
            if abs((state_[0, 4] - state_[0, 5]) * 360) <= 10:
                true_num += 1
            if true_num > 30:
                done = True
            if done:
                print('finished,step:', step)
                break
        # break
        # if len(all_q) % 100 == 0:
        #     if len(all_q) < 1000:
        #         plt.plot(np.arange(len(agent.record)), agent.record[..., 4], label='sgtf')
        #         plt.plot(np.arange(len(agent.record)), agent.record[..., 5], label='tar')
        #         plt.legend()
        #         plt.savefig('image/sgtf1')
        #         plt.close()
        #     else:
        #         plt.plot(np.arange(len(agent.record)), agent.record[..., 4], label='sgtf')
        #         plt.plot(np.arange(len(agent.record)), agent.record[..., 5], label='tar')
        #         plt.legend()
        #         plt.savefig('image/sgtf2')
        #         plt.close()
        #     print('step:{} | running time all: {:.4f} | running time : {:.4f}'.format(num, time.time() - t0,
        #                                                                               time.time() - t2))
        #
        # if len(all_q) % 100 ==
        # 0:
        #     df = pd.DataFrame(agent.record,
        #                       columns=['TF5', 'TF4', 'TF3', 'TF2', 'TF1', 'TAR', 's_', 'q', 'r', 'p', 'n',
        #                                'a_label'])
        #     if len(all_q) == 10000:
        #         df.to_csv('image/memory1.csv', index=False)
        #     elif len(all_q) == 15000:
        #         df.to_csv('image/memory2.csv', index=False)
        #     elif len(all_q) == 20000:
        #         df.to_csv('image/memory3.csv', index=False)
        #     else:
        #         df.to_csv('image/memory_all.csv', index=False)

        df = pd.DataFrame(agent.record, columns=['TF5', 'TF4', 'TF3', 'TF2', 'TF1', 'TAR', 's_', 'q', 'r', 'p', 'n',
                                                 'a_label', 'episode'])
        if len(all_q) == 10000:
            df.to_csv('image/memory1.csv', index=False)
        elif len(all_q) == 15000:
            df.to_csv('image/memory2.csv', index=False)
        elif len(all_q) == 20000:
            df.to_csv('image/memory3.csv', index=False)
        else:
            df.to_csv('image/memory_all.csv', index=False)
        #
        # # 保存loss数据
        # data_a_loss = np.array(all_actor_loss).reshape(len(all_actor_loss), 1)
        # data_c_loss = np.array(all_critic_loss).reshape(len(all_critic_loss), 1)
        # data_loss = np.hstack((data_a_loss, data_c_loss))
        # df_loss = pd.DataFrame(data_loss, columns=['actor_loss', 'critic_loss'])
        # df_loss.to_csv('image/loss1.csv', index=False)
        # # 画loss图像
        # plt.plot(np.arange(len(all_actor_loss)), all_actor_loss, 'b', label='actor loss')
        # plt.plot(np.arange(len(all_critic_loss)), all_critic_loss, 'g', label='critic loss')
        # plt.title('loss')
        # plt.legend()
        # if len(all_actor_loss) == 200:
        #     plt.savefig('image/loss1.jpg')
        # if len(all_actor_loss) == 400:
        #     plt.savefig('image/loss2.jpg')
        # if len(all_actor_loss) == 600:
        #     plt.savefig('image/loss3.jpg')
        # else:
        #     plt.savefig('image/loss4.jpg')
        # plt.close()
        # # loss后500步图像
        # plt.plot(np.arange(400), all_actor_loss[-400:], 'b', label='actor loss')
        # plt.plot(np.arange(400), all_critic_loss[-400:], 'g', label='critic loss')
        # plt.title('loss')
        # plt.legend()
        # plt.savefig('image/loss_last.jpg')
        # plt.close()
        # # r，q图像
        # plt.plot(np.arange(len(all_q)), all_q, 'b', label='q value')
        # plt.plot(np.arange(len(all_reward)), all_reward, 'g', label='reward')
        # plt.title('q value and reward')
        # plt.legend()
        # if len(all_q) == 2000:
        #     plt.savefig('image/qvalue.jpg')
        # if len(all_q) == 4000:
        #     plt.savefig('image/qvalue2.jpg')
        # if len(all_q) == 6000:
        #     plt.savefig('image/qvalue3.jpg')
        # else:
        #     plt.savefig('image/qvalue4.jpg')
        # plt.close()
        # # r，q后500步图像
        # plt.plot(np.arange(500), all_q[-500:], 'b', label='q value')
        # plt.plot(np.arange(500), all_reward[-500:], 'g', label='reward')
        # plt.title('q value and reward')
        # plt.legend()
        # plt.savefig('image/qvalue_last.jpg')
        # plt.close()

        if episode == 0:
            all_episode_reward.append(episode_reward)
        else:
            # all_episode_reward.append(all_episode_reward[-1] * 0.99 + episode_reward * 0.01)
            all_episode_reward.append(episode_reward)
        print(
            'Episode: {}/{}  |  Episode Reward: {:.4f}  |  Running Time Total: {:.4f} |Running Time Episode: {:.4f} | '
            'num{:}'.format(
                episode + 1, TRAIN_EPISODES, episode_reward, time.time() - t0, time.time() - t1, num
            ))
        df2 = pd.DataFrame()
        df2['ep_rew'] = all_episode_reward
        df2.to_csv('image/ep_rew.csv', index=False)
        plt.plot(np.arange(len(all_episode_reward)), all_episode_reward, label='all_episode_reward')
        plt.legend()
        # if not os.path.exists('image'):
        #     os.makedirs('image')
        plt.savefig(os.path.join('image', '_'.join('test1')))
        plt.close()
        plt.plot(np.arange(len(ep_r)), ep_r, 'g', label='episode_reward')
        plt.plot(np.arange(len(ep_q)), ep_q, 'b', label='episode_q')
        plt.legend()
        plt.savefig(os.path.join('image/episode', '_'.join(['episode_rq', str(episode)])))
        plt.close()
        # plt.ion()
        plt.plot(np.arange(len(all_tf1)), np.array(all_tf1) * 360, 'g', label='sgtf')
        plt.plot(np.arange(len(all_tf1)), all_tar * 360 * np.ones((len(all_tf1), 1)), 'b', label='tar')
        plt.legend()
        plt.savefig(os.path.join('image/episode', '_'.join(['sgtf', str(episode)])))
        # plt.pause(5)
        plt.close()

    # save = True
    # if save:
    #     agent.save()


if __name__ == '__main__':
    # state:'TF5', 'TF4', 'TF3', 'TF2', 'TF1', 'TAR'
    # new_gbdt = joblib.load(r'F:\A-4\model_ori\gbdt-d.m')
    # tf5,tr4,tf3,tf2,tf1,tar ,p,n
    main()
