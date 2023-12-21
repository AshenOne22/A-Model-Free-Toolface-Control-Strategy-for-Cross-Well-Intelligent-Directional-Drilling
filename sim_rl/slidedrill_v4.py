"""
以真实数据训练的LSTM模型作为被控对象提供反馈
时间步为look_back=2
输入dif，torp，torn，输出dif
"""

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
from tensorflow import keras


class SlideDrillEnv4(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, g=10.0):
        self.max_speed = 8
        self.max_torque = 2.
        self.dt = .05
        self.g = g
        self.m = 1.
        self.l = 1.
        self.viewer = None
        self.model = keras.models.load_model('../truedata_lstm/model/timestep_2_outdim_1_standardization_t0/lstm1.h5')
        self.observation_space_shape = 3
        self.action_space_shape = 2
        self.action_space_high = 1
        self.look_back=2
        high = np.array([1., 1., self.max_speed], dtype=np.float32)
        self.action_space = spaces.Box(
            low=-self.max_torque,
            high=self.max_torque, shape=(1,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-high,
            high=high,
            dtype=np.float32
        )

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        # th, thdot = self.state  # th := theta
        #
        # g = self.g
        # m = self.m
        # l = self.l
        # dt = self.dt
        #
        # u = np.clip(u, -self.max_torque, self.max_torque)[0]
        # self.last_u = u  # for rendering
        # costs = angle_normalize(th) ** 2 + .1 * thdot ** 2 + .001 * (u ** 2)
        #
        # newthdot = thdot + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3. / (m * l ** 2) * u) * dt
        # newth = th + newthdot * dt
        # newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        #
        # self.state = np.array([newth, newthdot])
        # 计算reward
        dif=self.state[-1,0]
        old_dif=self.state[-2,0]
        d_dif=dif-old_dif
        new_tor_p,new_tor_n=u
        costs = (dif*np.pi) ** 2 +.1 * (d_dif*np.pi) ** 2+ .001 * (((new_tor_p*2) ** 2)+((new_tor_n*2) ** 2))
        # 采用LSTM模型产生下一状态
        state0=np.array([dif,new_tor_p,new_tor_n])
        temp0_in = np.concatenate((self.state[-1,:],state0),axis=0)
        temp0_in = temp0_in.reshape([1, self.look_back, 3])
        temp_state = self.model.predict(temp0_in)
        temp_dif = temp_state[0, 0]
        state1=np.array([temp_dif,new_tor_p,new_tor_n])
        self.state = np.concatenate((self.state[-1,:],state1),axis=0)
        self.state=self.state.reshape(self.look_back,self.observation_space_shape)
        # s_o=self.state[-1,:]
        return self.state, -costs, False, {}

    def get_reward(self):
        # r1为角度偏差平方，r2为速度平方，r3为动作平方
        s=self.state
        s=s.reshape(5,3)
        alpha = 0.5
        k1, k2, k3, k4 = 1, 0.1, 0.01, 0.01

        dist = s[:, 4::-1] - s[0, 5]
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

        rew = (abs(dp) / self.nor_p + 1) * r
        rew = -rew
        return rew


    def reset(self):
        high = np.ones((self.look_back,self.observation_space_shape))
        self.state = self.np_random.uniform(low=-high, high=high)
        # state0=self.state.reshape(1,3)
        self.last_u = None
        # s_o=self.state[-1,:]
        return self.state

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot])

    def render(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi / 2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u / 2, np.abs(self.last_u) / 2)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


def angle_normalize(x):
    return (((x + np.pi) % (2 * np.pi)) - np.pi)
