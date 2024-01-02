import os
import numpy as np
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 设置字体为宋体
matplotlib.rcParams['font.sans-serif'] = ['STSong']

file0 = 'data/all_r.csv'
file1 = 'ddpg_0_all_r.csv'


def read_r(filename):
    df = pd.read_csv(filename, usecols=[0, 2, 3])[['sac','td3','ddpg_1']]
    data = df.values
    return data


def plot_rewards_sns(rewards, tag='train'):
    sns.set(font='STSong')
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.title("在线学习过程全局奖励值变化", fontsize=20)
    plt.xlabel('训练轮次', fontsize=20)
    plt.ylabel('奖励值', fontsize=20)
    plt.plot(rewards, label=['策略梯度', 'DDPG', '改进DDPG','ddpg2'])

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)  # 设置坐标标签字体大小

    plt.legend(fontsize=20)
    plt.savefig("{}_rewards_curve_sns".format(tag))
    plt.show()


def plot_rewards(rewards, tag='train'):
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.title("在线学习过程全局奖励值变化", fontsize=20)
    plt.xlabel('训练轮次', fontsize=20)
    plt.ylabel('奖励值', fontsize=20)
    plt.plot(rewards, label=['策略梯度', 'DDPG', 'improved DDPG'])

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)  # 设置坐标标签字体大小

    plt.grid()
    plt.legend(fontsize=20)
    plt.show()
    plt.savefig("{}_rewards_curve".format(tag))


def plot_rewards_en(rewards, tag='train'):
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    # plt.title("在线学习过程全局奖励值变化", fontsize=20)
    plt.xlabel('epoch', fontsize=20)
    plt.ylabel('reward', fontsize=20)
    plt.plot(rewards, label=['Improved DDPG','DDPG','PG'],linewidth=3.0)

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)  # 设置坐标标签字体大小

    plt.grid()
    plt.legend(fontsize=20)
    plt.show()
    plt.savefig("{}_rewards_curve".format(tag))



def plot_rewards_single(rewards, tag='single'):
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.title("在线学习过程全局奖励值变化", fontsize=20)
    plt.xlabel('训练轮次', fontsize=20)
    plt.ylabel('奖励值', fontsize=20)
    plt.plot(rewards[:, 2], label='奖励值')

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)  # 设置坐标标签字体大小

    plt.grid()
    plt.legend(fontsize=20)

    plt.savefig("{}_rewards_curve".format(tag))
    plt.show()


reward = read_r(file0)
plot_rewards_en(reward)
# plot_rewards(reward)
# plot_rewards_single(reward)
# plot_rewards_sns(reward)
