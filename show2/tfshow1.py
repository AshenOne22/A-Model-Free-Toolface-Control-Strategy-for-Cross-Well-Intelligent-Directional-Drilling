import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 设置字体为宋体
matplotlib.rcParams['font.sans-serif'] = ['STSong']

file0 = 'all_r.csv'
file1 = 'data/0118tf.csv'
file_l = 'learn-ex1.csv'


def read_r(filename):
    df = pd.read_csv(filename)
    data = df.values
    return data


def read_tf(filename):
    df = pd.read_csv(filename)
    data = df.values
    data = data / 6 * 180
    return data


def read_dist(filename):
    df = pd.read_csv(filename)
    data = df.values
    tf = data[:, 0]
    tar = data[0, 1]
    dist = tf - tar
    for i in range(len(tf)):
        if dist[i] > 180:
            dist[i] = dist[i] - 360
        elif dist[i] < -180:
            dist[i] += 360
    return dist,data


def plot_dist(dist, tag='explore'):
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.title("现场试验工具面控制效果", fontsize=25)
    plt.xlabel('调整时间步', fontsize=25)
    plt.ylabel('角度(°)', fontsize=25)
    plt.plot(dist, label='工具面距离目标角度')

    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)  # 设置坐标标签字体大小

    plt.grid()
    plt.legend(fontsize=25)

    plt.savefig("image/{}_dist_curve".format(tag))
    plt.show()

def plot_tf2(data, tag='explore'):
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.title("现场试验工具面控制效果", fontsize=25)
    plt.xlabel('调整时间步', fontsize=25)
    plt.ylabel('角度(°)', fontsize=25)
    plt.plot(data[:,0], label='工具面状态')
    plt.plot(data[:, 1], label='目标工具面')

    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)  # 设置坐标标签字体大小

    plt.grid()
    plt.legend(fontsize=25)

    plt.savefig("image/{}_tf_curve".format(tag))
    plt.show()


def plot_tf2_en(data, tag='en'):
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.title("Field experimental results", fontsize=25)
    plt.xlabel('steps', fontsize=25)
    plt.ylabel('toolface orientation(°)', fontsize=25)
    plt.plot(data[:, 0], label='toolface', linewidth=3.0)
    plt.plot(data[:, 1], label='target toolface', linewidth=3.0)

    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)  # 设置坐标标签字体大小

    plt.grid()
    plt.legend(fontsize=25)

    plt.savefig("image/{}_tf_curve".format(tag))
    plt.show()


def plot_tf(tf, tag='explore'):
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.title("收敛阶段-工具面控制效果", fontsize=25)
    plt.xlabel('调整时间步', fontsize=25)
    plt.ylabel('角度(°)', fontsize=25)
    plt.plot(tf[20:125], label='工具面状态')

    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)  # 设置坐标标签字体大小

    plt.grid()
    plt.legend(fontsize=25)

    plt.savefig("image/{}_tf_curve".format(tag))
    plt.show()


def plot_rewards_single(rewards, tag='single'):
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.title("测试-工具面变化", fontsize=25)
    plt.xlabel('调整时间步', fontsize=25)
    plt.ylabel('工具面角度', fontsize=25)
    plt.plot(rewards, label='工具面状态')

    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)  # 设置坐标标签字体大小

    plt.grid()
    plt.legend(fontsize=25)

    plt.savefig("image/{}_tf_curve".format(tag))
    plt.show()


dist,data = read_dist(file1)
plot_dist(dist, '0118')
plot_tf2(data,'0118')
plot_tf2_en(data)
