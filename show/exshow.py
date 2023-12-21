import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# 设置字体为宋体
matplotlib.rcParams['font.sans-serif'] = ['STSong']

file0 = 'all_r.csv'
# file1 = 'data/0110.csv'
file_l = 'learn-ex1.csv'
file_tor = 'data/0118tor.csv'


def read_r(filename):
    df = pd.read_csv(filename)
    data = df.values
    return data


def read_tf(filename):
    df = pd.read_csv(filename)
    data = df.values
    data = data / 6 * 180
    return data


def read_tor(filename, tag='0118'):
    df = pd.read_csv(filename, usecols=[2, 3, 4, 5])
    data = df.values
    print(data.shape)
    num = len(data)
    m_p, m_n, eng_p, eng_n = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
    exe_p, exe_n = 0, 0
    temp_p = abs(m_p - eng_p)
    temp_n = abs(m_n - eng_n)
    t1 = np.count_nonzero(temp_p == 0)
    t2 = np.count_nonzero(temp_n == 0)
    exe_p = t1 / num
    exe_n = t2 / num

    record1 = '算法计算{}次，其中正向扭矩执行{}次，占比{}，负向扭矩执行{}次，占比{}'.format(num, t1, exe_p, t2, exe_n)
    print(record1)

    error_p = np.sum(temp_p)
    p_sum = np.sum(eng_p)
    e_p_per1 = error_p / p_sum
    e_p1 = error_p / num
    error_n = np.sum(temp_n)
    n_sum = abs(np.sum(eng_n))
    e_n_per1 = error_n / n_sum
    e_n1 = error_n / num
    record2 = '总体正向扭矩平均百分比误差为{}，反向扭矩平均百分比误差为{}，正、反向扭矩平均误差为{}，{}' \
        .format(e_p_per1, e_n_per1, e_p1, e_n1)
    print(record2)

    t3 = np.count_nonzero(temp_p)
    t4 = np.count_nonzero(temp_n)
    dif_p_id = np.where(temp_p != 0)
    dif_n_id = np.where(temp_n != 0)
    error_p = np.sum(temp_p)
    p_sum2 = np.sum(eng_p[dif_p_id])
    e_p_per2 = error_p / p_sum2
    e_p2 = error_p / t3
    n_sum2 = abs(np.sum(eng_n[dif_n_id]))
    e_n_per2 = error_n / n_sum2
    e_n2 = error_n / t4
    record3 = '仅计算未采用部分数据，正向扭矩平均百分比误差为{}，反向扭矩平均百分比误差为{}，正、反向扭矩平均误差为{}，{}' \
        .format(e_p_per2, e_n_per2, e_p2, e_n2)
    print(record3)

    with open("image/test_{}.txt".format(tag), "w", encoding='utf-8') as f:
        f.write(record1)  # 自带文件关闭功能，不需要再写f.close()
        f.write('\n')
        f.write(record2)
        f.write('\n')
        f.write(record3)

    # return data, m_p, m_n, eng_p, eng_n
    return data


def read_tor2(filename, tag='0118错位'):
    # 算法输出参数与实际参数错位计算
    df = pd.read_csv(filename, usecols=[2, 3, 4, 5])
    data = df.values
    print(data.shape)

    m_p, m_n, eng_p, eng_n = data[:-1, 0], data[:-1, 1], data[1:, 2], data[1:, 3]
    num = len(m_p)
    exe_p, exe_n = 0, 0
    temp_p = abs(m_p - eng_p)
    temp_n = abs(m_n - eng_n)
    t1 = np.count_nonzero(temp_p == 0)
    t2 = np.count_nonzero(temp_n == 0)
    exe_p = t1 / num
    exe_n = t2 / num

    record1 = '算法计算{}次，其中正向扭矩执行{}次，占比{}，负向扭矩执行{}次，占比{}'.format(num, t1, exe_p, t2, exe_n)
    print(record1)

    error_p = np.sum(temp_p)
    p_sum = np.sum(eng_p)
    e_p_per1 = error_p / p_sum
    e_p1 = error_p / num
    error_n = np.sum(temp_n)
    n_sum = abs(np.sum(eng_n))
    e_n_per1 = error_n / n_sum
    e_n1 = error_n / num
    record2 = '总体正向扭矩平均百分比误差为{}，反向扭矩平均百分比误差为{}，正、反向扭矩平均误差为{}，{}' \
        .format(e_p_per1, e_n_per1, e_p1, e_n1)
    print(record2)

    t3 = np.count_nonzero(temp_p)
    t4 = np.count_nonzero(temp_n)
    dif_p_id = np.where(temp_p != 0)
    dif_n_id = np.where(temp_n != 0)
    error_p = np.sum(temp_p)
    p_sum2 = np.sum(eng_p[dif_p_id])
    e_p_per2 = error_p / p_sum2
    e_p2 = error_p / t3
    n_sum2 = abs(np.sum(eng_n[dif_n_id]))
    e_n_per2 = error_n / n_sum2
    e_n2 = error_n / t4
    record3 = '仅计算未采用部分数据，正向扭矩平均百分比误差为{}，反向扭矩平均百分比误差为{}，正、反向扭矩平均误差为{}，{}' \
        .format(e_p_per2, e_n_per2, e_p2, e_n2)
    print(record3)

    with open("image/correct_test_{}.txt".format(tag), "w", encoding='utf-8') as f:
        f.write(record1)  # 自带文件关闭功能，不需要再写f.close()
        f.write('\n')
        f.write(record2)
        f.write('\n')
        f.write(record3)

    # new_data=np.hstack((m_p, m_n, eng_p, eng_n))
    new_data = np.vstack((m_p, m_n, eng_p, eng_n))
    new_data = np.transpose(new_data)
    # return data, m_p, m_n, eng_p, eng_n
    return new_data


def read_dist(filename):
    df = pd.read_csv(filename, usecols=[0, 1])
    data = df.values
    tf = data[:, 0]
    tar = data[0, 1]
    dist = tf - tar
    for i in range(len(tf)):
        if dist[i] > 180:
            dist[i] = dist[i] - 360
        elif dist[i] < -180:
            dist[i] += 360
    return dist, data


def plot_dist(dist, tag='explore'):
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.title("现场试验工具面控制效果", fontsize=20)
    plt.xlabel('调整时间步', fontsize=20)
    plt.ylabel('角度(°)', fontsize=20)
    plt.plot(dist, label='工具面距离目标角度')

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)  # 设置坐标标签字体大小

    plt.grid()
    plt.legend(fontsize=20)

    plt.savefig("image/{}_dist_curve".format(tag))
    plt.show()


def plot_tf(data, tag='explore'):
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.title("现场试验工具面控制效果", fontsize=20)
    plt.xlabel('调整时间步', fontsize=20)
    plt.ylabel('角度(°)', fontsize=20)
    plt.plot(data[:, 0], label='工具面状态')
    plt.plot(data[:, 1], label='目标工具面')

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)  # 设置坐标标签字体大小

    plt.grid()
    plt.legend(fontsize=20)

    plt.savefig("image/{}_tf_curve".format(tag))
    plt.show()


def plot_tf_en(data, tag='en'):
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.title("Field experimental results", fontsize=20)
    plt.xlabel('steps', fontsize=20)
    plt.ylabel('toolface orientation(°)', fontsize=20)
    plt.plot(data[:, 0], label='toolface',linewidth=3.0)
    plt.plot(data[:, 1], label='target toolface',linewidth=3.0)

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)  # 设置坐标标签字体大小

    plt.grid()
    plt.legend(fontsize=20)

    plt.savefig("image/{}_tf_curve".format(tag))
    plt.show()


def plot_tor(data, tag='0118tor'):
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.title("现场试验算法输出参数与工程师采用参数对比", fontsize=20)
    plt.xlabel('调整时间步', fontsize=20)
    plt.ylabel('扭矩(N*m)', fontsize=20)

    plt.plot(data[:, 0], label='算法输出正向扭矩')
    plt.plot(data[:, 2], label='实际执行正向扭矩')
    plt.plot(data[:, 1], label='算法输出反向扭矩')
    plt.plot(data[:, 3], label='实际执行反向扭矩')

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)  # 设置坐标标签字体大小

    plt.grid()
    plt.legend(fontsize=20)

    plt.savefig("image/{}_tf_curve".format(tag))
    plt.show()


def plot_tor_en(data, tag='0118toren'):
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.title("Control parameter variation curve", fontsize=20)
    plt.xlabel('steps', fontsize=20)
    plt.ylabel('torque(N*m)', fontsize=20)

    plt.plot(data[:, 0], label='Forward torque output by algorithm',linewidth=3.0)
    plt.plot(data[:, 2], label='Forward torque performed by the Engineer',linewidth=3.0)
    plt.plot(data[:, 1], label='Reverse torque output by algorithm',linewidth=3.0)
    plt.plot(data[:, 3], label='Reverse torque performed by the Engineer',linewidth=3.0)

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)  # 设置坐标标签字体大小

    plt.grid()
    plt.legend(fontsize=20)

    plt.savefig("image/{}_tf_curve".format(tag))
    plt.show()


file1 = 'data/0110-3-modify.csv'
dist, data = read_dist(file1)
plot_dist(dist, '0110-3')
plot_tf(data, '0110-3')
plot_tf_en(data)
data = read_tor(file1, tag='0110-3')
plot_tor(data, '0110tor-3')
data2 = read_tor2(file1, tag='0110-3')
plot_tor(data2, '0110-3tor-错位')
plot_tor_en(data2)
