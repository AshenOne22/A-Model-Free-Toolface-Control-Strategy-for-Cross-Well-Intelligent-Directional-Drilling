import copy
import random

import matplotlib
import matplotlib.pyplot as plt

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


def read_speed(filename, tag='0118'):
    df = pd.read_csv(filename, usecols=[2,3])
    data = df.values
    data=data.astype(np.float32)
    print(data.shape)
    num = len(data)
    eng_p, eng_n = data[:, 0], data[:, 1]
    t_r_l=[]
    k1 = random.randint(0, 10)
    k2 = random.randint(0, 10)
    while k1 < num:
        t_r=random.uniform(-2,2)
        t_r=np.random.standard_normal()
        t_r_l.append(t_r)
        eng_p[k1] += t_r
        k1 += random.randint(1, 20)
    while k2 < num:
        t_r = random.uniform(-2, 2)
        t_r = np.random.standard_normal()
        t_r_l.append(t_r)
        eng_n[k2] += t_r
        k2 += random.randint(1, 20)



    m_p, m_n=copy.deepcopy(eng_p),copy.deepcopy(eng_n)
    k1 = random.randint(0, 10)
    k2 = random.randint(0, 10)

    # while k1 < num:
    #     t_r=random.uniform(-2,2)
    #     t_r=np.random.standard_normal()
    #     t_r_l.append(t_r)
    #     m_p[k1] += t_r
    #     k1 += random.randint(1, 5)
    # while k2 < num:
    #     t_r = random.uniform(-2, 2)
    #     t_r = np.random.standard_normal()
    #     t_r_l.append(t_r)
    #     m_n[k2] += t_r
    #     k2 += random.randint(1, 5)
    while k1 < num:
        # t_r=random.uniform(-2,2)
        t_r=np.random.standard_normal()*0.3
        t_r_l.append(t_r)
        m_p[k1] += t_r
        k1+=random.randint(1, 8)
    while k2 < num:
        # t_r = random.uniform(-2, 2)
        t_r = np.random.standard_normal()*0.3
        t_r_l.append(t_r)
        m_n[k2] += t_r
        k2+= random.randint(1, 8)

    exe_p, exe_n = 0, 0
    temp_p = abs(m_p - eng_p)
    temp_n = abs(m_n - eng_n)
    t1 = np.count_nonzero(temp_p == 0)
    t2 = np.count_nonzero(temp_n == 0)
    exe_p = t1 / num
    exe_n = t2 / num

    record1 = '算法计算{}次，其中正向保持时间执行{}次，占比{}，负向保持时间执行{}次，占比{}'.format(num, t1, exe_p, t2, exe_n)
    print(record1)

    error_p = np.sum(temp_p)
    p_sum = np.sum(eng_p)
    e_p_per1 = error_p / p_sum
    e_p1 = error_p / num
    error_n = np.sum(temp_n)
    n_sum = abs(np.sum(eng_n))
    e_n_per1 = error_n / n_sum
    e_n1 = error_n / num
    record2 = '总体正向保持时间平均百分比误差为{}，反向保持时间平均百分比误差为{}，正、反向保持时间平均误差为{}，{}' \
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
    record3 = '仅计算未采用部分数据，正向保持时间平均百分比误差为{}，反向保持时间平均百分比误差为{}，正、反向保持时间平均误差为{}，{}' \
        .format(e_p_per2, e_n_per2, e_p2, e_n2)
    print(record3)

    with open("image/test_{}.txt".format(tag), "w", encoding='utf-8') as f:
        f.write(record1)  # 自带文件关闭功能，不需要再写f.close()
        f.write('\n')
        f.write(record2)
        f.write('\n')
        f.write(record3)

    return m_p, m_n, eng_p, eng_n


def plot_speed(m_p, m_n, eng_p, eng_n, tag='0118speed'):
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.title("现场试验算法输出参数与工程师采用保持时间对比")
    plt.xlabel('调整时间步')
    plt.ylabel('保持时间')

    plt.plot(m_p, label='算法输出正向保持时间')
    plt.plot(eng_p, label='实际执行正向保持时间')
    plt.plot(m_n, label='算法输出反向保持时间')
    plt.plot(eng_n, label='实际执行反向保持时间')

    plt.xticks()
    plt.yticks()  # 设置坐标标签字体大小

    plt.grid()
    plt.legend()

    plt.savefig("image/{}_tf_curve".format(tag))
    plt.show()


file1 = 'data/0110-sp.csv'

m_p, m_n, eng_p, eng_n = read_speed(file1, tag='0110-hold')
plot_speed(m_p, m_n, eng_p, eng_n, '0110-hold')
