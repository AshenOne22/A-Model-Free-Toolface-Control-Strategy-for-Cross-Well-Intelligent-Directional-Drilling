import datetime
import os
import time

import pandas as pd
from pandas import read_csv
import numpy as np
import csv
import math

'''
把真实数据按时间段分开，并处理为模拟数据的格式
状态改为距离目标工具面的距离
dif_s代表距离目标工具面的距离并除以180归一化
'''

'''
df1 = read_csv('F:/LSTM_Data/data2_Label_new_p.csv',usecols=['DIFSTF','Vtf','Label_P','Label_N'], header=0)
# df1.to_csv('F:/LSTM_Data/a_data.csv')
df2=read_csv('F:/LSTM_Data/real_data_2.csv',usecols=['DIFSTF','Vtf','Label_P','Label_N'], header=0)
#df2.to_csv('F:/LSTM_Data/a_data.csv')
frames = [df1, df2]
df=pd.concat(frames)
df.to_csv('F:/LSTM_Data/a_data.csv')
'''

# df1 = read_csv('F:/LSTM_Data/data_real.csv',usecols=['ACTIME','TIME','TARSTF','SGTF','TOR_P','TOR_N'], header=0)
# # df1.to_csv('F:/LSTM_Data/a_data.csv')
# df2=read_csv('F:/LSTM_Data/real_data.csv',usecols=['ACTIME','TIME','TARSTF','SGTF','TOR_P','TOR_N'], header=0)
# #df2.to_csv('F:/LSTM_Data/a_data.csv')
# frames = [df1, df2]
# df=pd.concat(frames)
# df.to_csv('F:/LSTM_Data/a_data_ori.csv')

# 设置分割数据的保存路径
path = 'true_data4'
if not os.path.exists(path):
    os.makedirs(path)

df = read_csv('ori-data/0107-ori.csv',
              usecols=['ACTIME', 'TIME', 'TARSTF', 'SGTF', 'dif','dif_s', 'TOR_P', 'TOR_N', 'DATE', 'ss'])
data = df.values

length = len(data)
date = data[..., -1]

group = []
group_index = []
group_num = 0
dif = []
# # 增加diftf列表示距离目标工具面的距离
# for i in  range(length):
#     diftf=data[i,2]-data[i,3]
#     if diftf>180:
#         diftf=diftf-360
#     elif diftf<-180:
#         diftf=diftf+360
#     dif.append(diftf)
# new_dif=np.array(dif)
# new_dif=new_dif.reshape(length,1)
# data=np.concatenate((data,new_dif),axis=1)
# new_df=pd.DataFrame(data,columns=['ACTIME', 'TIME', 'TARSTF', 'SGTF', 'TOR_P', 'TOR_N', 'DATE', 'ss2','dif'])
# new_df.to_csv('b_data_ori.csv')


# 数据按时间分段
for i in range(length):
    temp = date[i]
    # 初始数据是新段
    if i == 0:
        group.append([])
        group_index.append([])
        temp_time = np.array([0])
        temp_data = np.concatenate((data[i], temp_time))
        group[group_num].append(temp_data)
        group_index[group_num].append(i)
    elif i > 0:
        # 时间差小于600秒，认为数据连续
        if abs(temp - date[i - 1]) < 3000:
            temp_time = np.array([data[i, -1] - data[i - 1, -1]])
            temp_data = np.concatenate((data[i], temp_time))
            group[group_num].append(temp_data)
            group_index[group_num].append(i)
        # 时间差大于600秒，数据不连续
        else:
            group_num += 1
            group.append([])
            temp_time = np.array([0])
            temp_data = np.concatenate((data[i], temp_time))
            group_index.append([])
            group[group_num].append(temp_data)
            group_index[group_num].append(i)
i = 0
group_num_1 = []

while i <= len(group):
    if len(group[i]) < 1:
        del group[i]
    else:
        group_num_1.append(len(group[i]))
        i += 1
    if i == len(group):
        break
print(group[1][0][0])
new_group = np.array(group)
newdata = locals()
prepare_list = locals()
df = locals()
# n段数据，创建n个列表
print(new_group.shape)
cols = ['ACTIME', 'TIME', 'TARSTF', 'SGTF','dif','dif_s', 'TOR_P', 'TOR_N', 'DATE', 'ss', 'time_interval']
for i in range(len(new_group)):
    df = pd.DataFrame(new_group[i], columns=cols)
    df.to_csv(os.path.join(path, str(i) + '.csv'))

# # 合并数据集
# for i in range(len(group)):
#     r_path = os.path.join('F:/A-1/com_data/true_data', '_'.join([ str(i), '.csv']))
#     df1 = read_csv(r_path, header=0)
#     if i == 0:
#         df = df1
#     else:
#         frames = [df, df1]
#         df = pd.concat(frames)

# df.to_csv('F:/A-1/com_data/true_data/0_end_all.csv', index=False)

print('finished')
