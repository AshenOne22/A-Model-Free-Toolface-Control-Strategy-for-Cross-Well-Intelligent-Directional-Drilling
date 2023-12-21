import os

import numpy as np
import pandas as pd
from pandas import read_csv

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

# 设置原数据路径
dir_path = 'ori-data2'
names = os.listdir(dir_path)
print(names)

# 读取数据
dir_path2 = 'ori-data2'
if not os.path.exists(dir_path2):
    os.makedirs(dir_path2)

file_path=os.path.join(dir_path,names[0])
df=pd.read_csv(file_path,usecols=[1,2,3,4,5,6,7,8])
df.to_csv(dir_path+'/'+'all_ori.csv',index=False)

for i in range(1,len(names)):
    file_path=os.path.join(dir_path,names[i])
    df=pd.read_csv(file_path,usecols=[1,2,3,4,5,6,7,8])
    df.to_csv(dir_path+'/'+'all_ori.csv',index=False,header=False, mode='a+')



