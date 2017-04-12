# -*- coding: utf-8 -*-

import pandas as pd
from pandas import Series, DataFrame
import numpy as np

#导入待处理数据
file = '144115194519432837_20170314_120930.mp4.txt'
data_raw = pd.read_csv(file,
                       sep=' ',
                       names=['xmin', 'ymin', 'xmax', 'ymax', 'notknown', 'picid', 'frame', 'confidenceI'])

#一层过滤原始数据：筛选置信区间值
data_raw_low = data_raw[data_raw['confidenceI'] > 0.50 ]
data_raw60 = data_raw[data_raw['confidenceI'] > 0.60 ]
data_raw65 = data_raw[data_raw['confidenceI'] > 0.65 ]
data_raw70 = data_raw[data_raw['confidenceI'] > 0.70 ]
data_raw75 = data_raw[data_raw['confidenceI'] > 0.75 ]

data_df = data_raw70
# #丢弃轴上原始数据
# data_drop_raw60 = data_raw.drop([['confidenceI'] > 0.70])

# 获得数据
xmin = data_df['xmin']
xmax = data_df['xmax']
ymin = data_df['ymin']
ymax = data_df['ymax']
frame = data_df['frame']

# 计算长宽
length = xmax - xmin
height = ymax - ymin

# 计算中心点xy坐标
center_x = xmin + length / 2
center_y = ymin + height / 2
#center = [center_x, center_y]

# 计算取整并转换整数
xmin = xmin.apply(np.round)
ymin = ymin.apply(np.round)
length = length.apply(np.round)
height = height.apply(np.round)

# 计算：将帧数转换成时间
second = frame//25      #秒数
frame_sec = frame%25    #秒内帧数
second_acc = frame/25   #浮点秒数

# 计算像素速度


# 重新排列序号
data_no = DataFrame(data_df,
                       columns=['no', 'xmin', 'ymax', 'xmax', 'ymin', 'notknown', 'picid', 'frame', 'confidenceI'])
data_no['no'] = np.arange(1, len(data_no)+1)

#生成最终数据
data_process = DataFrame(data_no,
                    columns=['no', 'xmin', 'ymax', 'xmax', 'ymin',
                             'length', 'height', 'center_x', 'center_y',
                             'frame', 'second', 'frame_sec', 'second_acc', 'confidenceI'])
data_process['xmin'] = xmin
data_process['ymin'] = ymin
data_process['length'] = length               #检测框长度
data_process['height'] = height               #检测框高度
data_process['center_x'] = center_x           #检测中心点x坐标
data_process['center_y'] = center_y           #检测中心点y坐标
data_process['second'] = second               #秒数
data_process['frame_sec'] = frame_sec         #秒内帧数
data_process['second_acc'] = second_acc       #浮点秒数

# 字段类型转换
data_process['xmin'] = data_process['xmin'].astype('int')
data_process['ymin'] = data_process['ymin'].astype('int')
data_process['length'] = data_process['length'].astype('int')
data_process['height'] = data_process['height'].astype('int')

#
tail_0 = DataFrame(data_process.iloc[[0]])
tail_0['frame'] = 0
tail_1 = DataFrame(data_process.iloc[[1]])
tail_1['frame'] = 1
data_process = data_process.append(tail_0, ignore_index=True)
data_process = data_process.append(tail_1, ignore_index=True)

data_final = data_process.reset_index(drop=True)


# test = DataFrame(frame, columns=['frame', 'second', 'frame_sec'])
# test['second'] = second
# test['frame_sec'] = frame_sec

print data_final

# 存储数据为csv格式
# data_final.to_csv('data/data_70.csv')