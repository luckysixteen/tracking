# -*- coding: utf-8 -*-

import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
# import matplotlib.patches as patches
# import os
import cv2
import tools as tl

data_raw = pd.read_csv('144115194519432837_20170314_120930.mp4.txt', sep=' ',
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


data_final = data_process.reset_index(drop=True)


# test = DataFrame(frame, columns=['frame', 'second', 'frame_sec'])
# test['second'] = second
# test['frame_sec'] = frame_sec

print data_final




cap = cv2.VideoCapture('video/144115194519432837_20170314_120930.mp4')

# 读取视频第一帧
success,frame = cap.read()
frame_0 = np.zeros((600, 800, 3), np.int)
# print frame.shape == frame_0.shape

# 初始化检测框、帧数统计
track_window = [[0, 0, 0, 0]]*10


frame_count = 1

# 扫描同一帧数的检测框
data_flag = 0
data_number = 1
frame_cerrent = data_final['frame'][data_flag]
frame_next = data_final['frame'][data_flag + 1]
track_window[0] = [data_final['xmin'][data_flag],
                   data_final['ymin'][data_flag],
                   data_final['length'][data_flag],
                   data_final['height'][data_flag]]

while (frame_cerrent == frame_next):
    data_flag = data_flag + 1
    track_window[data_number] = [data_final['xmin'][data_flag],
                                 data_final['ymin'][data_flag],
                                 data_final['length'][data_flag],
                                 data_final['height'][data_flag]]
    data_number = data_number + 1
    frame_next = data_final['frame'][data_flag + 1]


x1, y1, w1, h1 = track_window[0]


# x1, y1, w1, h1 = track_window[0]
# cv2.rectangle(img, (x1, y1), (x1 + w1, y1 + h1), 255, 2)
# cv2.rectangle(img, (100, 100), (100 + 100, 100 + 100), 255,2)
# cv2.imshow('test',img)
# k = cv2.waitKey(0) & 0xff

while(success):
    success, frame = cap.read()
    img = frame

    if frame_count == frame_cerrent:
        x1, y1, w1, h1 = track_window[0]
        cv2.rectangle(img, (x1, y1), (x1 + w1, y1 + h1), (255,0,0), 2)

        for case in tl.switch(data_number):
            if case(1):
                cv2.imshow('TRACKING', img)
                break
            if case(2):
                x2, y2, w2, h2 = track_window[1]
                cv2.rectangle(img, (x2, y2), (x2 + w2, y2 + h2), (0,255,0), 2)
                cv2.imshow('TRACKING', img)
                break
            if case(3):
                x2, y2, w2, h2 = track_window[1]
                x3, y3, w3, h3 = track_window[2]
                cv2.rectangle(img, (x2, y2), (x2 + w2, y2 + h2), (0,255,0), 2)
                cv2.rectangle(img, (x3, y3), (x3 + w3, y3 + h3), (0,0,255), 2)
                cv2.imshow('TRACKING', img)
                break
            if case(4):
                x2, y2, w2, h2 = track_window[1]
                x3, y3, w3, h3 = track_window[2]
                x4, y4, w4, h4 = track_window[3]
                cv2.rectangle(img, (x2, y2), (x2 + w2, y2 + h2), (0,255,0), 2)
                cv2.rectangle(img, (x3, y3), (x3 + w3, y3 + h3), (0,0,255), 2)
                cv2.rectangle(img, (x4, y4), (x4 + w4, y4 + h4), (255,255,0), 2)
                cv2.imshow('TRACKING', img)
                break

        track_window = [[0, 0, 0, 0]] * 10
        data_flag = data_flag + 1
        data_number = 1
        frame_cerrent = data_final['frame'][data_flag]
        frame_next = data_final['frame'][data_flag + 1]
        track_window[0] = [data_final['xmin'][data_flag],
                           data_final['ymin'][data_flag],
                           data_final['length'][data_flag],
                           data_final['height'][data_flag]]
        while (frame_cerrent == frame_next):
            data_flag = data_flag + 1
            track_window[data_number] = [data_final['xmin'][data_flag],
                                         data_final['ymin'][data_flag],
                                         data_final['length'][data_flag],
                                         data_final['height'][data_flag]]
            data_number = data_number + 1
            frame_next = data_final['frame'][data_flag + 1]
    else:
        img = cv2.rectangle(frame, (0, 0), (0, 0), 255, 2)
        cv2.imshow('TRACKING', img)

    # 键盘控制
    k = cv2.waitKey(1) & 0xff
    for case in tl.switch(k):
        if case(ord(' ')):
            k = cv2.waitKey(0) & 0xff
            continue
        if case(ord('c')):
            cv2.imwrite(str(frame_count) + ".jpg", img)
            print chr(k)
            continue
        if case(27):
            break
        if case():
            continue
    if k == 27:
        break

    frame_count = frame_count + 1

cv2.destroyAllWindows()
cap.release()









#----------------打印面板----------------
# plt.figure(figsize=(15,11.5))
#
# #中心点圆
# df = DataFrame(data_final[data_final['frame'] < 1709])
# plt.scatter(df.center_x, df.center_y, c=df.second_acc,
#             cmap=plt.cm.Blues, s=300,zorder=1)
# cbar = plt.colorbar(orientation="horizontal")
# cbar.ax.invert_xaxis()
# background =plt.imread("bg.png")
# plt.imshow(background,zorder=0, extent=[0,800,600,0])
# #plt.xlim(0,200)
#
# plt.show()

#检测框


# df = list(data_final[data_final['frame']==1704])
#
# background =plt.imread("bg.png")
# verts = [(df['xmin'], df.ymin), # left, bottom
#          (df.xmin, df.ymax), # left, top
#          (df.xmax, df.ymax), # right, top
#          (df.xmax, df.ymin), # right, bottom
#          (0., 0.)]          # ignored
# print df
# codes = [Path.MOVETO,
#          Path.LINETO,
#          Path.LINETO,
#          Path.CLOSEPOLY]
# path = Path(verts, codes)
# patch = patches.PathPatch(path, facecolor='orange', lw=1, alpha=0.2)
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.add_patch(patch)
# ax.set_xlim(0,800)
# ax.set_ylim(600,0)
# #plt.imshow(background,zorder=0, extent=[0,800,600,0])
# plt.show()



