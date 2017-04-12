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

# 数据和视频地址
# video_file = 'video/144115194519432837_20170314_120930.mp4'
# data_file = 'simple_data/data_70.csv'
video_file = 'simple_data/in_one_1.mp4'
data_file = 'simple_data/in_one_1.csv'

# 读取数据和视频
data_final = pd.read_csv(data_file)
cap = cv2.VideoCapture(video_file)
print data_final

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
    k = cv2.waitKey(100) & 0xff
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
    print frame_count

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



