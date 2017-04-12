# -*- coding: utf-8 -*-

import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import cv2
import tools as tl

# 数据和视频地址
# video_file = 'video/144115194519432837_20170314_120930.mp4'
# data_file = 'simple_data/data_70.csv'
# video_file = 'simple_data/in_one_1.mp4'
# data_file = 'simple_data/in_one_1.csv'

video_file = 'simple_data/complex_1.mp4'
data_file = 'simple_data/complex_1.csv'

# 读取数据和视频
data_final = pd.read_csv(data_file)
cap = cv2.VideoCapture(video_file)
print data_final

# 读取视频第一帧
success,frame = cap.read()
cap_fps=cap.get(cv2.CAP_PROP_FPS)
cap_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
cap_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_0 = np.zeros((cap_width, cap_height, 3), np.int)
# print frame.shape == frame_0.shape


# 初始化检测框、帧数统计
track_window_data = [[0, 0, 0, 0]]*10
frame_count = 1
numframes = cap.get(7)

# 初始化数据驱使的检测框
data_flag = 0
data_number = 1
frame_cerrent = data_final['frame'][data_flag]
frame_next = data_final['frame'][data_flag + 1]
track_window_data[0] = [data_final['xmin'][data_flag],
                   data_final['ymin'][data_flag],
                   data_final['length'][data_flag],
                   data_final['height'][data_flag]]

# 扫描同一帧数的检测框
while (frame_cerrent == frame_next):
    data_flag = data_flag + 1
    track_window_data[data_number] = [data_final['xmin'][data_flag],
                                      data_final['ymin'][data_flag],
                                      data_final['length'][data_flag],
                                      data_final['height'][data_flag]]
    data_number = data_number + 1
    frame_next = data_final['frame'][data_flag + 1]



# 初始化Camshift算法驱使的检测框
cam_xmin = data_final['xmin'][0]
cam_ymin = data_final['ymin'][0]
cam_length = data_final['length'][0]-20
cam_height = data_final['height'][0]-40
track_window_cam = (cam_xmin, cam_ymin, cam_length, cam_height)
print track_window_cam

# ROI-感兴趣区域-图像采集
roi = frame[cam_ymin:cam_ymin+cam_height, cam_xmin:cam_xmin+cam_length]
# cv2.imshow('test', roi)
# k = cv2.waitKey(0) & 0xff
hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1, 1 )

#背景分离
fgbg = cv2.createBackgroundSubtractorMOG2(500, 16, True)
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
#
# while frame_count < numframes:
#     frame_count = frame_count + 1
#     frame = cap.read()[1]
#     # cv2.imshow("Video",frame)
#
#     foremat = fgbg.apply(frame, learningRate=0.001)
#     opening = cv2.morphologyEx(foremat, cv2.MORPH_OPEN, kernel)
#     ret,thresh = cv2.threshold(opening,127,255,0)
#
#     cv2.imshow('Foreground',thresh)
#     # cv2.imshow('normal', frame)
#     k = cv2.waitKey(1) & 0xFF
#     if k == 27:
#         break

while(success):
    success, frame = cap.read()
    img = frame

    #数据驱使的检测框
    if frame_count == frame_cerrent:
        x1, y1, w1, h1 = track_window_data[0]
        cv2.rectangle(img, (x1, y1), (x1 + w1, y1 + h1), (255,0,0), 2)

        for case in tl.switch(data_number):
            # if case(1):
            #     cv2.imshow('TRACKING', img)
            #     break
            if case(2):
                x2, y2, w2, h2 = track_window_data[1]
                cv2.rectangle(img, (x2, y2), (x2 + w2, y2 + h2), (0,255,0), 2)
                # cv2.imshow('TRACKING', img)
                break
            if case(3):
                x2, y2, w2, h2 = track_window_data[1]
                x3, y3, w3, h3 = track_window_data[2]
                cv2.rectangle(img, (x2, y2), (x2 + w2, y2 + h2), (0,255,0), 2)
                cv2.rectangle(img, (x3, y3), (x3 + w3, y3 + h3), (0,0,255), 2)
                # cv2.imshow('TRACKING', img)
                break
            if case(4):
                x2, y2, w2, h2 = track_window_data[1]
                x3, y3, w3, h3 = track_window_data[2]
                x4, y4, w4, h4 = track_window_data[3]
                cv2.rectangle(img, (x2, y2), (x2 + w2, y2 + h2), (0,255,0), 2)
                cv2.rectangle(img, (x3, y3), (x3 + w3, y3 + h3), (0,0,255), 2)
                cv2.rectangle(img, (x4, y4), (x4 + w4, y4 + h4), (255,255,0), 2)
                # cv2.imshow('TRACKING', img)
                break

        track_window_data = [[0, 0, 0, 0]] * 10
        data_flag = data_flag + 1
        data_number = 1
        frame_cerrent = data_final['frame'][data_flag]
        frame_next = data_final['frame'][data_flag + 1]
        track_window_data[0] = [data_final['xmin'][data_flag],
                                data_final['ymin'][data_flag],
                                data_final['length'][data_flag],
                                data_final['height'][data_flag]]
        while (frame_cerrent == frame_next):
            data_flag = data_flag + 1
            track_window_data[data_number] = [data_final['xmin'][data_flag],
                                              data_final['ymin'][data_flag],
                                              data_final['length'][data_flag],
                                              data_final['height'][data_flag]]
            data_number = data_number + 1
            frame_next = data_final['frame'][data_flag + 1]
        # cv2.imshow('TRACKING', img)

    # # Camshift驱使的检测框
    # if frame_count >= data_final['frame'][0]:
    #     print frame_count
    #     print "TRUE!!!"
    #     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #     dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
    #
    #     # apply meanshift to get the new location
    #     success, track_window_cam = cv2.CamShift(dst, track_window_cam, term_crit)
    #
    #     # CamShift画板
    #     pts = cv2.boxPoints(success)
    #     pts = np.int0(pts)
    #     cv2.polylines(img, [pts], True, (0, 0, 255), 2)
    #
    #     # # apply meanshift to get the new location
    #     # success, track_window_cam = cv2.meanShift(dst, track_window_cam, term_crit)
    #     #
    #     # # meanShift画板
    #     # x, y, w, h = track_window_cam
    #     # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    #
    #     k = cv2.waitKey(0) & 0xff

    else:
        img = cv2.rectangle(frame, (0, 0), (0, 0), 255, 2)

    # 键盘控制
    cv2.imshow('TRACKING', img)
    k = cv2.waitKey(1) & 0xff
    for case in tl.switch(k):
        if case(ord('n')):
            k = cv2.waitKey(1000) & 0xff
            continue
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
