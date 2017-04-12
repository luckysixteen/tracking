# -*- coding: utf-8 -*-

import numpy as np
from pandas import DataFrame, Series
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from pykalman import KalmanFilter


file = 'singleball.mov'
cap = cv2.VideoCapture(file)
Measured=np.load("ballTrajectory.npy")
numframes = cap.get(7)
count = 0

MarkedMeasured = DataFrame(Measured, columns=['x', 'y', 'w', 'h'])
MarkedMeasured = DataFrame(MarkedMeasured,
                           columns=['x', 'y', 'w', 'h', 'cx', 'cy', 'kx', 'ky', 'kcx', 'kcy'])
MarkedMeasured['cx'] = MarkedMeasured['x'] + MarkedMeasured['w']/2
MarkedMeasured['cy'] = MarkedMeasured['y'] + MarkedMeasured['h']

# 找到小球进入画面检测到的第一帧
for i in range(int(numframes)):
    if MarkedMeasured['x'][i] != -1:
        ballShow = i
        break

measuredCenter = []
measuredSecond = []
for i in range(ballShow,int(numframes)):
    measuredCenter.append((MarkedMeasured['cx'][i], MarkedMeasured['cy'][i]))
    measuredSecond.append((MarkedMeasured['x'][i], MarkedMeasured['y'][i]))
measuredCenter = np.array(measuredCenter)
measuredSecond = np.array(measuredSecond)
measuredCenter = np.ma.masked_less(measuredCenter,0)
measuredSecond = np.ma.masked_less(measuredSecond,0)

# print measuredCenter
# print measuredSecond

fgbg = cv2.createBackgroundSubtractorMOG2(10, 200, False)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))


measuredTrack = np.zeros((int(numframes),4))-1
kalmanLine = []

# Kalman参数初始
Transition_Matrix = [[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]]
Observation_Matrix = [[1,0,0,0],[0,1,0,0]]
initcovariance=1.0e-4*np.eye(4)                 # describes the certainty of the initial state
transistionCov=1.0e-4*np.eye(4)                 # describes the certainty of the process model
observationCov=1.0e-1*np.eye(2)                 # describes the certainty of the measurement model


# 初始状态设置
xinit = measuredSecond[0,0]
yinit = measuredSecond[0,1]
vxinit = measuredSecond[1,0] - measuredSecond[0,0]
vyinit = measuredSecond[1,1] - measuredSecond[0,1]
initstate = [xinit, yinit, vxinit, vyinit]
second = KalmanFilter(transition_matrices=Transition_Matrix,
                  observation_matrices=Observation_Matrix,
                  initial_state_mean=initstate,
                  initial_state_covariance=initcovariance,
                  transition_covariance=transistionCov,
                  observation_covariance=observationCov)
cxinit = measuredCenter[0,0]
cyinit = measuredCenter[0,1]
cvxinit = measuredCenter[1,0] - measuredCenter[0,0]
cvyinit = measuredCenter[1,1] - measuredCenter[0,1]
cinitstate = [cxinit, cyinit, cvxinit, cvyinit]
center = KalmanFilter(transition_matrices=Transition_Matrix,
                  observation_matrices=Observation_Matrix,
                  initial_state_mean=cinitstate,
                  initial_state_covariance=initcovariance,
                  transition_covariance=transistionCov,
                  observation_covariance=observationCov)

(filtered_state_means, filtered_state_covariances) = second.smooth(measuredSecond)
(filtered_state_means_c, filtered_state_covariances_c) = center.smooth(measuredCenter)
MarkedMeasured['kx'][ballShow:int(numframes)] = filtered_state_means[:,0]
MarkedMeasured['ky'][ballShow:int(numframes)] = filtered_state_means[:,1]
MarkedMeasured['kcx'][ballShow:int(numframes)] = filtered_state_means_c[:,0]
MarkedMeasured['kcy'][ballShow:int(numframes)] = filtered_state_means_c[:,1]

print filtered_state_means
print filtered_state_means_c[:,0]

MarkedMeasured = MarkedMeasured.apply(np.round)
MarkedMeasured = MarkedMeasured.fillna(-1)
MarkedMeasured = MarkedMeasured.astype('int')

print MarkedMeasured


# kx = filtered_state_means[0]
# ky = filtered_state_means[1]
# kalmanLine.append((x,y))
# if len(kalmanLine) == 16:
#     del kalmanLine[0]
# for i in range(len(kalmanLine)-1):
#     # print i
#     # print kalmanLine
#     cv2.line(frame, kalmanLine[i], kalmanLine[i+1], (255, 0, 0), 2)

while count < numframes:
    count = count + 1
    frame = cap.read()[1]

    if MarkedMeasured['kx'][count-1] != -1:
        x = MarkedMeasured['kx'][count-1]
        y = MarkedMeasured['ky'][count-1]
        w = (MarkedMeasured['kcx'][count-1]-MarkedMeasured['kx'][count-1]) * 2
        h = MarkedMeasured['kcy'][count-1]-MarkedMeasured['ky'][count-1]
        print (x,y,w,h)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)

        kalmanLine.append((x+w/2, y+h))
        if len(kalmanLine) == 44:
            del kalmanLine[0]
        for i in range(len(kalmanLine)-1):
            cv2.line(frame, kalmanLine[i], kalmanLine[i+1], (0, 0, 255), 2)

    cv2.imshow('Tracking', frame)
    k = cv2.waitKey(150) & 0xFF
    if k == 27: break

# while count < numframes:
#     count = count + 1
#     frame = cap.read()[1]
#
#
#
#     foremat = fgbg.apply(frame, learningRate=0.001)
#     opening = cv2.morphologyEx(foremat, cv2.MORPH_OPEN, kernel)
#     ret,thresh = cv2.threshold(opening,127,255,0)
#     img, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#     # cv2.drawContours(frame, contours, -1, (0, 0, 255), 3)
#     # cv2.imshow("Video", frame)
#
#     cv2.waitKey(30)
#     # print len(contours)
#     if len(contours) > 0:
#         if count == 1:
#             plt.plot(-1, -1, 'ob')
#             continue
#
#         x = np.amin(contours[0], axis=0)[0][0]
#         y = np.amin(contours[0], axis=0)[0][1]
#         w = np.amax(contours[0], axis=0)[0][0] - x
#         h = np.amax(contours[0], axis=0)[0][1] - y
#
#         m = (x + w/2, y+h)
#         measuredTrack[count-1] = m
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)
#         # plt.plot(m[0],m[1],'ob')
#
#     if measuredTrack[count-1, 0] !=1 :
#         ball_flag = True
#         MarkedMeasured = np.ma.masked_less(measuredTrack[count - 1], 0)
#     # if measuredTrack[count-1, 0] != -1.:
#     if ball_flag:
#         # MarkedMeasured = np.ma.masked_less(measuredTrack[count-1],0)
#
#
#     cv2.imshow('Foreground',frame)
#     k = cv2.waitKey(100) & 0xFF
#     if k == 27: break




# print measuredTrack
# print np.shape(measuredTrack)

# np.save("ballTrajectory", measuredTrack)
# plt.show()

cap.release()
cv2.destroyAllWindows()