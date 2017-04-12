# -*- coding: utf-8 -*-

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pykalman import KalmanFilter


file = 'singleball.mov'
cap = cv2.VideoCapture(file)
numframes = cap.get(7)
count = 0

fgbg = cv2.createBackgroundSubtractorMOG2(10, 200, False)
(x, y, w, h) = (-1, -1, -1, -1)
ball_flag = False
# plt.figure()
# plt.hold(True)
# plt.axis([0,480,360,0])
# kernel = np.ones((5,5),np.uint8)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))

measuredTrack = np.zeros((int(numframes),4))-1
kalmanLine = []
# kalmanLine = np.zeros((15,2))-1
# kalmanLine = np.ma.masked_less(kalmanLine[:],0)


# Kalman初始------OpenCV
# kalman = cv2.KalmanFilter(4,2)
# kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
# kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],np.float32)
# kalman.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],np.float32) * 0.03
# kalman.measurementNoiseCov = np.array([[1,0],[0,1]],np.float32) * 0.00003

Transition_Matrix = [[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]]
Observation_Matrix = [[1,0,0,0],[0,1,0,0]]
initcovariance=1.0e-4*np.eye(4)                 # describes the certainty of the initial state
transistionCov=1.0e-4*np.eye(4)                 # describes the certainty of the process model
observationCov=1.0e-1*np.eye(2)                 # describes the certainty of the measurement model

while count < numframes:
    count = count + 1
    frame = cap.read()[1]


    foremat = fgbg.apply(frame, learningRate=0.001)
    opening = cv2.morphologyEx(foremat, cv2.MORPH_OPEN, kernel)
    ret,thresh = cv2.threshold(opening,127,255,0)
    img, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(frame, contours, -1, (0, 0, 255), 3)
    # cv2.imshow("Video", frame)

    cv2.waitKey(30)
    # print len(contours)
    if len(contours) > 0:
        if count == 1:
            plt.plot(-1, -1, 'ob')
            continue

        x = np.amin(contours[0], axis=0)[0][0]
        y = np.amin(contours[0], axis=0)[0][1]
        w = np.amax(contours[0], axis=0)[0][0] - x
        h = np.amax(contours[0], axis=0)[0][1] - y

        m = (x, y, w, h)
        print m
        measuredTrack[count-1] = m
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)
        # plt.plot(m[0],m[1],'ob')

    # if measuredTrack[count-1, 0] !=1 :
    #     ball_flag = True
    #     MarkedMeasured = np.ma.masked_less(measuredTrack[count - 1], 0)
    # # if measuredTrack[count-1, 0] != -1.:
    # if ball_flag:
    #     # MarkedMeasured = np.ma.masked_less(measuredTrack[count-1],0)
    #     MarkedMeasured = measuredTrack
    #     xinit = MarkedMeasured[count-1, 0]
    #     yinit = MarkedMeasured[count-1, 1]
    #     vxinit = MarkedMeasured[count, 0] - MarkedMeasured[count-1, 0]
    #     vyinit = MarkedMeasured[count, 1] - MarkedMeasured[count-1, 1]
    #     initstate = [xinit, yinit, vxinit, vyinit]
    #     kf = KalmanFilter(transition_matrices=Transition_Matrix,
    #                       observation_matrices=Observation_Matrix,
    #                       initial_state_mean=initstate,
    #                       initial_state_covariance=initcovariance,
    #                       transition_covariance=transistionCov,
    #                       observation_covariance=observationCov)
    #     (filtered_state_means, filtered_state_covariances) = kf.filter(MarkedMeasured)
    #
    #     kx = filtered_state_means[0]
    #     ky = filtered_state_means[1]
    #     kalmanLine.append((x,y))
    #     if len(kalmanLine) == 16:
    #         del kalmanLine[0]
    #     for i in range(len(kalmanLine)-1):
    #         # print i
    #         # print kalmanLine
    #         cv2.line(frame, kalmanLine[i], kalmanLine[i+1], (255, 0, 0), 2)

    cv2.imshow('Foreground',frame)
    k = cv2.waitKey(100) & 0xFF
    if k == 27: break
#cap.release()

# print measuredTrack
# print np.shape(measuredTrack)

np.save("ballTrajectory", measuredTrack)
# plt.show()

cap.release()
cv2.destroyAllWindows()