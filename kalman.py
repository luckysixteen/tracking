# -*- coding: utf-8 -*-

# import cv2
# import numpy as np
# from pykalman import KalmanFilter
#
# meas=[]
# pred=[]
# frame = np.zeros((600,800,3), np.uint8) # 绘制面板
# mp = np.array((2,1), np.float32) # 测量数据measurement
# tp = np.zeros((2,1), np.float32) # 跟踪和预测
#
#
# def onmouse(k,x,y,s,p):
#     global mp,meas
#     mp = np.array([[np.float32(x)],[np.float32(y)]])
#     meas.append((x,y))
#
# def paint():
#     global frame,meas,pred
#     for i in range(len(meas)-1): cv2.line(frame,meas[i],meas[i+1],(200,0,0),2)
#     for i in range(len(pred)-1): cv2.line(frame,pred[i],pred[i+1],(0,0,200),2)
#
# def reset():
#     global meas,pred,frame
#     meas=[]
#     pred=[]
#     frame = np.zeros((600,800,3), np.uint8)
#
# cv2.namedWindow("kalman")
# cv2.setMouseCallback("kalman",onmouse);
# kalman = cv2.KalmanFilter(4,2)
# kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
# kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],np.float32)
# kalman.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],np.float32) * 0.03
# kalman.measurementNoiseCov = np.array([[1,0],[0,1]],np.float32) * 0.00003
# while True:
#     kalman.correct(mp)
#     tp = kalman.predict()
#     pred.append((int(tp[0]),int(tp[1])))
#     paint()
#     cv2.imshow("kalman",frame)
#     k = cv2.waitKey(1) &0xFF
#     if k == 27: break
#     if k == 32: reset()
#     print meas
#     print pred




import numpy as np
from pykalman import KalmanFilter
from matplotlib import pyplot as plt
import cv2

file = 'singleball.mov'
cap = cv2.VideoCapture(file)
numframes = cap.get(7)
count = 0


# -------------Substraction--------------
fgbg = cv2.createBackgroundSubtractorMOG2(10, 200, False)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))

measuredTrack = np.zeros((int(numframes),2))-1
(x, y, w, h) = (-1, -1, -1, -1)

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
    ret, thresh = cv2.threshold(opening, 127, 255, 0)
    img, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.waitKey(30)

    if len(contours) > 0:
        if count == 1:
            plt.plot(-1, -1, 'ob')
            continue

        x = np.amin(contours[0], axis=0)[0][0]
        y = np.amin(contours[0], axis=0)[0][1]
        w = np.amax(contours[0], axis=0)[0][0] - x
        h = np.amax(contours[0], axis=0)[0][1] - y

        m = (x + w/2, y + h)
        # print m
        measuredTrack[count-1] = m
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)

    cv2.imshow('Foreground',frame)
    k = cv2.waitKey(100) & 0xFF
    if k == 27: break

np.save("ballTrajectoryOri", measuredTrack)
cap.release()
cv2.destroyAllWindows()

# -------------Kalman--------------

Measured=np.load("ballTrajectoryOri.npy")
# count = 0
# lenum = len(Measured)
# deletearr = []
# while True:
#     if count == lenum:
#         break
#     if Measured[count,0] == -1.:
#         deletearr.append(count)
#     count = count + 1
# Measured = np.delete(Measured,deletearr,0)
# Measured=np.delete(Measured,0,0)
while True:
    if Measured[0,0]==-1.:
        Measured=np.delete(Measured,0,0)
    else:
        break
print "\t Finish \n", Measured
numMeas = Measured.shape[0]
MarkedMeasure = np.ma.masked_less(Measured,0)

Transition_Matrix = [[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]]
Observation_Matrix = [[1,0,0,0],[0,1,0,0]]

xinit=MarkedMeasure[0,0]
yinit=MarkedMeasure[0,1]
vxinit=MarkedMeasure[1,0]-MarkedMeasure[0,0]
vyinit=MarkedMeasure[1,1]-MarkedMeasure[0,1]
initstate=[xinit,yinit,vxinit,vyinit]
initcovariance=1.0e-3*np.eye(4)                 # describes the certainty of the initial state
transistionCov=1.0e-4*np.eye(4)                 # describes the certainty of the process model
observationCov=1.0e-1*np.eye(2)                 # describes the certainty of the measurement model
kf = KalmanFilter(transition_matrices = Transition_Matrix,
                  observation_matrices = Observation_Matrix,
                  initial_state_mean = initstate,
                  initial_state_covariance = initcovariance,
                  transition_covariance = transistionCov,
                  observation_covariance = observationCov)
(filtered_state_means, filtered_state_covariances) = kf.smooth(MarkedMeasure)
print filtered_state_means[:,0]
print filtered_state_means[:,1]


plt.plot(MarkedMeasure[:,0],MarkedMeasure[:,1],'xr',label='measured')
plt.axis([0,520,360,0])
plt.hold(True)
# plt.plot(filtered_state_means[:,0],filtered_state_means[:,1],'ob',label='kalman output')
plt.plot(filtered_state_means[:,0].astype(np.int),filtered_state_means[:,1].astype(np.int),'ob',label='kalman output')

plt.legend(loc=2)
plt.title("Constant Velocity Kalman Filter")
plt.show()