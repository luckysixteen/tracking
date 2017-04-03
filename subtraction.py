# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
#
# file="singleball.mov"
# capture = cv2.VideoCapture(file)
# print "\t Width: ",capture.get(cv2.CAP_PROP_FRAME_WIDTH)
# print "\t Height: ",capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
# print "\t FourCC: ",capture.get(cv2.CAP_PROP_FOURCC)
# print "\t Framerate: ",capture.get(cv2.CAP_PROP_FPS)
# numframes=capture.get(7)
# print "\t Number of Frames: ",numframes
# count=0
# history = 10
# nGauss = 3
# bgThresh = 0.6
# noise = 20
# bgs = cv2.createBackgroundSubtractorMOG2(history,nGauss,bgThresh,noise)
# plt.figure()
# plt.hold(True)
# plt.axis([0,480,360,0])
# measuredTrack=np.zeros((numframes,2))-1
# while count<numframes:
#     count+=1
#     img2 = capture.read()[1]
#     cv2.imshow("Video",img2)
#     foremat=bgs.apply(img2)
#     cv2.waitKey(100)
#     foremat=bgs.apply(img2)
#     ret,thresh = cv2.threshold(foremat,127,255,0)
#     contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#     if len(contours) > 0:
#         m= np.mean(contours[0],axis=0)
#         measuredTrack[count-1,:]=m[0]
#         plt.plot(m[0,0],m[0,1],'ob')
#     cv2.imshow('Foreground',foremat)
#     cv2.waitKey(80)
# capture.release()
# print measuredTrack
# np.save("ballTrajectory", measuredTrack)
# plt.show()


import numpy as np
import cv2
import matplotlib.pyplot as plt

file = 'singleball.mov'
# file = 'video/144115194519432837_20170314_120930.mp4'
cap = cv2.VideoCapture(file)
numframes = cap.get(7)
count = 0
print "\t Number of Frames: ",numframes

fgbg = cv2.createBackgroundSubtractorMOG2(10, 16, False)

plt.figure()
plt.hold(True)
plt.axis([0,480,360,0])
# kernel = np.ones((5,5),np.uint8)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
print kernel

measuredTrack = np.zeros((int(numframes),2))-1
while count < numframes:
    count = count + 1
    frame = cap.read()[1]
    # cv2.imshow("Video",frame)

    foremat = fgbg.apply(frame, learningRate=0.7)
    opening = cv2.morphologyEx(foremat, cv2.MORPH_OPEN, kernel)
    # foremat=fgbg.apply(frame)
    ret,thresh = cv2.threshold(opening,127,255,0)
    img, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(frame, contours, -1, (0, 0, 255), 3)
    # cv2.imshow("Video", frame)
    # print contours

    cv2.waitKey(30)
    # print "\t contours: ",contours
    # print len(contours)
    if len(contours) > 0:
        m = np.mean(contours[0],axis=0)
        # print "\t m: ",m
        measuredTrack[count-1] = m[0]
        # print m[0]
        # print "\t measured: ", measuredTrack
        if count == 1:
            plt.plot(-1, -1, 'ob')
        else:
            plt.plot(m[0,0],m[0,1],'ob')
    cv2.imshow('Foreground',foremat)
    k = cv2.waitKey(30) & 0xFF
    if k == 27: break
#cap.release()
# print measuredTrack
# print np.shape(measuredTrack)

np.save("ballTrajectory", measuredTrack)
plt.show()

# while(1):
#     ret, frame = cap.read()
#
#     fgmask = fgbg.apply(frame)
#
#     cv2.imshow('frame',fgmask)
#     k = cv2.waitKey(1) & 0xff
#     if k == 27:
#         break

cap.release()
cv2.destroyAllWindows()