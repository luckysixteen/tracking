# -*- coding: utf-8 -*-

import os
import numpy as np
import cv2
import tools as tl
import pandas as pd

# Case模型函数
# class switch(object):
#     def __init__(self, value):
#         self.value = value
#         self.fall = False
#
#     def __iter__(self):
#         """Return the match method once, then stop"""
#         yield self.match
#         raise StopIteration
#
#     def match(self, *args):
#         """Indicate whether or not to enter a case suite"""
#         if self.fall or not args:
#             return True
#         elif self.value in args:  # changed for v1.5, see below
#             self.fall = True
#             return True
#         else:
#             return False


# videoCapture=cv2.VideoCapture('video/144115194519432837_20170314_120930.mp4')
# fps=videoCapture.get(cv2.CAP_PROP_FPS)
# size=(int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
#
# #videoWriter=cv2.VideoWriter('out.avi',cv2.VideoWriter_fourcc('I','4','2','0'),fps,size)
#
# success,frame=videoCapture.read()
# #print fps, size, frame, success
# frame_count = 1
# all_frames = []
# #print size
#
# while success:
#     # videoWriter.write(frame)
#     success,frame=videoCapture.read()
#     frame_count = frame_count + 1
#
# print frame_count
# print all_frames
# videoCapture.release()
# cv2.destroyAllWindows()

#---------test--gray video----------

# cap = cv2.VideoCapture('video/144115194519432837_20170314_120930.mp4')
# while(True):
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#     # Our operations on the frame come here
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     # Display the resulting frame
#     cv2.imshow('frame',gray)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# # When everything done, release the capture
# cap.release()
# cv2.destroyAllWindows()


#---------Meanshift in OpenCV-------------

# cap = cv2.VideoCapture('video/144115194519432837_20170314_120930.mp4')
#
# # take first frame of the video
# ret,frame = cap.read()
# frame_count = 1
# frame_0 = np.zeros((600, 800, 3), np.int)
#
# # setup initial location of window
# r,h,c,w = 250,90,100,125  # simply hardcoded the values
# track_window = (c,r,w,h)
#
# # set up the ROI for tracking
# roi = frame[r:r+h, c:c+w]
# hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
# mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
# roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
# cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
#
# # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
# term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
#
# while(1):
#     ret ,frame = cap.read()
#
#     if ret == True:
#         hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#         dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
#
#         # apply meanshift to get the new location
#         ret, track_window = cv2.meanShift(dst, track_window, term_crit)
#
#         # Draw it on image
#         x,y,w,h = track_window
#         img = cv2.rectangle(frame, (x,y), (x+w,y+h), 255,1)
#         cv2.imshow('tracking',img)
#
#         # k = cv2.waitKey(60) & 0xff
#         # if k == 27:
#         #     break
#         # else:
#         #     cv2.imwrite(chr(k)+".jpg",img)
#
#         k = cv2.waitKey(1) & 0xff
#         for case in tl.switch(k):
#             if case(ord(' ')):
#                 k = cv2.waitKey(0) & 0xff
#                 continue
#             if case(ord('c')):
#                 cv2.imwrite(str(frame_count) + ".jpg", img)
#                 print chr(k)
#                 continue
#             if case(27):
#                 break
#             if case():
#                 continue
#         if k == 27:
#             break
#         frame_count = frame_count + 1
#
#     else:
#         break
#
# cv2.destroyAllWindows()
# cap.release()

