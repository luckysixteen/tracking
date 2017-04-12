# -*- coding: utf-8 -*-

import cv2
import pandas as pd
from pandas import Series, DataFrame


# ———————————————————文件管理———————————————————

# 定义上下限帧数
upperBound = 8400
lowerBound = 9150
simple_name = 'complex_1'


# # ———————————————————视频处理———————————————————

# 获得视频的格式
file = 'video/144115194519432837_20170314_120930.mp4'
videoCapture = cv2.VideoCapture(file)

# 获得码率及尺寸
fps = videoCapture.get(cv2.CAP_PROP_FPS)
size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

# 指定写视频的格式, I420-avi, MJPG-mp4
videoWriter = cv2.VideoWriter('simple_data/'+simple_name+'.mp4', cv2.VideoWriter_fourcc(*'XVID'), fps, size)

# 读帧
success, frame = videoCapture.read()
count = 1

while success:
    if count > upperBound:
        cv2.imshow("Show", frame)       # 显示
        cv2.waitKey(1000 / int(fps))    # 延迟
        videoWriter.write(frame)        # 写视频帧
    if count == lowerBound:
        break

    success, frame = videoCapture.read()  # 获取下一帧
    count = count + 1

videoCapture.release()
cv2.destroyAllWindows()


# ———————————————————数据处理———————————————————

data_in = pd.read_csv('simple_data/data_70.csv')
data_new = data_in[(data_in['frame'] <= lowerBound) & (data_in['frame'] > upperBound)].reset_index(drop=True)

# 帧数、秒数、秒内帧数、浮点秒数刷新
new_frame = data_new['frame'] - upperBound
new_second = new_frame//25      #秒数
new_frame_sec = new_frame%25    #秒内帧数
new_second_acc = new_frame/25   #浮点秒数

data_new['frame'] = new_frame
data_new['second'] = new_second               #秒数
data_new['frame_sec'] = new_frame_sec         #秒内帧数
data_new['second_acc'] = new_second_acc       #浮点秒数

tail_0 = DataFrame(data_new.iloc[[0]])
tail_0['frame'] = 0
tail_1 = DataFrame(data_new.iloc[[1]])
tail_1['frame'] = 1
data_new = data_new.append(tail_0, ignore_index=True)
data_new = data_new.append(tail_1, ignore_index=True)
print data_new

# 数据写入文件csv
data_new.to_csv('simple_data/'+simple_name+'.csv')