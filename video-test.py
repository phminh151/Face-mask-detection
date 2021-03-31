import cv2
import numpy as np 
from helpers.yolo import *
import time

cap = cv2.VideoCapture(0)
frame_id = 0
start_time = time.time()

while True:
    _, frame = cap.read()
    frame_id += 1
    fps = round(frame_id/(time.time()-start_time),2)
    boxes, labels = cropping_image(frame)
    cv2.putText(frame, 'FPS:'+str(fps),(10,10),cv2.FONT_HERSHEY_PLAIN,1,(0,0,255),2)
    for i in range(len(boxes)):
        x, y, w, h =boxes[i]
        label = labels[i]
        print(label)
        if label == 'no_mask':
            cv2.rectangle(frame, (x,y),(x+w, y+h), (0,0,255), 1)
            cv2.putText(frame, label,(x,y-5), cv2.FONT_HERSHEY_PLAIN,1,(0,0,255),2)
        else:
            cv2.rectangle(frame, (x,y),(x+w, y+h), (0,255,0), 1)
            cv2.putText(frame, label,(x,y-5), cv2.FONT_HERSHEY_PLAIN,1,(0,255,0),2)
    cv2.imshow('frame',frame)
    key = cv2.waitKey(1)
    if key == 27:  # Esc key
        break