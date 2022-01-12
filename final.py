# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 17:08:13 2022

"""

import cv2
import time

COLORS = [(0,255,0),(255,255,0),(0,255,0),(255,0,0)]

class_names = []
with open("obj.names","r") as f:
    class_names = [cname.strip() for cname in f.readlines()]
    
#print(class_names)

#   Capture Webcam
cap = cv2.VideoCapture(0)

#   Load Model
net = cv2.dnn.readNet("Mask.weights","Mask.cfg")

#   Detect Model
model = cv2.dnn_DetectionModel(net)
model.setInputParams( scale = 1/255, size = (416,416), mean = None, swapRB = None, crop = None)


while True:
    
    #   Capture Frame
    _, frame = cap.read()
    
    #   Date
    start = time.time()
    
    #   Detect
    classes, scores, boxes = model.detect(frame, 0.1, 0.2)
    
    # Time
    end = time.time()
    
    # for loop
    for (classid, score, box) in zip(classes, scores, boxes):
        
        #   Apply Color
        color = COLORS[int(classid) % len(COLORS)]
        
        #   Label
        label = f"{class_names[classid[0]]} : {score}"
        
        #   Box
        cv2.rectangle(frame, box, color, 2)
        
        #   Box
        cv2.putText(frame, label, (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    #   FPS LABEL
    fps_label = f"FPS: {round((1.0/(end - start)), 2)}"
    
    #   Ewan
    cv2.putText(frame, fps_label, (0,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 5)
    cv2.putText(frame, fps_label, (0,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)
    
    #   Frame Title
    cv2.imshow("detections", frame)

    #   Exit
    if cv2.waitKey(1)==27:
        break

#   Super Exit
cap.release()
cv2.destroyAllWindows()        
    
