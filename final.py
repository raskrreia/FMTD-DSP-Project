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
    
print(class_names)

#   Capture Webcam
cap = cv2.VideoCapture(0)

#   Load Model
net = cv2.dnnreadNet("Mask.weights","Mask.cfg")

#   Detect Model
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416,416, scale = 1/255))

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
        
    
