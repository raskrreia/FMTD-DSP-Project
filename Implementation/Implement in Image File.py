# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 21:02:33 2022

@author: Mia Abbygale Flores & Normina Abo
"""

import cv2
import numpy as np

#Load the YOLO Trained Weights

net = cv2.dnn.readNet('yolov3.weights','yolov3.cfg')
classes = []
with open ('coco.names', 'r') as f:
    classes = f.read().splitlines()
    
print(classes)
