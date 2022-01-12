# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 16:59:48 2022

@author: rinkika
"""

import cv2
import time

COLORS = [(0,255,255),(255,255,0),(0,255,0),(255,0,0)]

class_names = []
with open('Mask.names', 'r') as f:
    class_names = [cname.strip() for cname in f.readlines()]
    
    print(class_names)
