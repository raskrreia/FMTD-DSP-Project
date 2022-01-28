import cv2 as cv
import numpy as np
import time

#   Confidence Threshold & Non-Max Suppression Threshold
Conf_threshold = 0.1
NMS_threshold = 0.7

COLORS = [(0,0,255),(255,0,255),(255,255,0),(0,255,255)]
"""
    Color Code:
        Red = 0,0,255 #No Mask
        Violet = 0,255,0 #Fabric Mask
        Yellow = 0,255,255 #FFP Mask
        Sky Blue = 255,0,0 #Surgical Mask
"""
#   Class Names: No Mask, Surgical Mask, Fabric Mask, FFP Mask
class_name = []
with open('MaskTypes.names','r') as f:
    class_name = [cname.strip() for cname in f.readlines()]
    
print(class_name)
print("\nFacemask Type Detector (DSP Project)")

#   Load YOLOv4-Tiny Model
net = cv.dnn.readNet('Final.weights','Final.cfg')
net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

#   Detect Model
model = cv.dnn_DetectionModel(net)
#   Set Parameters
model.setInputParams(size=(416,416), scale = 1/255, swapRB=True)

#   Source Feed: Webcam
cap = cv.VideoCapture(0) #0 for main, 1 = secondary cam,... 

while True:
    ret, frame =cap.read()

#   Bilateral Filter    
    dimpixel = 2
    sigmacolor = 12
    sigmaspace = 1
    
    bilate = cv.bilateralFilter(frame, dimpixel,sigmacolor,sigmaspace)

#   Feed Sharpening
    alpha = 3.5 #1.5,3.5, 7.5, 13.5
    beta = -2.5 #-0.5, -2.5, -6.5,-10.5
    gamma = 10
    
    sharpen = cv.addWeighted(frame, alpha, bilate, beta, gamma)
    
    if ret == False:
        break
    
    classes, scores, boxes = model.detect(sharpen,Conf_threshold, NMS_threshold)
    for (classid, score, box) in zip(classes, scores, boxes):
        
        classes = np.argmax(scores) 
        confidence = scores[classes]
        
        count=1

        color = COLORS[int(classid)%len(COLORS)]
        label = "%s : %f" % (class_name[classid[0]], (score*100))+"%"
        
#   Bounding Box Design

        cv.rectangle(sharpen,box,color,1)
        cv.putText(sharpen, label, (box[0], box[1]-10), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.45,color, 2)
        
    
#   Save picture with No mask
    
    if classid ==0:
        cv.putText(sharpen,'WEAR MASK!', (box[0], box[1]-30), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.45,color, 2)
        cv.rectangle(sharpen,box,color,1)
        cv.putText(sharpen, label, (box[0], box[1]-10), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.45,color, 2)
        t = time.strftime("%Y-%m-%d_%H-%M-%S")
        print("Image "+t+" saved"+" No Mask")
        
        #   Change path according to location
        
        path=r'C:\Users\rinki\OneDrive\Desktop\NMask\Image '+str(t)+" #"+str(count)+'.jpg'
        cv.imwrite(path,sharpen)
        count+=10
        if count<=0:
            break

    cv.imshow('Facemask Type Detection (DSP Project)', sharpen)
    
    key = cv.waitKey(1)

#   To exit our project, Press 'Q' sa keyboard.
    if key == ord('q'):
        break
    
print("Facemask Type Detector Closed")

cap.release()
cv.destroyAllWindows()
