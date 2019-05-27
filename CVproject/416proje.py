# -*- coding: utf-8 -*-
"""
Created on Wed May  8 17:47:37 2019

@author: Enes
"""

import torch
import cv2
import numpy as np
import cv2
import tensorflow as tf
model = torch.load("trained_resnet")


def predict_image(frame, model):
    with torch.no_grad():
        
#         frame = frame.to(device)
        frame = torch.tensor([frame])
       
        
        inputs = frame.unsqueeze(0)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        print(predicted)
        
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

img = cv2.imread("a.jpeg")
img = cv2.flip(img,1)
cv2.imshow("img", img)
predict_image(img, model)
'''cv2.waitKey(10000)
cv2.destroyAllWindows()'''
# cap = cv2.VideoCapture(0)

# while(True):
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#     if ret==True:
#         frame = cv2.flip(frame,1)
#         # Our operations on the frame come here
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         img = cv2.resize(frame,(64,64))
# #         predict_image(img,model)
#         # Display the resulting frame
#         cv2.imshow('frame',frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

# When everything done, release the capture
# cap.release()
# cv2.destroyAllWindows()

'''
import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret==True:
        frame = cv2.flip(frame,1)
        # Our operations on the frame come here
        

        # Display the resulting frame
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

'''




