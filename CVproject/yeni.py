# -*- coding: utf-8 -*-
"""
Created on Thu May  9 21:57:42 2019

@author: Enes
"""

import torch
import cv2
import numpy as np
import cv2
import torchvision
from torchvision import datasets, models, transforms
from torch.autograd import Variable


model = torch.load("trained_resnet")

from PIL import Image
normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
   transforms.Resize(256),
   transforms.CenterCrop(224),
   transforms.ToTensor(),
   normalize
   # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# An instance of your model.
img_pil = Image.open("a.png")
# img_pil.show()
img_tensor = preprocess(img_pil).float()

img_tensor = img_tensor.unsqueeze_(0)
inputs = Variable(img_tensor)
inputs = inputs.to(device)
model = model.to(device)
fc_out = model(inputs)
_, predicted = torch.max(fc_out, 1)
print(predicted)
'''
import cv2

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret==True:
        frame = cv2.flip(frame,1)
        pil_im = Image.fromarray(frame)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Our operations on the frame come here
        img_tensor = preprocess(pil_im).float()
        img_tensor = img_tensor.unsqueeze_(0)
        inputs = Variable(img_tensor)
        inputs = inputs.to(device)
        model = model.to(device)
        fc_out = model(inputs)
        _, predicted = torch.max(fc_out.data, 1)
        print(predicted)

        # Display the resulting frame
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
'''