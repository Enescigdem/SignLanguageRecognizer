import torch
import torchvision
#import tensorflow as tf
from torchvision import datasets, models, transforms
from torch.autograd import Variable
import numpy as np
import cv2
from matplotlib import cm
import keyboard
import os
cap = cv2.VideoCapture(0)
extraX = 0
extraY=0
model=torch.load('trained_50')
line_width=2
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


data_transforms = {
    
    'test': transforms.Compose([
            
            transforms.Resize(256),
            transforms.CenterCrop(224),
           
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
}

dataset_directory = 'test'
image_datasets = {x: datasets.ImageFolder(os.path.join(dataset_directory, x),data_transforms[x])for x in ['test']}
  #Batch size is set as 64 
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1,
                       shuffle=True, num_workers=0)
for x in ['test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['test']}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print({x: len(image_datasets[x]) for x in ['test']})
class_names = ['b','d','k','s']
model.eval()
model.to(device)

current_phase_correct_outputnumber = 0
x =[]
topk=0
with torch.no_grad():
    print("gardda")
    inputs, classes  = iter(dataloaders['test']).next()
    print("forda")
    inputs = inputs.to(device)
    classes = classes.to(device)
    print("burda")
    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)
    probabilities,labels = outputs.topk(5,dim=1)
    
    classes_size = labels.size(0)
    for p in range(classes_size):
      if classes[p] in labels[p]:
        topk+=1
    current_phase_correct_outputnumber += torch.sum(preds == classes.data)
x.append(preds.double())
epoch_acc = current_phase_correct_outputnumber.double() / dataset_sizes['test']
topk_acc = topk / dataset_sizes['test']
print(epoch_acc)
'''
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Our operations on the frame come here
    if ret==True:
       
        # Display the resulting frame
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if keyboard.is_pressed('g'):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if keyboard.is_pressed('c'):
            frame = cv2.Canny(frame, 100, 200)
        if keyboard.is_pressed('d'):
            extraX += 1
        if keyboard.is_pressed('a'):
            extraX -= 1
        if keyboard.is_pressed('w'):
            extraY-=1
        if keyboard.is_pressed('s'):
            extraY += 1
        if(ret==False):
            exit()
        #frame = cv2.resize(frame, (int8000), int(800)))

        x = int(len(frame[0])/2)
        y = int(len(frame[1]) / 2)
        marginX=int(x/2)
        marginY=int(y/2)


        cv2.rectangle(frame, (x-marginX+extraX, y-marginY+extraY), (x+marginX+extraX, y+marginY+extraY), (255, 255, 255), line_width)



        cv2.imshow(str('frame'),frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            roi = frame[y-marginY+line_width:y+marginY, x-marginX:x+marginX]
            roi = cv2.flip(roi,1)
            pil_im = Image.fromarray(roi)
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
            cv2.imwrite(str(predicted)+'.png', roi)
            break




# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


'''