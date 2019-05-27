import cv2
import numpy as np

cap = cv2.VideoCapture(0)

# capture frame-by-frame
# set Width
ret = cap.set(3, 160)
# set Height
ret = cap.set(4, 120)
cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
cv2.namedWindow('res', cv2.WINDOW_NORMAL)

# define range of red color in HSV (red is [180, 0, 0])
lower_red = np.array([0, 50, 50])
upper_red = np.array([15, 255, 255])


kernel = np.ones((3,3),np.uint8)

while(cv2.waitKey(24) & 0xFF != ord('q')): # Take each frame 
    ret, frame = cap.read() # Convert BGR to HSV 
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # Threshold the HSV image to get only blue colors 
    mask = cv2.inRange(hsv, lower_red, upper_red) # Bitwise-AND mask and original image 
     
    
    
    maskd= cv2.dilate(mask,kernel,iterations = 3)
    res = cv2.bitwise_and(frame, frame, mask=maskd) 
    
    cv2.imshow('frame', frame) 
    cv2.imshow('mask', maskd) 
    cv2.imshow('res', res)
    
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()