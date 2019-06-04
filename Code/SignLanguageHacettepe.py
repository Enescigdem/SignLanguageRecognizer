import cv2
import numpy as np
import keyboard
import ctypes
import time
from torchvision import transforms,models
import torch
from PIL import Image
from torch.autograd import Variable


def nothing(x):
    pass


class SignLanguageRecognizer:
    def __init__(self,torch_model=None,prediction_interval=0.5,save_key='s',quit_key='q',kernel_size=3):
        self.kernel_size=kernel_size
        self.torch_model = torch_model
        self.prediction_interval=prediction_interval
        self.save_key=save_key
        self.quit_key=quit_key

    def predict(self,img):
        data_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.torch_model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        img=cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        img_pil = Image.fromarray(img)

        img_tensor = data_transforms(img_pil).float()
        img_tensor = img_tensor.unsqueeze_(0)
        inputs = Variable(img_tensor)
        inputs = inputs.to(device)
        self.torch_model = self.torch_model.to(device)
        fc_out = self.torch_model(inputs)
        _, predicted = torch.max(fc_out.data, 1)
        
        return predicted


    def Track(self,save_dir, human_letter):

        """Create Video Capture Screen"""
        cap = cv2.VideoCapture(0)
        ret = cap.set(3, 860)
        ret = cap.set(4, 620)

        kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
        """Get Screen Size to adjust panels"""
        user32 = ctypes.windll.user32
        user32.SetProcessDPIAware()
        width, height = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)


        """Create Windows"""
        cv2.namedWindow('frame', cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow('frame', 0, 10)
        cv2.namedWindow('res', cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow('res', int(width / 2), 10)
        cv2.namedWindow('information', cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow('information', int(width / 2), int(height / 3 + height / 4 - 22))
        cv2.namedWindow('detailed', cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow('detailed', int(width / 2), int(height / 3))



        """Create trackbars for color change"""
        cv2.createTrackbar('LowerR', 'information', 0, 255, nothing)
        cv2.createTrackbar('LowerG', 'information', 50, 255, nothing)
        cv2.createTrackbar('LowerB', 'information', 50, 255, nothing)
        cv2.createTrackbar('UpperR', 'information', 155, 255, nothing)
        cv2.createTrackbar('UpperG', 'information', 255, 255, nothing)
        cv2.createTrackbar('UpperB', 'information', 255, 255, nothing)

        """Create switch for ON/OFF functionality"""
        switch = '0 : OFF \n1 : ON'
        cv2.createTrackbar(switch, 'information', 0, 1, nothing)



        """define range of red color in HSV (red is [180, 0, 0])"""
        lower_red = np.array([0, 50, 50])
        upper_red = np.array([155, 255, 255])



        color = (102, 0, 102)  # rgb
        white = (255, 225, 225)

        rect_margin = 30
        key = "c"
        predicted = 0
        while cv2.waitKey(24) & 0xFF != ord(self.quit_key):
            ret, frame = cap.read()
            """Convert to HSV"""
            lr = cv2.getTrackbarPos('LowerR', 'information')
            lg = cv2.getTrackbarPos('LowerG', 'information')
            lb = cv2.getTrackbarPos('LowerB', 'information')
            ur = cv2.getTrackbarPos('UpperR', 'information')
            ug = cv2.getTrackbarPos('UpperG', 'information')
            ub = cv2.getTrackbarPos('UpperB', 'information')
            s = cv2.getTrackbarPos(switch, 'information')

            if s == 0:
                lower_red = np.array([0, 50, 50])
                upper_red = np.array([155, 255, 255])

            else:
                lower_red = np.array([lr, lg, lb])
                upper_red = np.array([ur, ug, ub])



            frame = np.flip(frame, axis=1)
            """Draw a rectangle."""
            im = np.copy(frame)
            cv2.rectangle(im, (int(width / 2) - 612, 30), (int(width / 2) - 388, 230), color, 0)


            rect = cv2.rectangle(im, (int(width / 2) - 612, 15), (int(width / 2) - 388, 29), color,
                                 cv2.FILLED)  # background of prediction
            font = cv2.FONT_HERSHEY_SIMPLEX
            TopLeftCorner = (int(width / 2) - 612, 28)
            fontScale = 0.5
            fontColor = white
            lineType = 1

            cv2.putText(rect, str(human_letter[predicted]),
                        TopLeftCorner,
                        font,
                        fontScale,
                        fontColor,
                        lineType)

            cropped = im[30:230, int(width / 2) - 612:int(width / 2) - 388]

            """Threshold the HSV image to get only blue colors"""
            hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
            """Bitwise-AND mask and original image"""
            mask = cv2.inRange(hsv, lower_red, upper_red)

            mask_dilate= cv2.dilate(mask,kernel, iterations=3)
            res = cv2.bitwise_and(cropped, cropped, mask=mask_dilate)

            """Change the background to white"""
            res[np.all(res == [0, 0, 0], axis=2)] = [210, 210, 210]

            """Convert image to grayscale"""
            rgbRes=res
            res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
            res = np.flip(res, axis=1)
            # construct black screen
            information = np.zeros((int(width / 2), int(height / 3), 3), np.uint8)
            detailed = np.zeros((int(width / 2), int(height / 4), 3), np.uint8)

            im = cv2.resize(im, (int(width / 2), height))
            information = cv2.resize(information, (int(width / 2), int(height / 3)))
            detailed = cv2.resize(detailed, (int(width / 2), int(height / 5)))

            res = cv2.blur(res, (3, 3))
            predicted=self.predict(res)


            if keyboard.is_pressed(self.save_key):
                img = cv2.resize(res, (224, 224))
                cv2.imwrite(save_dir+"/saved_images" + str(predicted) + ".png", img)
            """Change Visual Screen"""
            if keyboard.is_pressed('b'):  # if key 'b' is pressed
                key = "b"
            if keyboard.is_pressed('c'):  # if key 'c' is pressed
                key = "c"
            if keyboard.is_pressed('h'):  # if key 'h' is pressed
                key = "h"

            # check key value
            if key == "b":
                mask_dilate = np.flip(mask_dilate, axis=1)
                mask_dilate = cv2.resize(mask_dilate, (int(width / 2), int(height / 3)))
                cv2.imshow('res', mask_dilate)
            if key == "c":
                rgbRes = cv2.blur(rgbRes, (3, 3))
                rgbRes = cv2.resize(rgbRes, (int(width / 2), int(height / 3)))
                cv2.imshow('res', rgbRes)
            if key == "h":
                # change to hog
                k_size = 3
                hog = np.float32(res) / 255.0
                gx = cv2.Sobel(hog, cv2.CV_64F, 1, 0, ksize=k_size)
                gy = cv2.Sobel(hog, cv2.CV_64F, 0, 1, ksize=k_size)
                mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
                mag = cv2.resize(mag, (int(width / 2), int(height / 3)))
                cv2.imshow('res', mag)


            # organize information window
            cv2.putText(detailed, 'suggested lowerRGB = [0, 50, 50] ; upperRGB = [155, 255, 225] in low-light environment',
                        (10, 20), font, fontScale, (255, 196, 255), 1)
            cv2.putText(detailed, 'suggested lowerRGB = [0, 31, 2] ; upperRGB = [255, 255, 225] in multi-light environment',
                        (10, 40), font, fontScale, (255, 196, 255), 1)
            cv2.putText(detailed, 'To see segmented hand press "c" from keyboard', (10, 60), font, fontScale, (255, 196, 255),
                        1)
            cv2.putText(detailed, 'To see binary form of hand press "b" from keyboard', (10, 80), font, fontScale,
                        (255, 196, 255), 1)
            cv2.putText(detailed, 'To see hog form of hand press "h" from keyboard', (10, 100), font, fontScale, (255, 196, 255), 1)
            cv2.putText(detailed, 'Press "s" from keyboard to save a frame', (10, 120), font, fontScale, (255, 196, 255), 1)
            cv2.imshow('frame', im)
            cv2.imshow('information', information)
            cv2.imshow('detailed', detailed)

            time.sleep(self.prediction_interval)
        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()