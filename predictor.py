import os;
import cv2;
import numpy as np;
from PIL import Image;

recognizer=cv2.createLBPHFaceRecognizer();

facedetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
eyedetect=cv2.CascadeClassifier('haarcascade_eye.xml');
camera=cv2.VideoCapture(0);
recognizer.load('train.yml');
id=0;
font=cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_COMPLEX_SMALL,5,1,0,4);
while(True):
    boo,img=camera.read();
    print(camera.read());
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY);
    faces=facedetect.detectMultiScale(gray,1.5,5);
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),4);
        id,config=recognizer.predict(gray[y:y+h,x:x+w]);
        cv2.cv.PutText(cv2.cv.fromarray(img),str(id),(x,y+h),font,255);
    cv2.imshow("Your Faces",img);
    if(cv2.waitKey(1) == ord('q')):
        break;
camera.release();
cv2.destroyAllWindows();
