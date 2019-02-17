
import cv2;
import numpy as np;
import os;
facedetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
eyedetect=cv2.CascadeClassifier('haarcascade_eye.xml');
camera=cv2.VideoCapture(0);
id=raw_input('Enter your name: ');
images=0;
while(True):
    bool,img=camera.read();
    print(camera.read());
    img=cv2.imread("virat.jpg");
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY);
    faces=facedetect.detectMultiScale(gray,1.5,5);
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),4);
        images=images+1;
        cv2.imwrite("Images/User."+str(id)+"."+str(images)+".jpg",gray[y:y+h,x:x+w]);
        #print(faces);
        #print(x);
        #print(y);
        #print(w);
        #print(h);
        eye_gray = gray[y:y+h, x:x+w]
        eye_color = img[y:y+h, x:x+w]
        eyes = eyedetect.detectMultiScale(eye_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(eye_color,(ex,ey),(ex+ew,ey+eh),(0,0,255),2);
             #print(eyes);
             #print(ex);
             #print(ey);
             #print(ew);
             #print(eh);
    cv2.imshow("Your Faces",img);
    cv2.waitKey(1);
    if(images>=20):
        break;
    
camera.release();
cv2.destroyAllWindows();
