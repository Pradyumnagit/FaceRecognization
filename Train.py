import os;
import cv2;
import numpy as np;
from PIL import Image;

recognizer=cv2.createLBPHFaceRecognizer();
path="Images";
def getPath(path):
    img_path=[os.path.join(path,f) for f in os.listdir(path)]
    print(img_path);
    faces=[];
    IDs=[];
    for i in img_path:
        faceimg=Image.open(i).convert('L');
        facenp=np.array(faceimg,'uint8');
        ID=int(os.path.split(i)[-1].split('.')[1]);
        faces.append(facenp);
        print(ID);
        IDs.append(ID);
        cv2.imshow("Screen",facenp);
        cv2.waitKey(10);
    return IDs,faces
        
ids,faces=getPath(path);
recognizer.train(faces,np.array(ids));
recognizer.save('recognizer/train.yml');
cv2.destroyAllWindows();
