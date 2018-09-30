#GenderDetect

import cv2 as cv
from numpy import *
from keras.models import load_model

FACE=cv.CascadeClassifier('Cascades/Haar/haarcascade_frontalface_default.xml')
vid=cv.VideoCapture(0)

model=load_model('Gender.h5')

def determine(img,model): #0: Male, 1: Female
    gender=model.predict(img.reshape((-1,32,32,3)))[0]
    return int(gender[1]==max(gender))

while 1:
    foo,img=vid.read()
    gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    faces=FACE.detectMultiScale(gray,1.3,5)
    for i in faces:
        if i[2]==i[3]:
            try:
                face=img[max(i[0]-50,0):min(i[0]+i[2]+50,img.shape[0]),max(i[1]-50,0):min(i[1]+i[3]+50,img.shape[1])]
                face=cv.resize(face,(32,32))
                gender=determine(face,model)
                cv.rectangle(img,(i[0]-50,i[1]-50),(i[0]+i[2]+50,i[1]+i[3]+50),(255,0,0) if gender==0 else (0,0,255),3)
            except Exception:
                pass

    cv.imshow('img',img)
    if cv.waitKey(1)&0xff==27:
        break

vid.release()
cv.destroyAllWindows()
print('End')
