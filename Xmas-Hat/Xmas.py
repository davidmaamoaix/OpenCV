#Xmas

import cv2 as cv
from numpy import *

FACE=cv.CascadeClassifier('Cascades/Haar/haarcascade_frontalface_default.xml')
HAT=cv.imread('files/hat.png',-1)
vid=cv.VideoCapture(0)
OFFSET=[25,50]

while 1:
    foo,img=vid.read()
    gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    faces=FACE.detectMultiScale(gray,1.3,5)
    for x,y,w,h in faces:
        new_hat=cv.resize(HAT,(int(w),int(h)))
        alpha=new_hat[:,:,3]/255
        imgAlpha=1-alpha
        
        try:
            for i in range(3):
                temp_hat=(alpha*new_hat[:,:,i]+imgAlpha*img[y-int(h)+OFFSET[1]:y+OFFSET[1],x+OFFSET[0]:x+int(w)+OFFSET[0],i])
                img[y-int(h)+OFFSET[1]:y+OFFSET[1],x+OFFSET[0]:x+int(w)+OFFSET[0],i]=temp_hat
        except ValueError as e:
            pass
    
    cv.imshow('img',img)
    if cv.waitKey(1)&0xff==27:
        break

vid.release()
cv.destroyAllWindows()
print('End')
