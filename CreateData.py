import cv2
import os
from ns import remove_noisy


detector=cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
print(0)
j=1
for i in range(6,7):
    while True:
        filename = 'setupData/anh.'  + str(i) + ' ' +"("+str(j)+")" + '.jpg'
        if not os.path.exists(filename): 
            break
        frame = remove_noisy(filename)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fa = cv2.detectMultiScale(filename)
        for(x,y,w,h) in fa:
                cv2.rectangle(frame,(x,y),(x+w, y+h),(0,255,0), 2)
                if not os.path.exists('dataset'):
                    os.makedirs('dataset')
                cv2.imwrite('dataset/anh'  + str(i) + '.' +str(j) + '.jpg', gray[y:y+h,x:x+w])


