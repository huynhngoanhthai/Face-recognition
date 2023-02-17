import cv2
import tensorflow as tf
import numpy as np

filename = 'temp/5.jpg'
image = cv2.imread(filename)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
save_model = tf.keras.models.load_model("khuonmat.h5")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

boxes = face_cascade.detectMultiScale(filename,1.3,5)

fontface = cv2.FONT_HERSHEY_SIMPLEX
for [x, y, w, h] in boxes:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_gray = cv2.resize(src=roi_gray, dsize=(100,100))
    roi_gray = roi_gray.reshape((100,100,1))
    roi_gray = np.array(roi_gray)
    result = save_model.predict(np.array([roi_gray]))  # type: ignore
    final = np.argmax(result)
    print(final)
    if(final >= 0):
        if final == 0:
            cv2.putText(image, "Ngan",(x+10,y+h+ 30), fontface, 1, (0,255,0),2)
        elif final == 1:
            cv2.putText(image, "Trang",(x+10,y+h+ 30), fontface, 1, (0,255,0),2)
        elif final == 2:
            cv2.putText(image, "Thoa",(x+10,y+h+ 30), fontface, 1, (0,255,0),2)
        elif final == 3:
            cv2.putText(image, "Tu",(x+10,y+h+ 30), fontface, 1, (0,255,0),2)
        elif final == 4:
            cv2.putText(image, "Vy",(x+10,y+h+ 30), fontface, 1, (0,255,0),2)
        elif final ==5:
            cv2.putText(image, "Thai",(x+10,y+h+ 30), fontface, 1, (0,255,0),2)
            
    else:
        cv2.putText(image, "Unknow",(x+10,y+h+ 30), fontface, 1, (0,255,0),2)
cv2.imshow('trainning',cv2.resize(src=image,dsize=(1000,1000)))
cv2.imwrite('test.jpg',image)
print(result) # type: ignore
cv2.waitKey(0)
cv2.destroyAllWindows()

def face_recognition(filename):
    image = cv2.imread(filename)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
    save_model = tf.keras.models.load_model("C:/Users/karic/OneDrive/Desktop/AnhThai/XLA/khuonmat.h5")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fontface = cv2.FONT_HERSHEY_SIMPLEX
    boxes = cv2.face_detect(filename)
    for box in boxes:
        [x, y, w, h] = box['box']
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(src=roi_gray, dsize=(100,100))
        roi_gray = roi_gray.reshape((100,100,1))
        roi_gray = np.array(roi_gray)
        result = save_model.predict(np.array([roi_gray]))  # type: ignore
        if(result[0][0] != 1):
            cv2.putText(image, "Unknow",(x+10,y+h+ 30), fontface, 1, (0,255,0),2)
        else:
            final = np.argmax(result)
            print(final)
            if(final >= 0):
                if final == 0:
                    cv2.putText(image, "Ngan",(x+10,y+h+ 30), fontface, 1, (0,255,0),2)
                elif final == 1:
                    cv2.putText(image, "Trang",(x+10,y+h+ 30), fontface, 1, (0,255,0),2)
                elif final == 2:
                    cv2.putText(image, "Thoa",(x+10,y+h+ 30), fontface, 1, (0,255,0),2)
                elif final == 3:
                    cv2.putText(image, "Tu",(x+10,y+h+ 30), fontface, 1, (0,255,0),2)
                elif final == 4:
                    cv2.putText(image, "Vy",(x+10,y+h+ 30), fontface, 1, (0,255,0),2)
                elif final ==5:
                    cv2.putText(image, "Thai",(x+10,y+h+ 30), fontface, 1, (0,255,0),2)
                    
            else:
                cv2.putText(image, "Unknow",(x+10,y+h+ 30), fontface, 1, (0,255,0),2)
    cv2.imshow('trainning',cv2.resize(src=image,dsize=(1000,1000)))
