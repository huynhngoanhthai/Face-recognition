from itertools import count
import cv2
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import learning_curve
data = [] 
label = []
x=10000
y=6
count = 0
for j in range (1,y+1):
  for i in range (1,x):
    filename = './dataset/anh'+ str(j) + '.'  + str(i) + '.jpg'
    if not os.path.exists(filename):
      break
    Img = cv2.imread(filename) 
    Img = cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)
    Img = cv2.resize(Img,dsize=(100,100))
    Img = np.array(Img)
    data.append(Img)
    label.append(j-1)
    count+=1
data1 = np.array(data)
label = np.array(label)
data1 = data1.reshape((count,100,100,1))

X_train = data1/255.0
from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer()
trainY = lb.fit_transform(label)
from keras.models import Model
from keras.models import Sequential
from keras.layers import AveragePooling2D
from keras.layers import MaxPooling2D
from keras.layers import Conv2D
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import Dense
from keras.layers import concatenate
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
Model = Sequential()
shape = (100,100, 1)
Model.add(Conv2D(32,(3,3),padding="same",input_shape=shape))
Model.add(Activation("relu"))
Model.add(Conv2D(32,(3,3), padding="same"))
Model.add(Activation("relu"))
Model.add(MaxPooling2D(pool_size=(2,2)))
Model.add(Conv2D(64,(3,3), padding="same"))
Model.add(Activation("relu"))
Model.add(MaxPooling2D(pool_size=(2,2)))
Model.add(Flatten())
Model.add(Dense(512))
Model.add(Activation("relu"))
Model.add(Dense(y))
Model.add(Activation("softmax"))
Model.summary()
Model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
print("start training")
Model.fit(X_train,trainY,batch_size=5,epochs=10)
Model.save("khuonmat.h5")
