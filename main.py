from ctypes.wintypes import MSG
from multiprocessing.connection import wait
import os
from dotenv import load_dotenv

load_dotenv()
from PyQt5 import QtGui, QtWidgets, QtCore
from PyQt5.QtWidgets import QFileDialog,QMessageBox
from ns import remove_noisy
from edge import edge_detection
from displayv2 import face_recognition
import sys, threading, winsound
import view.home 

import cv2

ui = ''
app = QtWidgets.QApplication(sys.argv)
MainWindow = QtWidgets.QMainWindow()
face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')

def mainUi():
    global ui
    ui = view.home.Ui_Form() 
    ui.setupUi(MainWindow)

    if 'uxJEDrDV3M'==os.getenv('KEY'):
        ui.widget.hide()
    
    ui.pushButton_noise.clicked.connect(clickedButtonNoise)
    ui.pushButton_edge.clicked.connect(clickedButtonEdge)
    ui.pushButton_detect.clicked.connect(clickedButtonDetect)
    ui.pushButton_recognition.clicked.connect(clickedButtonRecognition)
    ui.pushButton_exit.clicked.connect(app.quit)
    MainWindow.show()

def clickedButtonNoise():
    filename = QFileDialog.getOpenFileName(QFileDialog(),"loai bo noise","C:/Users/karic/OneDrive/Desktop/AnhThai/XLA/setupData")
    if filename[0] == '':
        return
    elif filename[0][-3:] != "jpg" and filename[0][-3:] != "png":
        msg = QMessageBox()
        msg.setText("Error")
        msg.setInformativeText('file must .jpg')
        msg.setWindowTitle("Error")
        msg.exec_()
        print(filename[0][-3:])
        return
    cv2.imshow("median.jpg",cv2.resize(remove_noisy(filename[0]),dsize=(1000,1000)))
# Nhận diện khuôn mặt trong ảnh
def clickedButtonEdge():
    filename = QFileDialog.getOpenFileName(QFileDialog(),"loai bo noise","C:/Users/karic/OneDrive/Desktop/AnhThai/XLA/setupData")
    if filename[0] == '':
        return
    elif filename[0][-3:] != "jpg" and filename[0][-3:] != "png":
        msg = QMessageBox()
        msg.setText("Error")
        msg.setInformativeText('file must .jpg')
        msg.setWindowTitle("Error")
        msg.exec_()
        print(filename[0][-3:])
        return
    edge_detection(filename[0])

def  clickedButtonDetect():
    filename = QFileDialog.getOpenFileName(QFileDialog(),"xac dinh khuong mat","C:/Users/karic/OneDrive/Desktop/AnhThai/XLA/temp")
    if filename[0] == '':
        return
    elif filename[0][-3:] != "jpg"  and filename[0][-3:] != "png":
        msg = QMessageBox()
        msg.setText("Error")
        msg.setInformativeText('file must .jpg')
        msg.setWindowTitle("Error")
        msg.exec_()
        return
    img=remove_noisy(filename[0])
    fa = face_cascade.detectMultiScale(
    img,
    scaleFactor=1.3,
    minNeighbors=5,
    minSize=(80, 80),)
    for  [x, y, w, h] in fa:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imshow("face_detect.jgp",cv2.resize(img,dsize=(1000,1000)))
def clickedButtonRecognition():
    filename = QFileDialog.getOpenFileName(QFileDialog(),"xac dinh khuong mat","C:/Users/karic/OneDrive/Desktop/AnhThai/XLA/temp")
    if filename[0] == '':
        return
    elif filename[0][-3:] != "jpg"  and filename[0][-3:] != "png":
        msg = QMessageBox()
        msg.setText("Error")
        msg.setInformativeText('file must .jpg')
        msg.setWindowTitle("Error")
        msg.exec_()
        return
    face_recognition(filename[0])

mainUi()
sys.exit(app.exec_())