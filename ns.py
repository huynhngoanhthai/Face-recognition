
import cv2

def remove_noisy(filename):
    img = cv2.imread(filename)
    median = cv2.medianBlur(img,5)
    return median
