import cv2
from ns import remove_noisy
def edge_detection(filename):
    img = remove_noisy(filename)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (3,3), 0) 
    edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)
    cv2.imshow('Edge Detection', cv2.resize(edges,(1000,1000)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()