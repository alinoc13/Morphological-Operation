import cv2
import numpy as np
from myFunctions import imageFill

img = cv2.imread("Resources/Butterfly.jpg")
imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

imgBlur = cv2.GaussianBlur(imgGray,(3,3),0)
imgSobelx = cv2.Sobel(imgBlur,-1,dx=1, dy=0, ksize=3)
imgSobely = cv2.Sobel(imgBlur,-1,dx=0, dy=1, ksize=3)
imgSobel = imgSobelx + imgSobely
binImg = imgSobel.copy()
status = cv2.imwrite("Resources/bmp image.bmp",binImg)
print("Image written to file system:",status)

kernel = np.ones((3,3),np.uint8)

#source = https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
imgErode = cv2.erode(imgSobel,kernel,iterations=1)
imgDilate = cv2.dilate(imgSobel,kernel,iterations=1)
imgOpening = cv2.morphologyEx(imgSobel,cv2.MORPH_OPEN,kernel)
imgClosing = cv2.morphologyEx(imgSobel,cv2.MORPH_CLOSE,kernel)

#source https://learnopencv.com/filling-holes-in-an-image-using-opencv-python-c/
imgFill = imageFill(imgSobel)

cv2.imshow("Eroded image",imgErode)
cv2.waitKey(0)

