import cv2
import numpy as np

eye = cv2.CascadeClassifier('haarcascade_eye.xml')

#img = cv2.imread("test2.jpg")
img = cv2.imread("After_filter.jpg")
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

eyes = eye.detectMultiScale(gray_image, 1.2, 10)
for (x,y,w,h) in eyes:
    img = cv2.rectangle(img, (x,y),(x+w, y+h), (255, 0,0), 5 )
    cropped_image = gray_image[y:y+h, x:x+w]

ret,thresh1 = cv2.threshold(cropped_image,200,255,cv2.THRESH_BINARY)
#ret,thresh1 = cv2.threshold(cropped_image,60,255,cv2.THRESH_BINARY_INV)

#cv2.imshow("baic", img)
cv2.imshow("canny", thresh1)
cv2.imshow("Image2", cropped_image)
cv2.waitKey(0)
