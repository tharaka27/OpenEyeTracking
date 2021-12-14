from __future__ import print_function
import cv2 as cv

def detectAndDisplay(frame):
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)
    #-- Detect eyes
    eyes = eyes_cascade.detectMultiScale(frame, 1.2, 10)
    for (x,y,w,h) in eyes:
        frame = cv.rectangle(frame, (x,y),(x+w, y+h), (255, 0,0), 5 )
        cropped_image = frame_gray[y:y+h, x:x+w]
        cv.imshow('cropped', cropped_image)
        
    cv.imshow('Capture - Face detection', frame)

face_cascade = cv.CascadeClassifier()
eyes_cascade = cv.CascadeClassifier('haarcascade_eye.xml')

cap = cv.VideoCapture(0)
if not cap.isOpened:
    print('--(!)Error opening video capture')
    exit(0)
while True:
    ret, frame = cap.read()
    print("Capturing")
    if frame is None:
        print('--(!) No captured frame -- Break!')
        break
    detectAndDisplay(frame)
    if cv.waitKey(10) == 27:
        break
