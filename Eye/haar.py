from __future__ import print_function
import cv2 as cv
import numpy as np

def detectAndDisplay(frame):
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)
    #-- Detect eyes
    eyes = eyes_cascade.detectMultiScale(frame, 1.2, 10)
    for (x,y,w,h) in eyes:
        frame = cv.rectangle(frame, (x,y),(x+w, y+h), (255, 0,0), 5 )
        cropped_image = frame_gray[y:y+h, x:x+w]
        ret,glint_image = cv.threshold(cropped_image,200,255,cv.THRESH_BINARY)
        ret,pupil_image = cv.threshold(cropped_image,30,255,cv.THRESH_BINARY_INV)
        

        contours, hierarchy = cv.findContours(glint_image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        #cv.drawContours(glint_image, contours, -1, (0,255,0), 3)


        #convert all back to color
        pupil_color = cv.cvtColor(pupil_image,cv.COLOR_GRAY2RGB)
        glint_color = cv.cvtColor(glint_image,cv.COLOR_GRAY2RGB)
        cropped_color = cv.cvtColor(cropped_image,cv.COLOR_GRAY2RGB)
        

        #find the center of the pupil
        # calculate moments of binary image
        M = cv.moments(pupil_image)
        # calculate x,y coordinate of center
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0

        points_list = []

        #Mark the contours in the image 
        for c in contours:
            # calculate moments for each contour
            M = cv.moments(c)

            # calculate x,y coordinate of center
            if M["m00"] != 0:
                g_cX = int(M["m10"] / M["m00"])
                g_cY = int(M["m01"] / M["m00"])
                points_list.append([g_cX, g_cY])
            else:
                g_cX = 0
                g_cY = 0
            cv.circle(glint_color, (g_cX, g_cY), 5, (0, 0, 255), -1)

        print(points_list)

        # draw the centroid in the pupil image
        pupil_color = cv.circle(pupil_color, (cX, cY), radius=5, color=(0,0,255), thickness=-1)
        
        numpy_horizontal = np.hstack((cropped_color, glint_color, pupil_color))
        #cv.imshow('threshold lower', cropped_image_color)
        #cv.imshow('cropped', cropped_image)
        #cv.imshow('threshold upper', thresh1)
        #cv.imshow('threshold lower', thresh2)
        cv.imshow('threshold lower', numpy_horizontal)
        
    cv.imshow('Capture - Face detection', frame)

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
