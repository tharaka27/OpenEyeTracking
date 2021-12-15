from __future__ import print_function
import cv2 as cv
import numpy as np

def findFourNearestPointsToCenter(points, size):
    center_x = size[0]//2
    center_y = size[1]//2
    points.append([0,0])
    points.append([0,0])
    points.append([0,0])
    points.append([0,0])

    #print(size)
    #print("center ", center_x, center_y)

    points.sort(key = lambda x: (x[0]- center_x)**2 + (x[1]- center_y)**2)
    return points[:4]
    

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
            #cv.circle(glint_color, (g_cX, g_cY), 5, (0, 0, 255), -1)

        #print(points_list)
        nearest = findFourNearestPointsToCenter(points_list, glint_color.shape)
        print("nearest -> ",nearest)
        print("eye point ->", cX, cY)
        for g_cX,g_cY in nearest:
            cv.circle(glint_color, (g_cX, g_cY), 5, (0, 0, 255), -1)
        
        t_r_x , t_r_y = nearest[0][0], nearest[0][1]
        t_l_x , t_l_y = nearest[3][0], nearest[3][1]
        b_r_x , b_r_y = nearest[1][0], nearest[1][1]
        b_l_x , b_l_y = nearest[2][0], nearest[2][1]

        x_1 = (t_l_x + b_l_x ) // 2
        x_2 = (t_r_x + b_r_x ) // 2

        y_1 = (t_l_y + t_r_y ) // 2
        y_2 = (b_l_y + b_r_y ) // 2

        if( x_2 != x_1 and y_2 != y_1):
            estimate_x = int( (1920*cX)//(x_2 - x_1))
            estimate_y = int( (1024*cY)//(y_2 - y_1))
        else:
            estimate_x = 0
            estimate_y = 0
            


        if estimate_x > 0  and estimate_x < 1920:
             if estimate_y > 0  and estimate_y < 1024:
                 print("estimated x" , estimate_x , " estimated_y ", estimate_y)



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
