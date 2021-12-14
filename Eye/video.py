"""
Demonstration of the GazeTracking library.
Check the README.md for complete documentation.
"""

import cv2
from gaze_tracking import GazeTracking

gaze = GazeTracking()
webcam = cv2.VideoCapture(0)
eye = cv2.CascadeClassifier('haarcascade_eye.xml')
#webcam = cv2.VideoCapture("eye.mp4")


while True:
    # We get a new frame from the webcam
    _, frame = webcam.read()
    cv2.imshow("original", frame)

    
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    eyes = eye.detectMultiScale(frame, 1.2, 10)
    
    for (x,y,w,h) in eyes:
        frame = cv2.rectangle(frame, (x,y),(x+w, y+h), (255, 0,0), 5 )
        cropped_image = gray_image[y:y+h, x:x+w]


    if cropped_image is None:
        continue
    else:
        cv2.imshow("Demo", cropped_image)
    
    if cv2.waitKey(1) == 27:
        break
   
webcam.release()
cv2.destroyAllWindows()
