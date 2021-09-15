from picamera import PiCamera
from time import sleep
import math
import cv2
import numpy as np

camera = PiCamera()
camera.resolution = (1920,1080)
camera.start_preview()
camera.capture('/home/pi/Desktop/img.jpg')
sleep(5)
camera.stop_preview()

img = cv2.imread('/home/pi/Desktop/img.jpg')
scaling_factor = 0.7