from picamera import PiCamera
from time import sleep

camera = PiCamera()
camera.resolution = (1920,1080)
camera.image_effect = 'denoise'
#camera.rotation = 60
camera.framerate = 30
camera.start_preview()
camera.start_recording('/home/pi/Desktop/video.h264')
sleep(10)
camera.stop_recording()
camera.stop_preview()