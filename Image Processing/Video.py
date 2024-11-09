import cv2 as cv
import numpy as np


# This file generates a video window so that the camera can be properly positioned.

# Press the 'q' key to exit stream!

camera = cv.VideoCapture(1)

while True:
    ret, frame = camera.read()
    cv.imshow('camera', frame)
    if cv.waitKey(1) == ord('q'):
        break