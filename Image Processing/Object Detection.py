import numpy as np
import cv2 as cv
import time
import sys

cap = cv.VideoCapture(1)

threshold = 10

count = 0

while True:
    ret, frame = cap.read()
    # convert to a cylindrical-coordinate color mapping system to isolate saturation
    HSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    std = np.std(HSV[:,:,1])
    if std > threshold:
        cv.imwrite("test.jpg", frame)
        print("image Saved")
        count += 1
    time.sleep(1)