import numpy as np
import cv2 as cv
import time
import sys

cap = cv.VideoCapture(1)

std_threshold = 10

count = 0

ctime = time.time()
while time.time() < ctime + 2:
    cap.read()

print('timeout past')

ret, last = cap.read()

in_motion = False

while True:
    ret, frame = cap.read()
    # convert to a cylindrical-coordinate color mapping system to isolate saturation
    HSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    std = np.std(HSV[:,:,1])
    diff = np.mean(cv.absdiff(last, frame))

    if diff >= 0.5:
        in_motion = True

    if in_motion and diff < 0.5:
        if std > std_threshold:
            cv.imwrite("test.jpg",frame)
            print('Image saved')
        in_motion = False
    
    last = frame
    time.sleep(0.2)