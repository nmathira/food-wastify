import numpy as np
import cv2 as cv
import time

cap = cv.VideoCapture(1)

threshold = 25

count = 0

while True:
    ret, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    std = np.std(gray)
    if std > threshold:
        cv.imwrite("Image Processing\Images\frame" + str(count) + ".jpg",frame)
        print("image Saved")

        count += 1
    time.sleep(1)