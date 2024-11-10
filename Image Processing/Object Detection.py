import numpy as np
import cv2 as cv
import time
import requests

cam = cv.VideoCapture(1)

# standard deviatoin of saturation threshold for food detection
std_threshold = 10

# initialize the "last" frame
ret, last = cam.read()

# no initial motion
in_motion = False

# to ignore the first few frames where AWB/exposure take place.
counter = 0

while True:
    ret, frame = cam.read()

    # convert to a cylindrical-coordinate color mapping to isolate saturation
    HSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    std = np.std(HSV[:,:,1])
    # Calculate difference or "motion" between current and last frames
    motion = np.mean(cv.absdiff(last, frame))

    if counter > 15:

        if motion >= 10:
            in_motion = True

        if in_motion and motion < 1:
            if std > std_threshold:
                cv.imwrite("test.jpg",frame)

                # Uncomment to send to API

                # _, encoded = cv.imencode(".jpg", frame)
                # request = requests.post("http://192.168.137.1:8080//upload-food", files={"file": ("image.jpg", io.BytesIO(encoded), "image/jpeg")})
                
                print('Image saved')
                in_motion = False
    
    last = frame
    counter += 1

    time.sleep(0.2)