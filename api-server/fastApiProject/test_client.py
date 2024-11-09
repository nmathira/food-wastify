import io

import cv2
import os
import requests
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")
camera = cv2.VideoCapture(0)
frame_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
while True:
    data = camera.read()
    if data[0]:
        cv2.imshow("asdf",data[1])
        cv2.waitKey(0)
        request = requests.post("localhost", data=io.BytesIO(data[1]))
    cv2.destroyAllWindows()
