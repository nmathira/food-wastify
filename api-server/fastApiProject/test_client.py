import io

import cv2
import requests

camera = cv2.VideoCapture(0)
while True:
    data = camera.read()
    if data[0]:
        cv2.imshow("asdf", data[1])
        cv2.waitKey(0)
        _, encoded = cv2.imencode(".jpg", data[1])
        request = requests.post("localhost", files={"file": ("image.jpg", io.BytesIO(encoded[1]), "image/jpeg")})
    cv2.destroyAllWindows()
