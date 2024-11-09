import cv2

camera = cv2.VideoCapture(0)
frame_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

data = camera.read()
if data[0]:
    cv2.imshow("asdf",data[1])

