import requests

files={'file': open('test.jpg','rb')}
request = requests.post("http://192.168.137.1:8080//upload-food", files=files)