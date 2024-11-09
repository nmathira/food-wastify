import io

from fastapi import FastAPI, UploadFile
from PIL import Image

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}


@app.post("/upload-food")
async def upload_image(file: UploadFile):
    try:
        image = Image.open(io.BytesIO(await file.read()))
        image.verify()
        image.save("/home/niranjan/downloads/asdf/api-image"+str(hash(image)))
        return {"message": "yipeee the image verified lol"}
    except Exception as e:
        return {"error": str(e)}
