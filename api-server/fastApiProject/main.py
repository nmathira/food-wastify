import io

import uvicorn
from PIL import Image
from fastapi import FastAPI, UploadFile

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
        image_bytes = await file.read()

        image = Image.open(io.BytesIO(image_bytes))
        # image.verify()  # Verify the image is not corrupted

        # Re-open the image to save it, as `verify()` makes the image unusable
        image = Image.open(io.BytesIO(image_bytes))

        # Generate a unique filename
        image_path = f"/home/niranjan/downloads/asdf/api-image-{hash(image)}.jpg"

        image.save(image_path)
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8080)
