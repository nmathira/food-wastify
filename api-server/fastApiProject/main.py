import io

import uvicorn
from PIL import Image
from fastapi import FastAPI, UploadFile

app = FastAPI()

annotation_reference = {
    "meat": 0.3,
    "rice": 0.4,
    "noodle": 0.9,
    "vegetable": 0.2,
    "bread": 0.4
}

item_reference = ["meat", "noodle", "rice", "vegetable", "bread"]

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


# 4 0.6341974139213562 0.27568519115448 0.1594523787498474 0.03641393780708313
def calculate_waste_percent(annotations):
    areas = {}
    # for index in range(0, len(annotations)):
    for annotation in annotations:
        if annotation[0] not in areas:
            areas[annotation[0]] = 0

        area = annotation[3] * annotation[4]

        areas[annotation[0]] += area
        # print(areas)
        # break

    return areas


def test_run_annotations(filepath):
    data_2d_array = []

    # Read the file and split each line into elements
    with open(filepath, 'r') as file:
        for line in file:
            if line == "No valid detections\n":
                continue
            # Strip any extra whitespace or newlines and split by spaces
            row = line.strip().split()
            # Convert each string in the row to its appropriate type, e.g., float or int
            row = [int(row[0])] + [float(x) for x in row[1:]]

            # Append the row to the 2D array
            data_2d_array.append(row)

    # Now data_2d_array is a 2D array with each row as a list of numbers
    return data_2d_array


if __name__ == "__main__":
    # uvicorn.run(app, host="localhost", port=8080)
    annotations = test_run_annotations("/home/niranjan/documents/projects/hack-umass-2024/ZeroByte/yippeee.txt")
    araes = calculate_waste_percent(annotations)
    print(araes)



