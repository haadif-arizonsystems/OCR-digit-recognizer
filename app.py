from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
import numpy as np
import io

app = FastAPI()

model = YOLO("weights/best.pt")

@app.post("g")
async def predict_digits(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_np = np.array(image)

    results = model.predict(img_np, conf=0.5)[0]
    boxes = results.boxes
    digits_with_x = [
        (int(cls.item()), box[0].item())  # (class_id, x1)
        for cls, box in zip(boxes.cls, boxes.xyxy)
    ]

    digits_sorted = sorted(digits_with_x, key=lambda d: d[1])
    digit_string = ''.join(str(d[0]) for d in digits_sorted)

    return JSONResponse(content={"digits": digit_string})
