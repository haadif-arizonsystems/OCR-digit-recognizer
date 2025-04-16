from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import numpy as np
import io
import os

app = FastAPI(docs_url=None, redoc_url=None)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory
current_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(current_dir, "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

model = YOLO("weights/best.pt")

@app.get('/', response_class=HTMLResponse)
async def home():
    with open(os.path.join(static_dir, "index.html"), "r") as f:
        return f.read()

@app.get('/health')
async def health():
    return 'App is Running'

@app.post("/predict/")
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