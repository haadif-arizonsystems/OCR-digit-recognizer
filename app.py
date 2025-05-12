from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import io
import os
import shutil
from datetime import datetime

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

# Create uploads directory
uploads_dir = os.path.join(current_dir, "uploads")
os.makedirs(uploads_dir, exist_ok=True)

# Create a directory for preprocessed images
preprocessed_dir = os.path.join(current_dir, "preprocessed")
os.makedirs(preprocessed_dir, exist_ok=True)

model = YOLO("./weights/new-train-best.pt")

def preprocess_image(image_path):
    # Read the image in grayscale
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    
    # Convert grayscale to 3-channel BGR for prediction
    img_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    # Resize images to 640x640 (YOLO default size)
    img_resized = cv2.resize(img_3ch, (640, 640))
    gray_resized = cv2.resize(gray, (640, 640))  # resized grayscale for visualization
    
    return img_resized, gray_resized

@app.get('/', response_class=HTMLResponse)
async def home():
    with open(os.path.join(static_dir, "index.html"), "r") as f:
        return f.read()

@app.get('/health')
async def health():
    return 'App is Running'

@app.post("/predict/")
async def predict_digits(file: UploadFile = File(...)):
    # Save the uploaded file to the uploads directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_extension = os.path.splitext(file.filename)[1]
    file_name = f"{timestamp}{file_extension}"
    file_path = os.path.join(uploads_dir, file_name)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # Preprocess the saved image
        img_preprocessed, gray_resized = preprocess_image(file_path)
        
        # Save the preprocessed image for reference
        preprocessed_path = os.path.join(preprocessed_dir, f"prep_{file_name}")
        cv2.imwrite(preprocessed_path, img_preprocessed)
        
        # Run prediction on the preprocessed image
        results = model.predict(img_preprocessed, conf=0.5)[0]
        
        boxes = results.boxes
        names = results.names
        
        digits_with_x = [
            (int(cls.item()), box[0].item())  # (class_id, x1)
            for cls, box in zip(boxes.cls, boxes.xyxy)
        ]
        
        # Sort digits by their x-coordinate
        digits_sorted = sorted(digits_with_x, key=lambda d: d[1])
        digit_string = ''.join(names[d[0]] for d in digits_sorted)
        
        return JSONResponse(content={
            "digits": digit_string,
            "original_file": file_path,
            "preprocessed_file": preprocessed_path
        })
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Processing failed: {str(e)}"}
        )

# Add an endpoint to list all uploaded files
@app.get("/uploads/")
async def list_uploads():
    files = os.listdir(uploads_dir)
    return {"uploads": files}

# Add an endpoint to list all preprocessed files
@app.get("/preprocessed/")
async def list_preprocessed():
    files = os.listdir(preprocessed_dir)
    return {"preprocessed": files}
