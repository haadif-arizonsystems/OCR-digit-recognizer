# Project Structure
# yolo_mlops/
# ├── README.md
# ├── requirements.txt
# ├── .gitignore
# ├── config.yaml
# ├── run_pipeline.py
# ├── app/
# │   ├── __init__.py
# │   ├── main.py            # FastAPI application
# │   └── frontend/
# │       └── index.html     # Frontend code
# └── pipeline/
#     ├── __init__.py
#     ├── steps/
#     │   ├── __init__.py
#     │   ├── data_acquisition.py
#     │   ├── data_preprocessing.py
#     │   ├── model_training.py
#     │   ├── model_evaluation.py
#     │   └── model_deployment.py
#     └── pipeline.py        # Main pipeline definition

# requirements.txt
"""
fastapi==0.109.0
uvicorn==0.27.0
python-multipart==0.0.9
pillow==10.2.0
numpy==1.26.3
ultralytics==8.1.0
zenml==0.55.1
mlflow==2.10.0
roboflow==1.1.16
pydantic==2.5.3
python-dotenv==1.0.0
"""

# config.yaml
"""
roboflow:
  api_key: "your_roboflow_api_key"
  project_id: "your_project_id"
  version: 1
  format: "yolov8"
  
model:
  name: "yolov12"
  img_size: 640
  epochs: 100
  batch_size: 16
  conf_threshold: 0.5
  
mlflow:
  tracking_uri: "http://localhost:5000"
  experiment_name: "yolov12_digit_detection"
"""

# .gitignore
"""
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# Distribution / packaging
dist/
build/
*.egg-info/

# Unit test / coverage reports
htmlcov/
.tox/
.coverage
.coverage.*
.cache
coverage.xml
*.cover

# Jupyter Notebook
.ipynb_checkpoints

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# ZenML and MLflow
.zenml/
mlruns/
runs/
models/
.mlflow/

# Project specific
data/
weights/
logs/
*.pt
"""

# pipeline/steps/data_acquisition.py
import os
import yaml
from typing import Tuple, Dict, Any
from zenml import step
from roboflow import Roboflow
import logging

@step
def data_acquisition(
    config_path: str = "config.yaml"
) -> Tuple[str, Dict[str, Any]]:
    """
    Download dataset from Roboflow.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        dataset_path: Path to downloaded dataset
        dataset_info: Information about the dataset
    """
    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    roboflow_config = config["roboflow"]
    
    # Initialize Roboflow
    rf = Roboflow(api_key=roboflow_config["api_key"])
    project = rf.workspace().project(roboflow_config["project_id"])
    
    # Download dataset
    dataset_path = "data"
    dataset = project.version(roboflow_config["version"]).download(
        roboflow_config["format"],
        location=dataset_path
    )
    
    # Get dataset information
    dataset_info = {
        "dataset_path": dataset_path,
        "train_path": os.path.join(dataset_path, "train"),
        "val_path": os.path.join(dataset_path, "valid"),
        "test_path": os.path.join(dataset_path, "test"),
        "num_classes": dataset.instance_count,
        "classes": dataset.classes
    }
    
    logging.info(f"Dataset downloaded to {dataset_path}")
    logging.info(f"Dataset info: {dataset_info}")
    
    return dataset_path, dataset_info

# pipeline/steps/data_preprocessing.py
import os
import yaml
import shutil
from typing import Tuple, Dict, Any
from zenml import step
import logging

@step
def data_preprocessing(
    dataset_path: str,
    dataset_info: Dict[str, Any],
    config_path: str = "config.yaml"
) -> Tuple[str, Dict[str, Any]]:
    """
    Preprocess the dataset for YOLOv12 training.
    
    Args:
        dataset_path: Path to downloaded dataset
        dataset_info: Information about the dataset
        config_path: Path to configuration file
        
    Returns:
        processed_data_path: Path to processed data
        data_config: YAML configuration for YOLOv12
    """
    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Create processed data directory
    processed_data_path = os.path.join(dataset_path, "processed")
    os.makedirs(processed_data_path, exist_ok=True)
    
    # Create data.yaml for YOLOv12
    data_config = {
        "path": os.path.abspath(dataset_path),
        "train": os.path.relpath(dataset_info["train_path"], dataset_path),
        "val": os.path.relpath(dataset_info["val_path"], dataset_path),
        "test": os.path.relpath(dataset_info["test_path"], dataset_path),
        "nc": len(dataset_info["classes"]),
        "names": dataset_info["classes"]
    }
    
    # Write data.yaml
    data_config_path = os.path.join(processed_data_path, "data.yaml")
    with open(data_config_path, "w") as f:
        yaml.dump(data_config, f, default_flow_style=False)
    
    logging.info(f"Data preprocessing completed. Configuration saved to {data_config_path}")
    
    return processed_data_path, data_config

# pipeline/steps/model_training.py
import os
import yaml
import mlflow
from typing import Tuple, Dict, Any
from zenml import step
from ultralytics import YOLO
import logging

@step
def model_training(
    processed_data_path: str,
    data_config: Dict[str, Any],
    config_path: str = "config.yaml"
) -> Tuple[str, Dict[str, Any]]:
    """
    Train YOLOv12 model using the processed dataset.
    
    Args:
        processed_data_path: Path to processed data
        data_config: YAML configuration for YOLOv12
        config_path: Path to configuration file
        
    Returns:
        model_path: Path to trained model
        training_metrics: Training metrics
    """
    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    model_config = config["model"]
    mlflow_config = config["mlflow"]
    
    # Configure MLflow
    mlflow.set_tracking_uri(mlflow_config["tracking_uri"])
    mlflow.set_experiment(mlflow_config["experiment_name"])
    
    # Start MLflow run
    with mlflow.start_run(run_name="yolov12_training"):
        # Log parameters
        mlflow.log_params({
            "model": model_config["name"],
            "img_size": model_config["img_size"],
            "epochs": model_config["epochs"],
            "batch_size": model_config["batch_size"],
            "num_classes": data_config["nc"]
        })
        
        # Initialize model
        model = YOLO(model_config["name"])
        
        # Train model
        data_config_path = os.path.join(processed_data_path, "data.yaml")
        results = model.train(
            data=data_config_path,
            epochs=model_config["epochs"],
            imgsz=model_config["img_size"],
            batch=model_config["batch_size"],
            name="yolov12_run"
        )
        
        # Get model path
        model_path = os.path.join("runs", "detect", "yolov12_run", "weights", "best.pt")
        
        # Log metrics
        training_metrics = {
            "precision": results.results_dict["metrics/precision(B)"],
            "recall": results.results_dict["metrics/recall(B)"],
            "mAP50": results.results_dict["metrics/mAP50(B)"],
            "mAP50-95": results.results_dict["metrics/mAP50-95(B)"]
        }
        
        mlflow.log_metrics(training_metrics)
        
        # Log model
        mlflow.log_artifact(model_path, "model")
        
        logging.info(f"Model training completed. Model saved to {model_path}")
        logging.info(f"Training metrics: {training_metrics}")
    
    return model_path, training_metrics

# pipeline/steps/model_evaluation.py
import os
import yaml
import mlflow
from typing import Tuple, Dict, Any
from zenml import step
from ultralytics import YOLO
import logging

@step
def model_evaluation(
    model_path: str,
    data_config: Dict[str, Any],
    config_path: str = "config.yaml"
) -> Tuple[float, Dict[str, Any]]:
    """
    Evaluate the trained YOLOv12 model.
    
    Args:
        model_path: Path to trained model
        data_config: YAML configuration for YOLOv12
        config_path: Path to configuration file
        
    Returns:
        model_score: Model evaluation score (mAP50-95)
        eval_metrics: Evaluation metrics
    """
    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    model_config = config["model"]
    mlflow_config = config["mlflow"]
    
    # Configure MLflow
    mlflow.set_tracking_uri(mlflow_config["tracking_uri"])
    mlflow.set_experiment(mlflow_config["experiment_name"])
    
    # Start MLflow run
    with mlflow.start_run(run_name="yolov12_evaluation"):
        # Load model
        model = YOLO(model_path)
        
        # Evaluate model
        test_path = os.path.join(data_config["path"], data_config["test"])
        results = model.val(
            data=test_path,
            imgsz=model_config["img_size"],
            batch=model_config["batch_size"],
            conf=model_config["conf_threshold"]
        )
        
        # Extract metrics
        eval_metrics = {
            "precision": results.results_dict["metrics/precision(B)"],
            "recall": results.results_dict["metrics/recall(B)"],
            "mAP50": results.results_dict["metrics/mAP50(B)"],
            "mAP50-95": results.results_dict["metrics/mAP50-95(B)"],
            "conf_threshold": model_config["conf_threshold"]
        }
        
        # Log metrics
        mlflow.log_metrics(eval_metrics)
        
        # Use mAP50-95 as the model score
        model_score = eval_metrics["mAP50-95"]
        
        logging.info(f"Model evaluation completed. Score: {model_score}")
        logging.info(f"Evaluation metrics: {eval_metrics}")
    
    return model_score, eval_metrics

# pipeline/steps/model_deployment.py
import os
import shutil
import yaml
import mlflow
from typing import Dict, Any
from zenml import step
import logging

@step
def model_deployment(
    model_path: str,
    model_score: float,
    eval_metrics: Dict[str, Any],
    config_path: str = "config.yaml"
) -> str:
    """
    Deploy the trained YOLOv12 model.
    
    Args:
        model_path: Path to trained model
        model_score: Model evaluation score
        eval_metrics: Evaluation metrics
        config_path: Path to configuration file
        
    Returns:
        deployment_path: Path to deployed model
    """
    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    deployment_dir = "weights"
    os.makedirs(deployment_dir, exist_ok=True)
    
    # Copy model to deployment directory
    deployment_path = os.path.join(deployment_dir, "best.pt")
    shutil.copy(model_path, deployment_path)
    
    # Create model metadata
    model_metadata = {
        "model_name": config["model"]["name"],
        "model_version": mlflow.active_run().info.run_id[:8],
        "model_score": model_score,
        "evaluation_metrics": eval_metrics,
        "conf_threshold": config["model"]["conf_threshold"]
    }
    
    # Save metadata
    metadata_path = os.path.join(deployment_dir, "metadata.yaml")
    with open(metadata_path, "w") as f:
        yaml.dump(model_metadata, f, default_flow_style=False)
    
    logging.info(f"Model deployed to {deployment_path}")
    logging.info(f"Model metadata saved to {metadata_path}")
    
    return deployment_path

# pipeline/pipeline.py
from zenml import pipeline
from zenml.config import DockerSettings
from .steps.data_acquisition import data_acquisition
from .steps.data_preprocessing import data_preprocessing
from .steps.model_training import model_training
from .steps.model_evaluation import model_evaluation
from .steps.model_deployment import model_deployment

@pipeline(enable_cache=False, settings={"docker": DockerSettings(required_integrations=["mlflow"])})
def yolo_training_pipeline(config_path: str = "config.yaml"):
    """
    Complete YOLOv12 training pipeline with MLflow tracking.
    
    Args:
        config_path: Path to configuration file
    """
    # Step 1: Acquire data from Roboflow
    dataset_path, dataset_info = data_acquisition(config_path)
    
    # Step 2: Preprocess data for YOLOv12
    processed_data_path, data_config = data_preprocessing(dataset_path, dataset_info, config_path)
    
    # Step 3: Train YOLOv12 model
    model_path, training_metrics = model_training(processed_data_path, data_config, config_path)
    
    # Step 4: Evaluate YOLOv12 model
    model_score, eval_metrics = model_evaluation(model_path, data_config, config_path)
    
    # Step 5: Deploy YOLOv12 model
    deployment_path = model_deployment(model_path, model_score, eval_metrics, config_path)
    
    return deployment_path

# run_pipeline.py
import os
import yaml
import logging
from zenml.repository import Repository
from pipeline.pipeline import yolo_training_pipeline

def main():
    """
    Main function to run the YOLOv12 training pipeline.
    """
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()]
    )
    
    # Initialize ZenML repository
    repo = Repository()
    
    # Check if config.yaml exists
    config_path = "config.yaml"
    if not os.path.exists(config_path):
        logging.error(f"Configuration file {config_path} not found.")
        return
    
    # Run pipeline
    logging.info("Starting YOLOv12 training pipeline...")
    yolo_training_pipeline(config_path=config_path)
    logging.info("Pipeline completed successfully!")

if __name__ == "__main__":
    main()

# app/main.py
import os
import yaml
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import numpy as np
import io
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# Initialize FastAPI app
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
static_dir = os.path.join(current_dir, "frontend")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Load model
model_path = os.path.join("weights", "best.pt")
if not os.path.exists(model_path):
    logging.warning(f"Model not found at {model_path}. Please run the training pipeline first.")
    model = None
else:
    model = YOLO(model_path)
    logging.info(f"Model loaded from {model_path}")

# Load model metadata
metadata_path = os.path.join("weights", "metadata.yaml")
if os.path.exists(metadata_path):
    with open(metadata_path, "r") as f:
        model_metadata = yaml.safe_load(f)
    conf_threshold = model_metadata.get("conf_threshold", 0.5)
    logging.info(f"Model metadata loaded from {metadata_path}")
else:
    conf_threshold = 0.5
    logging.warning(f"Model metadata not found at {metadata_path}. Using default confidence threshold: {conf_threshold}")

@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the frontend HTML page."""
    html_file = os.path.join(static_dir, "index.html")
    if os.path.exists(html_file):
        with open(html_file, "r") as f:
            return f.read()
    else:
        return "<html><body><h1>App is Running</h1><p>Frontend not found. Make sure to place index.html in the frontend directory.</p></body></html>"

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict/")
async def predict_digits(file: UploadFile = File(...)):
    """
    Predict digits from an uploaded image.
    
    Args:
        file: Uploaded image file
        
    Returns:
        JSON response with predicted digits
    """
    # Check if model is loaded
    if model is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Model not loaded. Please run the training pipeline first."}
        )
    
    try:
        # Read and process image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_np = np.array(image)
        
        # Run inference
        results = model.predict(img_np, conf=conf_threshold)[0]
        
        # Process results
        boxes = results.boxes
        digits_with_x = [
            (int(cls.item()), box[0].item())  # (class_id, x1)
            for cls, box in zip(boxes.cls, boxes.xyxy)
        ]
        
        # Sort digits by x-coordinate
        digits_sorted = sorted(digits_with_x, key=lambda d: d[1])
        digit_string = ''.join(str(d[0]) for d in digits_sorted)
        
        return JSONResponse(content={"digits": digit_string})
    
    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error processing image: {str(e)}"}
        )

# app/frontend/index.html
"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Digit Prediction App</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .container {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 20px;
        }
        .upload-section {
            width: 100%;
            border: 2px dashed #ccc;
            padding: 20px;
            text-align: center;
            border-radius: 5px;
            background-color: #f9f9f9;
        }
        .upload-btn {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            margin-top: 10px;
        }
        .upload-btn:hover {
            background-color: #45a049;
        }
        .preview-section {
            display: none;
            width: 100%;
            margin-top: 20px;
        }
        .image-preview {
            max-width: 100%;
            max-height: 300px;
            margin: 10px auto;
            display: block;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        .result-section {
            display: none;
            width: 100%;
            padding: 20px;
            background-color: #e9f7ef;
            border-radius: 5px;
            text-align: center;
        }
        .loading {
            display: none;
            text-align: center;
            margin: 20px 0;
        }
        .spinner {
            border: 4px solid rgba(0, 0, 0, 0.1);
            width: 36px;
            height: 36px;
            border-radius: 50%;
            border-left-color: #4CAF50;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .error-message {
            display: none;
            color: #f44336;
            text-align: center;
            margin: 10px 0;
        }
    </style>
</head>
<body>
    <h1>Digit Prediction App</h1>
    <div class="container">
        <div class="upload-section">
            <h2>Upload an Image</h2>
            <p>Select an image containing digits to get the prediction</p>
            <input type="file" id="imageInput" accept="image/*" style="display: none;">
            <button class="upload-btn" onclick="document.getElementById('imageInput').click()">Choose File</button>
            <p id="fileName">No file chosen</p>
        </div>
        
        <div class="loading" id="loadingSpinner">
            <div class="spinner"></div>
            <p>Processing...</p>
        </div>
        
        <div class="error-message" id="errorMessage"></div>
        
        <div class="preview-section" id="previewSection">
            <h2>Image Preview</h2>
            <img id="imagePreview" class="image-preview" src="" alt="Image preview">
            <button class="upload-btn" id="predictBtn">Predict Digits</button>
        </div>
        
        <div class="result-section" id="resultSection">
            <h2>Prediction Result</h2>
            <p>The predicted digits are: <strong id="predictedDigits"></strong></p>
        </div>
    </div>

    <script>
        // DOM elements
        const imageInput = document.getElementById('imageInput');
        const fileName = document.getElementById('fileName');
        const imagePreview = document.getElementById('imagePreview');
        const previewSection = document.getElementById('previewSection');
        const predictBtn = document.getElementById('predictBtn');
        const resultSection = document.getElementById('resultSection');
        const predictedDigits = document.getElementById('predictedDigits');
        const loadingSpinner = document.getElementById('loadingSpinner');
        const errorMessage = document.getElementById('errorMessage');
        
        // Selected file for upload
        let selectedFile = null;
        
        // Handle file selection
        imageInput.addEventListener('change', function(event) {
            selectedFile = event.target.files[0];
            
            if (selectedFile) {
                fileName.textContent = selectedFile.name;
                
                // Display image preview
                const reader = new FileReader();
                reader.onload = function(e) {
                    imagePreview.src = e.target.result;
                    previewSection.style.display = 'block';
                    resultSection.style.display = 'none';
                    errorMessage.style.display = 'none';
                };
                reader.readAsDataURL(selectedFile);
            } else {
                fileName.textContent = 'No file chosen';
                previewSection.style.display = 'none';
            }
        });
        
        // Handle prediction
        predictBtn.addEventListener('click', async function() {
            if (!selectedFile) {
                showError('Please select an image first');
                return;
            }
            
            // Show loading spinner
            loadingSpinner.style.display = 'block';
            errorMessage.style.display = 'none';
            
            try {
                const formData = new FormData();
                formData.append('file', selectedFile);
                
                const response = await fetch('/predict/', {
                    method: 'POST',
                    body: formData
                });
                
                if (!response.ok) {
                    throw new Error('Server error: ' + response.status);
                }
                
                const data = await response.json();
                
                // Display results
                predictedDigits.textContent = data.digits;
                resultSection.style.display = 'block';
            } catch (error) {
                showError('Error: ' + error.message);
            } finally {
                loadingSpinner.style.display = 'none';
            }
        });
        
        // Show error message
        function showError(message) {
            errorMessage.textContent = message;
            errorMessage.style.display = 'block';
            loadingSpinner.style.display = 'none';
        }
    </script>
</body>
</html>

"""

# README.md
"""
# YOLOv12 Digit Detection MLOps Pipeline

This project implements a complete MLOps pipeline for training and deploying a YOLOv12 model for digit detection, using ZenML and MLflow.

## Project Structure

```
yolo_mlops/
├── README.md
├── requirements.txt
├── .gitignore
├── config.yaml
├── run_pipeline.py
├── app/
│   ├── __init__.py
│   ├── main.py            # FastAPI application
│   └── frontend/
│       └── index.html     # Frontend code
└── pipeline/
    ├── __init__.py
    ├── steps/
    │   ├── __init__.py
    │   ├── data_acquisition.py
    │   ├── data_preprocessing.py
    │   ├── model_training.py
    │   ├── model_evaluation.py
    │   └── model_deployment.py
    └── pipeline.py        # Main pipeline definition
```

## Setup

1. Clone the repository
2. Install requirements:
   ```
   pip install -r requirements.txt
   ```
3. Initialize ZenML:
   ```
   zenml init
   ```
4. Set up MLflow tracking server:
   ```
   mlflow server --host 0.0.0.0 --port 5000
   ```
5. Configure `config.yaml` with your Roboflow API key and other settings

## Running the Pipeline

Execute the pipeline with:
```
python run_pipeline.py
```

The pipeline will:
1. Download the dataset from Roboflow
2. Preprocess the data for YOLOv12
3. Train the model with MLflow tracking
4. Evaluate the model performance
5. Deploy the model for inference

## Serving the Model

Start the FastAPI server:
```
uvicorn app.main:app --reload
```

Access the web interface at http://localhost:8000

## Monitoring and Tracking

Access MLflow UI at http://localhost:5000 to monitor experiments, compare runs, and analyze model performance.

## Pipeline Steps

1. **Data Acquisition**: Downloads data from Roboflow
2. **Data Preprocessing**: Prepares data for YOLOv12 training
3. **Model Training**: Trains YOLOv12 model with MLflow tracking
4. **Model Evaluation**: Evaluates model performance
5. **Model Deployment**: Deploys model for serving
"""
