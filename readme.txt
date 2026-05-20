# Tongue Analysis Flask API

A production‑ready Flask API for automated tongue diagnosis using five complementary deep learning / computer vision models. The API accepts a tongue image and returns predictions for **colour**, **texture**, **shape**, **cracks**, **coating**, and an overall **confidence score**.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
  - [Virtual Environment](#virtual-environment)
  - [Dependencies](#dependencies)
- [Model Files](#model-files)
- [Running the API](#running-the-api)
- [API Endpoints](#api-endpoints)
  - [Health Check](#health-check)
  - [Analyse Image](#analyse-image)
- [Response Format](#response-format)
- [Confidence Score](#confidence-score)
- [Error Handling](#error-handling)
- [Testing](#testing)
- [Deployment](#deployment)
- [Customisation](#customisation)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Overview

This API combines five specialised models to analyse tongue images:

| Task      | Model                     | Output classes                                      |
|-----------|---------------------------|-----------------------------------------------------|
| Colour    | EfficientNet‑V2 + HSV     | pink, red, indigo_violet, purple, white             |
| Texture   | ResNet‑50                 | normal, tender, tough                               |
| Shape     | EfficientNet‑B2 + ML ensemble | normal, oval, rectangle                          |
| Cracks    | ResNet‑50 binary          | cracked, non_cracked                                |
| Coating   | HSV blob detection        | white, none                                         |

All models are loaded once at startup. The API returns a unified JSON response.

## Features

- Multi‑modal tongue analysis
- Base64 thumbnail of uploaded image included in response
- Confidence score with descriptive band (Almost Perfect, Substantial, etc.)
- Graceful error handling – partial results returned if one model fails
- Easy to extend or replace individual models

## Project Structure
tongue_api/
├── app.py # Main Flask application
├── requirements.txt # Python dependencies
├── README.md # This file
├── models/ # Trained model files (not included in repo)
│ ├── efficientnetv2_tongue_color.pth
│ ├── class_names.json
│ ├── best_tongue_model.pth
│ ├── cracked_binary_resnet50.pth
│ ├── tongue_cnn_v3.pth
│ ├── tongue_ml_ensemble.pkl
│ └── confidence_lr.pkl # optional
└── utils/ # Helper modules
├── init.py
├── color_utils.py
├── texture_utils.py
├── shape_utils.py
├── crack_utils.py
└── image_utils.py

text

## Requirements

- Python 3.10 or higher
- pip

## Installation

### Virtual Environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate          # Linux / macOS
.venv\Scripts\activate             # Windows
Dependencies
Install PyTorch first (choose CPU or GPU version):

CPU version (works on any machine):

bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
GPU version (CUDA 11.8) :

bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
Then install the remaining packages:

bash
pip install -r requirements.txt
Contents of requirements.txt:

text
flask==2.3.3
timm==0.9.2
opencv-python-headless==4.8.1.78
pillow==10.0.0
numpy==1.24.3
scikit-learn==1.3.0
joblib==1.3.2
Model Files
Place the trained model files inside the models/ folder. The API expects exact filenames as listed below. If a file is missing, the corresponding prediction will return {"label": "error"}.

Filename	Description	Source notebook
efficientnetv2_tongue_color.pth	Colour EfficientNet‑V2	efficientnetv2_tongue_color_30epochs_pth.ipynb
class_names.json	Colour class names	same notebook
best_tongue_model.pth	Texture ResNet‑50	tongue_texture_training.ipynb
cracked_binary_resnet50.pth	Cracks ResNet‑50	crack_vs_non_cracked_training.ipynb
tongue_cnn_v3.pth	Shape EfficientNet‑B2	tongue_shape.ipynb
tongue_ml_ensemble.pkl	Shape ML ensemble (RF+GBM+SVM)	tongue_shape.ipynb
confidence_lr.pkl	(optional) Confidence LR	custom training
Note: If confidence_lr.pkl is not provided, the API falls back to a heuristic confidence score.

Running the API
bash
python app.py
The server will start at http://0.0.0.0:5000. Use Ctrl+C to stop.

API Endpoints
Health Check
Endpoint: GET /health

Response:

json
{
  "status": "ok",
  "device": "cpu"
}
Analyse Image
Endpoint: POST /analyze

Request: multipart/form-data with field image (upload a tongue image – supported formats: jpg, jpeg, png, bmp, webp).

Response: JSON object with all predictions.

Response Format
json
{
  "status": "success",
  "uploaded_image": "base64_encoded_thumbnail",
  "color": { "label": "pink" },
  "texture": { "label": "normal" },
  "shape": { "label": "oval" },
  "cracks": { "label": "non_cracked" },
  "coating": { "label": "none" },
  "confidence_score": {
    "score": 0.85,
    "band": "Substantial"
  }
}
uploaded_image – Base64‑encoded JPEG thumbnail (max 800px) of the uploaded image.

Each prediction is a dictionary with at least a "label" key; some also include probabilities (not shown above).

Confidence Score
The confidence score (0–1) is computed from the five predictions using either:

A trained logistic regression model (confidence_lr.pkl), if present.

A heuristic weighted average (fallback).

Bands:

Score range	Band
0.8 – 1.0	Almost Perfect
0.6 – 0.8	Substantial
0.4 – 0.6	Moderate
0.2 – 0.4	Fair
0.0 – 0.2	Poor
Error Handling
If any model fails to load or raises an exception during inference, the API returns {"label": "error"} for that attribute. Other attributes are still processed. The overall request will still return "status": "success" as long as the image was read successfully.

Testing
Using cURL
bash
curl -X POST -F "image=@/path/to/tongue.jpg" http://localhost:5000/analyze
Using Python
python
import requests

with open("tongue.jpg", "rb") as f:
    files = {"image": f}
    resp = requests.post("http://localhost:5000/analyze", files=files)
    print(resp.json())
Deployment
For production, use a WSGI server like gunicorn (Linux) or waitress (Windows).

Example with gunicorn:

bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
Example with waitress (Windows):

bash
waitress-serve --port=5000 app:app
Set environment variables for model paths if needed, and ensure debug=False in app.py (already set).

Customisation
Colour refinement – Edit refine_color() in utils/color_utils.py to adjust HSV post‑processing rules.

Confidence LR – Train your own logistic regression and replace confidence_lr.pkl in models/.

Model paths – Modify the file paths inside each utility module (color_utils.py, etc.) if your models are stored elsewhere.

Troubleshooting
Error	Likely cause	Solution
FileNotFoundError for a .pth file	Model file missing or wrong filename	Place correct file in models/ and ensure name matches code
ModuleNotFoundError: No module named 'utils'	utils/__init__.py missing	Create an empty __init__.py inside the utils/ folder
CUDA out of memory	GPU memory insufficient	Reduce batch size (not applicable in inference) or force CPU by changing DEVICE to cpu
Slow inference on CPU	CPU only	Use GPU if available, or reduce input size in image_utils.encode_image()
License
MIT – free for academic and commercial use.

